import logging
import math
import os
import random
from pathlib import Path
from typing import Iterable, Optional
from tqdm.auto import tqdm
import json
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from torchvision import transforms

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from transformers import CLIPTextModel, CLIPTokenizer

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from model import model_types
from config import parse_args
from utils_model import save_model, load_model
from utils_data import get_dataloader, get_test_data

import wandb 
from PIL import Image

logger = get_logger(__name__)


def unfreeze_layers_unet(unet):
    print("Num trainable params unet: ", sum(p.numel() for p in unet.parameters() if p.requires_grad))
    return unet


def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, scheduler, epoch):
    logger.info("Running validation... ")

    device=torch.device('cuda')

    model=StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    model=model.to(device)
    model.set_progress_bar_config(disable=True)

    def predict_cond(model, prompt, seed, condition, img_size):
        generator = torch.Generator("cuda").manual_seed(seed)
        output = model(prompt=prompt, height=img_size, width=img_size, num_inference_steps=20, generator=generator, controlnet_cond=condition)
        image = output[0][0]
        return image

    test_dataloader=get_test_data(data_dir=args.train_data_dir)

    images=[]
    for unique in range(2):
        seed=unique+1023123789
        for prompt, concept in zip(*test_dataloader):
            images.append(predict_cond(model=model, prompt=prompt, seed=seed, condition=concept, img_size=args.resolution))
    
    for tracker in accelerator.trackers:
        if tracker.name  == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        if tracker.name == 'wandb':
            images = [np.array(image) for image in images]
            images = np.concatenate(images, axis=1)
            pil_image = Image.fromarray(images)
            downsample_factor = 8
            downsample_image = pil_image.resize((pil_image.size[0]//downsample_factor, pil_image.size[1]//downsample_factor))
            tracker.log({"validation_images": wandb.Image(downsample_image)})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del model
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    yaml = YAML()
    yaml.dump(vars(args), open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    if args.use_esd:
        load_model(unet, 'baselines/diffusers-nudity-ESDu1-UNET.pt')

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    mlp=model_types[args.model_type](resolution=args.resolution//64)
    unet.set_controlnet(mlp)
    unet = unfreeze_layers_unet(unet)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer = torch.optim.Adam(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")


    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        input_ids = inputs.input_ids
        return input_ids

    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def collate_fn(examples):
        pixel_values = torch.stack([example[0] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = [example[1] for example in examples]
        padded_tokens = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
        input_conditions = torch.stack([example[2] for example in examples])
        return {
            "pixel_values": pixel_values,
            "input_ids": padded_tokens.input_ids,
            "attention_mask": padded_tokens.attention_mask,
            "input_conditions": input_conditions,
        }

    train_dataloader = get_dataloader(args.train_data_dir, batch_size=args.train_batch_size, shuffle=True,
                                      transform=train_transforms, tokenizer=tokenize_captions, collate_fn=collate_fn,
                                      num_workers=4, max_concept_length=100, select=args.select)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    print('weight_dtype',weight_dtype)

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        exp_name = f'{args.output_dir}_prompt_{args.prompt}_lr{str(args.learning_rate)}'
        accelerator.init_trackers(
            project_name="diffusion-explainer", 
            config={k:v for k,v in vars(args).items() if k!='config'},
            init_kwargs={"wandb": {"name": exp_name}}
            )


    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    device=torch.device("cuda")
    print("Start training")
    loss_history=[]
    train_loss = 0.0
    curious_time=0
    global_step = 0
    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            latents = vae.encode(batch["pixel_values"].to(weight_dtype).to(device)).latent_dist.sample()
            latents = latents * 0.18215

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, controlnet_cond=batch["input_conditions"].to(device)).sample
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            train_loss += loss.item()
            curious_time += timesteps.sum().item()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


            progress_bar.update(1)
            global_step += 1
            if global_step%1==0:
                train_loss = train_loss/1
                accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                loss_history.append(train_loss)
                train_loss = 0.0
                curious_time = 0

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

            if not args.skip_evaluation and (global_step)%args.log_every_steps==0:
                save_model(unet, args.output_dir+'/unet.pth')
                plt.figure()
                plt.plot(loss_history)
                plt.savefig(args.output_dir+'/loss_history.png')
                plt.close()

        if epoch%args.log_every_epochs==0:
            log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, noise_scheduler, epoch)
            save_model(unet, args.output_dir+'/unet.pth')


    save_model(unet, args.output_dir+'/unet.pth')
    plt.figure()
    plt.plot(loss_history)
    plt.savefig(args.output_dir+'/loss_history.png')
    plt.close()

if __name__ == "__main__":
    main()

import logging
import os
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPProcessor, CLIPModel

from model import model_types
from config import parse_args
from utils_model import save_model, load_model
from utils_data import get_dataloader, get_test_data, get_i2p_data

from PIL import Image
import clip


def unfreeze_layers_unet(unet, condition):
    print("Num trainable params unet: ", sum(p.numel() for p in unet.parameters() if p.requires_grad))
    return unet


def cvtImg(img):
    img = img.permute([0, 2, 3, 1])
    img = img - img.min()
    img = (img / img.max())
    return img.numpy().astype(np.float32)

def show_examples(x):
    plt.figure(figsize=(10, 10))
    imgs = cvtImg(x)
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(imgs[i])
        plt.axis('off')

def show_examples(x):
    plt.figure(figsize=(10, 5),dpi=200)
    imgs = cvtImg(x)
    for i in range(8):
        plt.subplot(1, 8, i+1)
        plt.imshow(imgs[i])
        plt.axis('off')

def show_images(images):
    images = [np.array(image) for image in images]
    images = np.concatenate(images, axis=1)
    return Image.fromarray(images)

def prompt_with_template(profession, template):
    profession = profession.lower()
    custom_prompt = template.replace("{{placeholder}}", profession)
    return custom_prompt


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    yaml = YAML()
    yaml.dump(vars(args), open(os.path.join(args.output_dir, 'test_config.yaml'), 'w'))

    # Load models and create wrapper for stable diffusion
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

    if args.scheduler == 'ddim':
        scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, 
            beta_schedule="scaled_linear", 
            clip_sample=False, 
            set_alpha_to_one=False,
            num_train_timesteps=1000,
            steps_offset=1,
        )
    elif args.scheduler == 'pndm':
        scheduler = PNDMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="scheduler"
        )
    elif args.scheduler == 'ddpm':
        scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler"
        )
    else:
        raise NotImplementedError(args.scheduler)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    mlp=model_types[args.model_type](resolution=args.resolution//64)
    unet.set_controlnet(mlp)
    load_model(unet, args.output_dir+'/unet.pth')

    device=torch.device('cuda')

    if args.use_sld:
        print("Using SLDPipeline")
        from baselines.sld_pipeline import SLDPipeline
        model=SLDPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            # requires_safety_checker=False,
        )
    else:
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
    if args.fp16:
        print('Using fp16')
        model.unet=model.unet.half()
        model.vae=model.vae.half()
        model.text_encoder=model.text_encoder.half()

    if args.evaluation_type=="inference_i2p":
        dataloader=get_i2p_data(data_dir=args.train_data_dir, given_prompt=args.prompt, given_concept=args.concept, max_concept_length=100)
    else:
        dataloader=get_test_data(data_dir=args.train_data_dir, given_prompt=args.prompt, given_concept=args.concept, max_concept_length=100)

    if args.evaluation_type=="eval":
        evaluate(model=model, dataloader=dataloader, device=device, args=args)
    elif args.evaluation_type=="interpolate":
        evaluate_interpolate(model=model, dataloader=dataloader, device=device, args=args)
    elif args.evaluation_type=="winobias":
        evaluate_inference_winobias(model=model, dataloader=dataloader, device=device, args=args)
    elif args.evaluation_type=="i2p":
        evaluate_inference_i2p(model=model, dataloader=dataloader, device=device, args=args)
    else:
        raise NotImplementedError(args.evaluation_type)


def predict_cond(model, 
                 prompt, 
                 seed, 
                 condition, 
                 img_size,
                 num_inference_steps=50,
                 interpolator=None, 
                 negative_prompt=None,
                 ):
    
    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
    output = model(prompt=prompt, height=img_size, width=img_size, 
                   num_inference_steps=num_inference_steps, 
                   generator=generator, 
                   controlnet_cond=condition,
                   controlnet_interpolator=interpolator,
                   negative_prompt=negative_prompt,
                   )
    image = output[0][0]
    return image


def evaluate(model, dataloader, device, args):
    save_image_dir=args.output_dir+'/'+args.image_dir
    os.makedirs(save_image_dir, exist_ok=True)
    for j in range(args.num_test_samples):
        images=[]
        seed=j
        for prompt, concept in zip(*dataloader):
            images.append(predict_cond(model=model, prompt=prompt, seed=seed, condition=concept, img_size=args.resolution,
                                        num_inference_steps=args.num_inference_steps,
                                        negative_prompt=args.negative_prompt,
                                       ))

        images=show_images(images)
        images.save(f"{save_image_dir}/eval{j}.jpg")


def evaluate_interpolate(model, dataloader, device, args):
    save_image_dir=args.output_dir+'/'+args.image_dir
    os.makedirs(save_image_dir, exist_ok=True)
    for j in range(args.num_test_samples):
        images=[]
        seed=j
        for prompt, concept in zip(*dataloader):
            if concept is not None:
                for z in np.linspace(0,1,11):
                    images.append(predict_cond(model=model, prompt=prompt, seed=seed, condition=concept, img_size=args.resolution, 
                                                interpolator=lambda x,y: x+y*z,
                                                num_inference_steps=args.num_inference_steps,
                                                negative_prompt=args.negative_prompt,
                                               ))
            else:
                images.append(predict_cond(model=model, prompt=prompt, seed=seed, condition=None, img_size=args.resolution,
                                        num_inference_steps=args.num_inference_steps,
                                        negative_prompt=args.negative_prompt,
                                           ))

        images=show_images(images)
        images.save(f"{save_image_dir}/inter{j}.jpg")


def evaluate_inference_winobias(model, dataloader, device, args):
    from metrics.CLIP_classify import CLIP_classification_function, add_winobias_metrics
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)
    
    seed=None
    logging = []
    root_dir = os.path.join(args.output_dir, 'winobias') if not args.original_sd else os.path.join(args.output_dir, 'winobias_original_sd')
        
    root_dir = os.path.join(*[root_dir, args.image_dir])
    root_dir = os.path.join(root_dir, f'template{str(args.template_key)}')
    print(f'images saved to: {root_dir}')
    
    from winobias_cfg import professions, templates
    for profession in professions:  
        save_image_dir = os.path.join(root_dir, profession)
        os.makedirs(save_image_dir, exist_ok=True)
        global_id=0
        template_lst = templates[args.template_key]
        prompts = [prompt_with_template(profession, temp) for temp in template_lst]
        for prompt in prompts:
            print(f'creating images with prompt: {prompt}')
            for j in range(args.num_test_samples):
                for _, concept in zip(*dataloader):
                    if args.original_sd:
                        image=predict_cond(model=model, prompt=prompt, seed=seed, condition=None, img_size=args.resolution,
                                            num_inference_steps=args.num_inference_steps,
                                            negative_prompt=args.negative_prompt,
                                           )
                        image.save(f"{save_image_dir}/{global_id}.jpg")
                        
                    else:
                        if concept is not None:
                            image=predict_cond(model=model, prompt=prompt, seed=seed, condition=concept, img_size=args.resolution,
                                                num_inference_steps=args.num_inference_steps,
                                                negative_prompt=args.negative_prompt,
                                                )
                            image.save(f"{save_image_dir}/{global_id}.jpg")
                    global_id+=1
                
        df = CLIP_classification_function(save_image_dir, args.clip_attributes, model=clip_model, processor=processor, return_df=True)
        result = {'profession': profession}
        sums = df.sum().to_dict()
        result.update(sums)
        logging.append(result)
        print(result)

    logging = pd.DataFrame(logging)
    logging = add_winobias_metrics(logging.set_index('profession'))
    save_name = '_'.join([s.replace(' ', '_') for s in args.clip_attributes])
    save_name += '_result.csv'
    save_path = os.path.join(root_dir, save_name)
    logging.to_csv(save_path, index=True)
    print(f'CLIP classification results saved to {save_path}')


def evaluate_inference_i2p(model, dataloader, device, args):
    seed=None
    save_image_dir=args.output_dir+'/'+args.image_dir
    os.makedirs(save_image_dir, exist_ok=True)
    labels=[]
    
    model.set_progress_bar_config(disable=True)
    global_id=0
    for j in range(args.num_test_samples):
        for prompt, concept, concept_str in tqdm(dataloader, total=len(dataloader)*(args.num_test_samples-j)):
            image=predict_cond(model=model, prompt=prompt, seed=seed, condition=concept, img_size=args.resolution,
                                num_inference_steps=args.num_inference_steps,
                                negative_prompt=args.negative_prompt,
                               )
            image.save(f"{save_image_dir}/{global_id}.jpg")
            global_id+=1
            labels.append([prompt, concept_str])

    prompts, labels = list(zip(*labels))
    from metrics.nudenet_classify import detect_nude_and_q16
    predictions=detect_nude_and_q16(folder=save_image_dir)
    logging=pd.DataFrame({'prompt':prompts, 'label':labels, 'prediction':predictions})
    logging.to_csv(save_image_dir + '/i2p.csv', index=False)
    stats=pd.read_csv(save_image_dir + '/i2p.csv').groupby("label").prediction.mean()
    stats.to_csv(save_image_dir + '/i2p_stats.csv', index=True)


if __name__ == "__main__":
    main()

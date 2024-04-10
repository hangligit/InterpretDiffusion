import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, Optional


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument("--revision",type=str,default=None,required=False,)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default=None,help="The directory where the downloaded models and datasets will be stored.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--center_crop", action="store_true",help="Whether to center crop images before resizing to resolution (if not set, random crop will be used)",)
    parser.add_argument("--random_flip", action="store_true", help="whether to randomly flip images horizontally",)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision",type=str,default=None,choices=["no", "fp16", "bf16"],)
    parser.add_argument("--report_to",type=str,default="wandb",)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--max_train_steps",type=int,default=None,)
    parser.add_argument("--gradient_checkpointing", action="store_true",)

    parser.add_argument("--train_data_dir", type=str, default="datasets/person")
    parser.add_argument("--output_dir", type=str, default="exps/exp_person")
    parser.add_argument("--resolution",type=int,default=512)
    parser.add_argument("--model_type",type=str,default="MLP")

    #training
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--select", type=str, default="random")

    parser.add_argument("--learning_rate",type=float,default=1e-1,help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--scale_lr",action="store_true",default=False,help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",)
    parser.add_argument("--lr_scheduler",type=str,default="constant",help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'' "constant", "constant_with_warmup"]'),)
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler. was 500")
    
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=0, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    #validation
    parser.add_argument('--skip_evaluation', action='store_true')
    parser.add_argument('--log_every_steps', type=int, default=1000)
    parser.add_argument('--log_every_epochs', type=int, default=5)

    #testing
    parser.add_argument('--prompt', type=str, default="")
    parser.add_argument('--template_key', type=str, default="0")
    parser.add_argument('--concept', nargs='+') 
    parser.add_argument('--clip_attributes', type=str, nargs='+') 
    parser.add_argument('--num_test_samples', type=int, default=2)
    parser.add_argument('--original_sd', action='store_true')
    parser.add_argument('--interpolate_type', type=str, default="") 
    parser.add_argument('--interpolate_steps', nargs='+', type=float) 

    parser.add_argument('--evaluation_type', type=str, default="eval", choices=['eval','interpolate','winobias','i2p'])
    parser.add_argument('--image_dir', type=str, default="images")
    parser.add_argument('--prompt_file', type=str, default=None)

    # for fid and kid
    parser.add_argument('--src_img_dir', type=str) 
    parser.add_argument('--gen_img_dir', type=str) 
    parser.add_argument('--kid_subset_size', type=int, default=1000)

    parser.add_argument('--use_sld', action='store_true', help="use safety latent diffusion for testing")
    parser.add_argument('--use_esd', action='store_true', help="use erasing stable diffusion, https://erasing.baulab.info/weights/esd_models/NSFW/diffusers-nudity-ESDu1-UNET.pt")

    parser.add_argument('--negative_prompt', default=None, type=str, help="negative prompts for SD")

    parser.add_argument('--scheduler', default='pndm', type=str, choices=['pndm', 'ddim', 'ddpm'])
    parser.add_argument('--num_inference_steps', default=50, type=int)

    parser.add_argument('--fp16', action='store_true', help="use float16 precision")


    args = parser.parse_args()
    return args


if __name__=='__main__':
    args=parse_args()

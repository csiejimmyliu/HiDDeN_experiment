from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler,AutoencoderKL
import torch
import os
import time
import pprint
import argparse
import numpy as np
import pickle
import utils
import csv

from model.hidden import Hidden
from noise_layers.noiser import Noiser
from average_meter import AverageMeter
from noise_argparser import NoiseArgParser

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import random

def fix_deex(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
parser = argparse.ArgumentParser(description='Generate 5000 imgaes by diffusino model for fid calculation.')
parser.add_argument('--vae_path', '-v', type=str,help='Paht of watermarked vae.')
parser.add_argument('--save_folder', '-sf', required=True, type=str,help='The directory to save images.')
parser.add_argument('--cap_path', '-c',default='../cap_list.json' , type=str,help='Paht of 5000 captions.')
parser.add_argument('--seeds_path', '-seed',default='../5000_seeds.json' , type=str,help='Paht of 5000 seeds.')
parser.add_argument('--batch_size', '-b',required=True , type=int,help='Batch size')
parser.add_argument('--size', '-s',default= 512, type=int,help='Generated image size')
parser.add_argument('--model_id', '-m',default= "runwayml/stable-diffusion-v1-5", type=str,help='Diffusion model path')
parser.add_argument('--seed', default=870110, type=int,help='Random seed.')

args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

caps=json.load(open(args.cap_path))
seeds=json.load(open(args.seeds_path))

pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
if args.vae_path==None:
    pipe.vae=AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5",subfolder='vae')
else:
    tmpe_vae=AutoencoderKL.from_pretrained(args.vae_path)
    pipe.vae.decoder=tmpe_vae.decoder
    pipe.vae.post_quant_conv=tmpe_vae.post_quant_conv

def dummy(images, **kwargs):
    return images, False

#pipe.enable_vae_slicing()

pipe.safety_checker = dummy
pipe=pipe.to(device)
generator = torch.Generator(device=device)


for i in range(int(len(seeds)/args.batch_size)):
    output=pipe(prompt=caps[i*args.batch_size:i*args.batch_size+args.batch_size],height=args.size,width=args.size,guidance_scale=3.0,generator = [generator.manual_seed(gen_seed) for gen_seed in seeds[i*args.batch_size:i*args.batch_size+args.batch_size]])
    for j in range(args.batch_size):
        output.images[j].save(os.path.join(args.save_folder,f'{i*args.batch_size+j}.png'))
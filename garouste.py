#!/usr/bin/env python3

import os
import sys
import torch
import gc
from server_utils import save_tensors_as_images, get_face_lora_partitions, get_value_at_index
import random
import argparse

sys.path.append('/root/home/github/ComfyUI')

from custom_nodes import (
    KSampler,
    VAEDecode,
    EmptyLatentImage,
    NODE_CLASS_MAPPINGS,
    CLIPTextEncode,
    CheckpointLoaderSimple,
    LoraLoader,
)

def generate_paintings(prompt, face_lora_path, output_image_dir, batch_size=8, style=2, size="big", 
                       loras_dir="/root/home/github/garouste_server/loras"):

    style_dict = {2: 0.4, 1: 0.3, 0: 0.15}
    style_weight = style_dict.get(style)

    size_dict = {"small":{"width":832,"height":1152},
                 "big":{"width":1024,"height":1532},
                 "xbig":{"width":1532,"height":2024},
                 "small2":{"width":900,"height":1256},
                 "big2":{"width":1256,"height":1752},
                 }
    width, height = size_dict.get(size)["width"], size_dict.get(size)["height"]

    face_lora_weights = get_face_lora_partitions(n=4, start=1.1, end=1.55)
    
    print("face_lora_weights: ", face_lora_weights)
    print("style_weight: ", style_weight)
    print("width: ", width)
    print("height: ", height)
    
    with torch.inference_mode():
        for face_weight in face_lora_weights:
            checkpointloadersimple = CheckpointLoaderSimple()
            checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
                ckpt_name="sd_xl_base_1.0.safetensors"
            )

            loraloader = LoraLoader()
            face_lora_model = loraloader.load_lora(
                lora_name=face_lora_path,
                strength_model=face_weight,
                strength_clip=1,
                model=get_value_at_index(checkpointloadersimple_4, 0),
                clip=get_value_at_index(checkpointloadersimple_4, 1),
                use_path=True
            )

            composition_lora_model = loraloader.load_lora(
                lora_name=os.path.join(loras_dir,"portraits_bnha_i4500.safetensors"),
                strength_model=style_weight,
                strength_clip=1,
                model=get_value_at_index(face_lora_model, 0),
                clip=get_value_at_index(face_lora_model, 1),
                use_path=True
            )

            style_lora_model = loraloader.load_lora(
                lora_name=os.path.join(loras_dir,"garouste_raphael_style_2.0.safetensors"),
                strength_model=1,
                strength_clip=1,
                model=get_value_at_index(composition_lora_model, 0),
                clip=get_value_at_index(composition_lora_model, 1),
                use_path=True
            )

            cliptextencode = CLIPTextEncode()
            clip_positive_encoding = cliptextencode.encode(
                text=str(prompt),
                clip=get_value_at_index(style_lora_model, 1),
            )

            clip_negative_encoding = cliptextencode.encode(
                text="(blurry:1.5), watercolor, text, signature, ugly face", clip=get_value_at_index(style_lora_model, 1)
            )

            emptylatentimage = EmptyLatentImage()
            emptylatentimage_22 = emptylatentimage.generate(
                width=width, height=height, batch_size=batch_size
            )

            ksampler = KSampler()
            vaedecode = VAEDecode()


            ksampler_3 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=25,
                cfg=7,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                denoise=1,
                model=get_value_at_index(style_lora_model, 0),
                positive=get_value_at_index(clip_positive_encoding, 0),
                negative=get_value_at_index(clip_negative_encoding, 0),
                latent_image=get_value_at_index(emptylatentimage_22, 0),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
            )

            save_tensors_as_images(get_value_at_index(vaedecode_8, 0), output_image_dir, file_prefix=str(face_weight))
            torch.cuda.empty_cache()
            gc.collect()   
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your method description here.")
    parser.add_argument('--metadata_path', type=str, required=True, help='Workflow metadata')
    parser.add_argument('--face_lora_path', type=str, required=True, help='Path to face LORA data')
    parser.add_argument('--output_image_dir', type=str, required=True, help='Directory to save output images')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing (default: 8)')
    parser.add_argument('--style', type=int, default=0, help='Style parameter (default: 0)')
    parser.add_argument('--size', type=str, default='big', choices=['big', 'small'], help='Size option (default: big)')
    args = parser.parse_args()
    print(args)

    face_lora_path = args.face_lora_path
    output_image_dir = args.output_image_dir
    batch_size = args.batch_size
    style = args.style
    size  = args.size
    metadata_path  = args.metadata_path
    
    with open(metadata_path, 'r') as file:
        for line in file:
            if line.startswith("prompt: "):
                # Extracting the prompt variable
                prompt = line[len("prompt: "):].strip()

    generate_paintings(prompt=prompt,
    face_lora_path=args.face_lora_path,
    output_image_dir=args.output_image_dir,
    batch_size=args.batch_size,
    style=args.style,
    size=args.size)

    torch.cuda.empty_cache()
    gc.collect()
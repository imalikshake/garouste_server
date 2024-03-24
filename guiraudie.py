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


def generate_image(prompt, face_lora_path, output_image_dir, batch_size=8, 
                       loras_dir="/root/home/github/garouste_server/loras", neg_prompt=""):
    face_weight = 1
    with torch.inference_mode():
        
        emptylatentimage = EmptyLatentImage()
        emptylatentimage_in = emptylatentimage.generate(
            width=1296, height=704, batch_size=batch_size
        )

        checkpointloadersimple = CheckpointLoaderSimple()
        base_stable_diffisuion_model = checkpointloadersimple.load_checkpoint(
            ckpt_name="sd_xl_base_1.0.safetensors"
        )

        loraloader = LoraLoader()
        face_lora_model = loraloader.load_lora(
            lora_name=face_lora_path,
            strength_model=face_weight,
            strength_clip=1,
            model=get_value_at_index(base_stable_diffisuion_model, 0),
            clip=get_value_at_index(base_stable_diffisuion_model, 1),
            use_path=True
        )

        cliptextencode = CLIPTextEncode()
        pos_cliptextencode_in = cliptextencode.encode(
            text=str(prompt),
            clip=get_value_at_index(face_lora_model, 1),
        )

        neg_prompt += ",(worst quality:1.2), (low quality:1.2), normal quality, (jpeg artifacts:1.3)"
        
        neg_cliptextencode_in = cliptextencode.encode(
            text=str(neg_prompt),
            clip=get_value_at_index(face_lora_model, 1),
        )

        ksampler = KSampler()
        vaedecode = VAEDecode()

        ksampler_out = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=12,
            cfg=7,
            sampler_name="dpmpp_sde",
            scheduler="normal",
            denoise=1,
            model=get_value_at_index(face_lora_model, 0),
            positive=get_value_at_index(pos_cliptextencode_in, 0),
            negative=get_value_at_index(neg_cliptextencode_in, 0),
            latent_image=get_value_at_index(emptylatentimage_in, 0),
        )

        vaedecode_8 = vaedecode.decode(
            samples=get_value_at_index(ksampler_out, 0),
            vae=get_value_at_index(base_stable_diffisuion_model, 2),
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
    args = parser.parse_args()
    print(args)

    face_lora_path = args.face_lora_path
    output_image_dir = args.output_image_dir
    batch_size = args.batch_size
    metadata_path  = args.metadata_path
    
    prompt = ""
    neg_prompt = ""
    with open(metadata_path, 'r') as file:
        for line in file:
            if line.startswith("prompt: "):
                # Extracting the prompt variable
                prompt = line[len("prompt: "):].strip()
            if line.startswith("neg_prompt: "):
                # Extracting the prompt variable
                neg_prompt = line[len("neg_prompt: "):].strip()

    if not prompt:
        print("NO PROMPT")
        exit(1)

    generate_image(prompt=prompt,
    face_lora_path=args.face_lora_path,
    output_image_dir=args.output_image_dir,
    batch_size=args.batch_size,
    neg_prompt=neg_prompt)
    
    torch.cuda.empty_cache()
    gc.collect()
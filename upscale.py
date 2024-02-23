#!/usr/bin/env python3

import os
import sys
import torch
import gc
from server_utils import save_tensors_as_images, get_face_lora_partitions, get_value_at_index, import_custom_nodes
import random
import argparse
from typing import Sequence, Mapping, Any, Union

sys.path.append('/root/home/github/ComfyUI')


from custom_nodes import (
    LoadImage,
    VAEEncode,
    SaveImage,
    LoraLoader,
    CheckpointLoaderSimple,
    ImageScaleBy,
    VAEDecode,
    KSampler,
    CLIPTextEncode,
)

from nodes import (
    NODE_CLASS_MAPPINGS
)

def upscale(filename, image_path, output_image_dir, loras_dir="/root/home/github/garouste_server/loras"):
    import_custom_nodes()
    base_name = os.path.basename(image_path)
    filename, _ = os.path.splitext(base_name)

    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="sd_xl_base_1.0.safetensors"
        )

        loraloader = LoraLoader()
        loraloader_10 = loraloader.load_lora(
            lora_name=os.path.join(loras_dir,"garouste_raphael_style_2.0.safetensors"),
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
            use_path=True            
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(
            text='person playing chess, by Gerard Garouste, 2019 an oil painting, oil on canvas. The painting features a person sitting on a stool, holding a checkered box. A window is open in the background you can see the countryside. There are dark and expressive shadows. Clear face, sharp eyes, clear eyebrows',
            clip=get_value_at_index(loraloader_10, 1),
        )

        cliptextencode_24 = cliptextencode.encode(
            text="(blurry:1.5), smooth", clip=get_value_at_index(loraloader_10, 1)
        )

        upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        upscalemodelloader_25 = upscalemodelloader.load_model(
            model_name="4x_Nickelback_70000G.pth"
        )

        loadimage = LoadImage()
        loadimage_46 = loadimage.load_image(image=image_path, use_path=True)

        imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
        imageupscalewithmodel_47 = imageupscalewithmodel.upscale(
            upscale_model=get_value_at_index(upscalemodelloader_25, 0),
            image=get_value_at_index(loadimage_46, 0),
        )

        imagescaleby = ImageScaleBy()
        imagescaleby_38 = imagescaleby.upscale(
            upscale_method="area",
            scale_by=0.5,
            image=get_value_at_index(imageupscalewithmodel_47, 0),
        )

        vaeencode = VAEEncode()
        vaeencode_40 = vaeencode.encode(
            pixels=get_value_at_index(imagescaleby_38, 0),
            vae=get_value_at_index(checkpointloadersimple_4, 2),
        )

        ksampler = KSampler()
        vaedecode = VAEDecode()
        saveimage = SaveImage()


        ksampler_39 = ksampler.sample(
            seed=random.randint(1, 2**64),
            # seed=2,
            steps=50,
            cfg=6,
            sampler_name="euler",
            scheduler="karras",
            denoise=0.35000000000000003,
            model=get_value_at_index(loraloader_10, 0),
            positive=get_value_at_index(cliptextencode_6, 0),
            negative=get_value_at_index(cliptextencode_24, 0),
            latent_image=get_value_at_index(vaeencode_40, 0),
        )

        vaedecode_41 = vaedecode.decode(
            samples=get_value_at_index(ksampler_39, 0),
            vae=get_value_at_index(checkpointloadersimple_4, 2),
        )

        # save_tensors_as_images(get_value_at_index(vaedecode_41, 0), output_image_dir, file_prefix="medium-retextured")

        imageupscalewithmodel_43 = imageupscalewithmodel.upscale(
            upscale_model=get_value_at_index(upscalemodelloader_25, 0),
            image=get_value_at_index(vaedecode_41, 0),
        )
        save_tensors_as_images(get_value_at_index(imageupscalewithmodel_43, 0), output_image_dir, file_prefix=f"{filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscalings script.")
    parser.add_argument('--image_path', type=str, required=True, help='Input image path')
    parser.add_argument('--output_image_dir', type=str, default="/root/home/github/garouste_server/out_up/", help='Input image path')
    args = parser.parse_args()
    image_path = args.image_path
    output_image_dir = args.output_image_dir
    filename = os.path.splitext(os.path.basename(image_path))[0]
    upscale(filename, image_path, output_image_dir)

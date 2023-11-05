from typing import Sequence, Mapping, Any, Union
import os
import random
import sys
import torch
from PIL import Image
import numpy as np
import logging

sys.path.append('/home/paperspace/github/ComfyUI')

from nodes import (
    SaveImage,
    KSampler,
    VAEDecode,
    EmptyLatentImage,
    NODE_CLASS_MAPPINGS,
    CLIPTextEncode,
    CheckpointLoaderSimple,
    LoraLoader,
)

class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt):
        super().__init__(fmt)
        self.colors = {
            logging.DEBUG: '\033[32m',
            logging.INFO: '\033[34m',
            logging.WARNING: '\033[33m',
            logging.ERROR: '\033[31m',
            logging.CRITICAL: '\033[35m',
        }

    def format(self, record):
        level_color = self.colors.get(record.levelno, '')
        message = super().format(record)
        return f'{level_color}{message}\033[0m'
    

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def save_images(image_tensor_list, output_folder):
    for n, image in enumerate(image_tensor_list):
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        file = f"{n}.png"
        img.save(os.path.join(output_folder, file), compress_level=4)
    return


def generate_job(job_id, prompt, face_lora_path, output_images_dir):
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="sd_xl_base_1.0.safetensors"
        )

        loraloader = LoraLoader()
        face_lora_model = loraloader.load_lora(
            lora_name=face_lora_path,
            strength_model=1.1,
            strength_clip=1,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
            use_path=True
        )

        composition_lora_model = loraloader.load_lora(
            lora_name="portraits_bnha_i4500.safetensors",
            strength_model=0.7000000000000001,
            strength_clip=1,
            model=get_value_at_index(face_lora_model, 0),
            clip=get_value_at_index(face_lora_model, 1),
        )

        style_lora_model = loraloader.load_lora(
            lora_name="garouste_raphael_style_2.0.safetensors",
            strength_model=1.2,
            strength_clip=0.5,
            model=get_value_at_index(composition_lora_model, 0),
            clip=get_value_at_index(composition_lora_model, 1),
        )

        cliptextencode = CLIPTextEncode()
        clip_positive_encoding = cliptextencode.encode(
            text=str(prompt),
            clip=get_value_at_index(style_lora_model, 1),
        )

        clip_negative_encoding = cliptextencode.encode(
            text="blurry, watercolor", clip=get_value_at_index(style_lora_model, 1)
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_22 = emptylatentimage.generate(
            width=1024, height=1536, batch_size=4
        )

        ksampler = KSampler()
        vaedecode = VAEDecode()
        saveimage = SaveImage()

        for q in range(1):
            ksampler_3 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
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

            # saveimage_9 = saveimage.save_images(
            #     filename_prefix="ComfyUI", images=get_value_at_index(vaedecode_8, 0)
            # )
            save_images(get_value_at_index(vaedecode_8, 0), output_images_dir)

from typing import Sequence, Mapping, Any, Union
import os
import random
import sys
import torch
from PIL import Image
import numpy as np
import logging
import base64
from io import BytesIO
import toml
import subprocess
import threading
import subprocess
import gc
import time

sys.path.append('/home/paperspace/github/ComfyUI')

from custom_nodes import (
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


def segment_images(basedir, newdir, sizes=4, colors=2, dir="/home/paperspace/github/garouste_server/"):
     subprocess.call([f"{dir}preproc.py",
                      '--input',
                      basedir,
                      '--output',
                      newdir,
                      '--sizes',
                      str(sizes)
                      ])   

def create_directories_for_job(job_id, proj_path):
    job_dir = os.path.join(proj_path, job_id)
    output_image_dir = os.path.join(job_dir, "out_images")
    face_image_dir = os.path.join(job_dir, "face_images")
    dataset_dir = os.path.join(job_dir, "dataset")
    face_lora_dir = os.path.join(job_dir, "lora")
    face_lora_path = os.path.join(face_lora_dir, "last.safetensors")

    # Create job directory if it doesn't exist
    if not os.path.exists(job_dir):
        os.mkdir(job_dir)

    # Create output_image_dir if it doesn't exist
    if not os.path.exists(output_image_dir):
        os.mkdir(output_image_dir)

    # Create face_image_dir if it doesn't exist
    if not os.path.exists(face_image_dir):
        os.mkdir(face_image_dir)

    # Create dataset_dir if it doesn't exist
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    # Create face_lora_dir if it doesn't exist
    if not os.path.exists(face_lora_dir):
        os.mkdir(face_lora_dir)

    return job_dir, output_image_dir, face_image_dir, dataset_dir, face_lora_dir, face_lora_path


def encode_images_from_directory(output_image_dir):
    base64_encoded_images = []

    for filename in os.listdir(output_image_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(output_image_dir, filename)

            # Open and read the image
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

            # Base64 encode the image
            encoded_image = base64.b64encode(image_data).decode('utf-8')

            # Append the base64 encoded image to the list
            base64_encoded_images.append(encoded_image)

    return base64_encoded_images


def save_images_from_request(images_dict, face_image_dir):
    for i, entry in enumerate(images_dict):
        name = entry["name"]
        image_bytes = entry["data"]
        decoded_image = base64.b64decode(image_bytes.encode('utf-8'))
        image = Image.open(BytesIO(decoded_image))
        image = image.convert('RGB')
        image.save(os.path.join(face_image_dir, name))


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


def save_tensors_as_images(image_tensor_list, output_folder, file_prefix):
    for n, image in enumerate(image_tensor_list):
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        file = f"{file_prefix}_{n}.png"
        img.save(os.path.join(output_folder, str(int(time.time() * 1000) % 100000000)+"_"+file), compress_level=4)
    return

    
def get_face_lora_partitions(n=4, start=1.1, end=1.5):
    step = (end - start) / (n - 1) if n > 1 else 0
    return [start + i * step for i in range(n)]


# def generate_paintings(job_id, prompt, face_lora_path, output_image_dir, batch_size=8, style=0, size="big"):
#     style_dict = {2: 0.4, 1: 0.3, 0: 0.15}
#     style_weight = style_dict.get(style)

#     size_dict = {"small":{"width":832,"height":1152},
#                  "big":{"width":1024,"height":1532},
#                  "xbig":{"width":1532,"height":2024},
#                  "small2":{"width":900,"height":1256},
#                  "big2":{"width":1256,"height":1752},
#                  }
#     width, height = size_dict.get(size)["width"], size_dict.get(size)["height"]

#     face_lora_weights = get_face_lora_partitions(n=3, start=1.1, end=1.4)
    
#     print("face_lora_weights: ", face_lora_weights)
#     print("style_weight: ", style_weight)
#     print("width: ", width)
#     print("height: ", height)
    
#     with torch.inference_mode():
#         for face_weight in face_lora_weights:
#             checkpointloadersimple = CheckpointLoaderSimple()
#             checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
#                 ckpt_name="sd_xl_base_1.0.safetensors"
#             )

#             loraloader = LoraLoader()
#             face_lora_model = loraloader.load_lora(
#                 lora_name=face_lora_path,
#                 strength_model=face_weight,
#                 strength_clip=1,
#                 model=get_value_at_index(checkpointloadersimple_4, 0),
#                 clip=get_value_at_index(checkpointloadersimple_4, 1),
#                 use_path=True
#             )

#             composition_lora_model = loraloader.load_lora(
#                 lora_name="portraits_bnha_i4500.safetensors",
#                 strength_model=style_weight,
#                 strength_clip=1,
#                 model=get_value_at_index(face_lora_model, 0),
#                 clip=get_value_at_index(face_lora_model, 1),
#             )

#             style_lora_model = loraloader.load_lora(
#                 lora_name="garouste_raphael_style_2.0.safetensors",
#                 strength_model=1,
#                 strength_clip=1,
#                 model=get_value_at_index(composition_lora_model, 0),
#                 clip=get_value_at_index(composition_lora_model, 1),
#             )

#             cliptextencode = CLIPTextEncode()
#             clip_positive_encoding = cliptextencode.encode(
#                 text=str(prompt),
#                 clip=get_value_at_index(style_lora_model, 1),
#             )

#             clip_negative_encoding = cliptextencode.encode(
#                 text="(blurry:1.5),  watercolor, text, signature, ugly face", clip=get_value_at_index(style_lora_model, 1)
#             )

#             emptylatentimage = EmptyLatentImage()
#             emptylatentimage_22 = emptylatentimage.generate(
#                 width=width, height=height, batch_size=batch_size
#             )

#             ksampler = KSampler()
#             vaedecode = VAEDecode()


#             ksampler_3 = ksampler.sample(
#                 seed=random.randint(1, 2**64),
#                 steps=25,
#                 cfg=7,
#                 sampler_name="dpmpp_2m",
#                 scheduler="karras",
#                 denoise=1,
#                 model=get_value_at_index(style_lora_model, 0),
#                 positive=get_value_at_index(clip_positive_encoding, 0),
#                 negative=get_value_at_index(clip_negative_encoding, 0),
#                 latent_image=get_value_at_index(emptylatentimage_22, 0),
#             )

#             vaedecode_8 = vaedecode.decode(
#                 samples=get_value_at_index(ksampler_3, 0),
#                 vae=get_value_at_index(checkpointloadersimple_4, 2),
#             )

#             save_tensors_as_images(get_value_at_index(vaedecode_8, 0), output_image_dir, file_prefix=str(face_weight))
#             del face_lora_model
#             del composition_lora_model
#             del style_lora_model
#             del clip_positive_encoding
#             del vaedecode_8
#             torch.cuda.empty_cache()
#             gc.collect() 
#     torch.cuda.empty_cache()
#     gc.collect()            
    
def generate(job_id, prompt, face_lora_path, output_image_dir, batch_size=8, style=0, size="big", dir="/home/paperspace/github/garouste_server/"):
     subprocess.call([f"{dir}generate_paintings.py",
                      '--job_id',
                      job_id,
                      '--prompt',
                      prompt,
                      '--face_lora_path',
                      face_lora_path,
                      '--output_image_dir',
                      output_image_dir,
                      '--batch_size',
                      str(batch_size),
                      '--style',
                      str(style),
                      '--size',
                      str(size)])   


def prepare_config_tomls(config_toml_path, dataset_toml_path, job_dir, face_lora_dir, dataset_dir):
        # Load the TOML file
    with open(config_toml_path, 'r') as f:
        config_dict = toml.load(f)
    with open(dataset_toml_path, 'r') as f:
        dataset_dict = toml.load(f)
    config_dict['model']['output_dir'] = face_lora_dir
    dataset_dict['datasets'][0]['subsets'][0]['image_dir'] = dataset_dir
    job_config_toml_path = os.path.join(job_dir, f"config.toml")
    job_dataset_toml_path = os.path.join(job_dir, f"dataset.toml")
    with open(job_config_toml_path, 'w') as f:
        toml.dump(config_dict, f)
    with open(job_dataset_toml_path, 'w') as f:
        toml.dump(dataset_dict, f)
    
    return job_config_toml_path, job_dataset_toml_path

def train_model(train_script_path, dataset_config, config_file):
    subprocess.call(['accelerate', 
                     'launch', 
                     '--num_cpu_threads_per_process', 
                     '4', 
                     train_script_path,
                     '--dataset_config',
                     dataset_config,
                     '--config_file',
                     config_file])

def train_model_gpu(gpu_id, train_script_path, dataset_config, config_file):
    subprocess.call(['accelerate', 
                     'launch', 
                     '--num_cpu_threads_per_process', 
                     '4', 
                     train_script_path,
                     '--dataset_config',
                     dataset_config,
                     '--config_file',
                     config_file])


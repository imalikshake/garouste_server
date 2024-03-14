from typing import Sequence, Mapping, Any, Union
import os
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import toml
import subprocess
import subprocess
import time


def train_model(train_script_path, dataset_config, config_file, gpu_id=0):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    subprocess.call(['accelerate', 
                     'launch', 
                     '--num_cpu_threads_per_process', 
                     '4', 
                     train_script_path,
                     '--dataset_config',
                     dataset_config,
                     '--config_file',
                     config_file], env=env)



def segment_images(basedir, newdir, sizes=4, colors=2, dir="/root/home/github/garouste_server/", gpu_id=0):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    subprocess.call([f"{dir}segment.py",
                      '--input',
                      basedir,
                      '--output',
                      newdir,
                      '--sizes',
                      str(sizes)
                      ], env=env)   

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
         

def combine_toml(dict1, dict2):
  """
  Combines two dictionaries and updates common values from the second dictionary.

  Args:
    dict1 (dict): The first dictionary.
    dict2 (dict): The second dictionary.

  Returns:
    dict: The combined dictionary.
  """
  combined_data = {}
  for section, section_data in dict1.items():
    combined_data[section] = section_data.copy()
    if section in dict2:
      combined_data[section].update(dict2[section])

  # Add remaining keys from dict2 not present in dict1
  for section, section_data in dict2.items():
    if section not in combined_data:
      combined_data[section] = section_data.copy()

  return combined_data


def prepare_config_tomls(experience_toml_path, config_toml_path, dataset_toml_path, job_dir, face_lora_dir, dataset_dir):
    with open(experience_toml_path, 'r') as f:
        exp_dict = toml.load(f)
    with open(config_toml_path, 'r') as f:
        config_dict = toml.load(f)
    with open(dataset_toml_path, 'r') as f:
        dataset_dict = toml.load(f)
    dataset_dict['datasets'][0]['subsets'][0]['image_dir'] = dataset_dir
    merged_config_dict = combine_toml(config_dict, exp_dict)
    merged_config_dict['model']['output_dir'] = face_lora_dir
    job_config_toml_path = os.path.join(job_dir, f"config.toml")
    job_dataset_toml_path = os.path.join(job_dir, f"dataset.toml")
    with open(job_config_toml_path, 'w') as f:
        toml.dump(merged_config_dict, f)
    with open(job_dataset_toml_path, 'w') as f:
        toml.dump(dataset_dict, f)
    return job_config_toml_path, job_dataset_toml_path

import requests
from server_utils import segment_images, train_model
import subprocess
import os
import toml

class Experience:
    def __init__(self, job_dir, output_image_dir, face_image_dir, dataset_dir, face_lora_dir, face_lora_path, job_config_toml_path, job_dataset_toml_path, train_script_path):
        self.job_dir = job_dir
        self.output_image_dir = output_image_dir
        self.face_image_dir = face_image_dir
        self.dataset_dir = dataset_dir
        self.face_lora_dir = face_lora_dir
        self.face_lora_path = face_lora_path
        self.job_config_toml_path = job_config_toml_path
        self.job_dataset_toml_path = job_dataset_toml_path
        self.train_script_path = train_script_path
        self._init_params()

    def _init_params(self):
        with open(self.job_config_toml_path, 'r') as f:
            param_dict = toml.load(f)
        self.resizes = param_dict["segment"]["sizes"]

    def train(self, gpu_id="0"):
        segment_images(basedir=self.face_image_dir, newdir=self.dataset_dir, sizes=self.resizes,  gpu_id=gpu_id)
        print(self.train_script_path)
        print(self.job_dataset_toml_path)
        print(self.job_config_toml_path)
        print(gpu_id)
        train_model(train_script_path=self.train_script_path, dataset_config=self.job_dataset_toml_path, config_file=self.job_config_toml_path, gpu_id=gpu_id)
    
    def return_output(self):
        url = f'https://ebb.global/_vh_api/aiapi/genValidator.php?id={job_id}'
        requests.get(url)


class GarousteExperience(Experience):
    def __init__(self, job_dir, output_image_dir, face_image_dir, dataset_dir, face_lora_dir, face_lora_path, job_config_toml_path, job_dataset_toml_path, train_script_path, metadata_path):
        super().__init__(job_dir, output_image_dir, face_image_dir, dataset_dir, face_lora_dir, face_lora_path, job_config_toml_path, job_dataset_toml_path, train_script_path)
        self.metadata_path = metadata_path

    def generate(self, batch_size=8, style=0, size="big", dir="/root/home/github/garouste_server/", gpu_id="0"):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        subprocess.call([f"{dir}garouste.py",
                        '--metadata_path',
                        self.metadata_path,
                        '--face_lora_path',
                        self.face_lora_path,
                        '--output_image_dir',
                        self.output_image_dir,
                        '--batch_size',
                        str(batch_size),
                        '--style',
                        str(style),
                        '--size',
                        str(size)], env=env)   


class GuiraudieExperience(Experience):
    def __init__(self, job_dir, output_image_dir, face_image_dir, dataset_dir, face_lora_dir, face_lora_path, job_config_toml_path, job_dataset_toml_path, train_script_path, metadata_path):
        super().__init__(job_dir, output_image_dir, face_image_dir, dataset_dir, face_lora_dir, face_lora_path, job_config_toml_path, job_dataset_toml_path, train_script_path)
        self.metadata_path = metadata_path

    def generate(self, batch_size=8, style=0, size="big", dir="/root/home/github/garouste_server/", gpu_id="0"):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        subprocess.call([f"{dir}guiraudie.py",
                        '--metadata_path',
                        self.metadata_path,
                        '--face_lora_path',
                        self.face_lora_path,
                        '--output_image_dir',
                        self.output_image_dir,
                        '--batch_size',
                        str(batch_size)], env=env)
                           
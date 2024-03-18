from server_utils import prepare_config_tomls
import argparse
import os 
from experience import GuiraudiExperience, GarousteExperience
from mapping import experience_mapping

def create_directories(job_dir):
    output_image_dir = os.path.join(job_dir, "out_images")
    face_image_dir = os.path.join(job_dir, "face_images")
    dataset_dir = os.path.join(job_dir, "dataset")
    face_lora_dir = os.path.join(job_dir, "lora")
    face_lora_path = os.path.join(face_lora_dir, "last.safetensors")
    if not os.path.exists(output_image_dir):
        os.mkdir(output_image_dir)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    if not os.path.exists(face_lora_dir):
        os.mkdir(face_lora_dir)
    return job_dir, output_image_dir, face_image_dir, dataset_dir, face_lora_dir, face_lora_path
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="Single-pass execution.")
    parser.add_argument('--gpu_id', default=0, type=int, help="GPU ID")
    parser.add_argument('--job_dir', type=str, help="Job directory")
    parser.add_argument('--job_id', type=int, help="Job ID")
    parser.add_argument('--experience_id', type=int, help="Experience ID")
    return parser.parse_args()
 

if __name__ == '__main__':
    #Parsing
    args = parse_args()
    job_id = args.job_id
    job_dir = args.job_dir
    gpu_id = str(args.gpu_id)
    experience_id = str(args.experience_id)

    # Static filenames. Need to clean this.
    config_toml_path = "/root/home/github/garouste_server/configs/general/general_config.toml"
    dataset_toml_path = "/root/home/github/garouste_server/configs/general/general_dataset.toml"
    experience_toml_path = f"/root/home/github/garouste_server/configs/experience/{experience_id}.toml"
    train_script_path = "/root/home/github/sd-scripts/sdxl_train_network.py"
    metadata_path = os.path.join(job_dir, "metadata.txt")
    
    # Create directories
    job_dir, output_image_dir, face_image_dir, dataset_dir, face_lora_dir, face_lora_path = create_directories(
        job_dir=job_dir
    )   
    
    #Update tomls
    job_config_toml_path, job_dataset_toml_path = prepare_config_tomls(experience_toml_path=experience_toml_path, config_toml_path=config_toml_path, dataset_toml_path=dataset_toml_path, job_dir=job_dir, face_lora_dir=face_lora_dir, dataset_dir=dataset_dir)
 

    # Dynamically get experience from experience ID.
    experience = experience_mapping[experience_id](job_dir, output_image_dir, face_image_dir, dataset_dir, face_lora_dir, face_lora_path, job_config_toml_path, job_dataset_toml_path, train_script_path, metadata_path)

    experience.train()
    experience.generate()
from server_utils import segment_images, generate, train_model, prepare_config_tomls
import argparse
import os 
import requests
 
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
 
    # Create face_lora_dir if it doesn't exist
    if not os.path.exists(face_lora_dir):
        os.mkdir(face_lora_dir)
 
    return job_dir, output_image_dir, face_image_dir, dataset_dir, face_lora_dir, face_lora_path
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="Single-pass execution.")
    parser.add_argument('--gpu_id', default=0, type=int, help="GPU ID")
    parser.add_argument('--job_dir', type=str, help="Job directory")
    parser.add_argument('--job_id', type=int, help="Job ID")
    return parser.parse_args()
 
args = parse_args()
 
if __name__ == '__main__':
    args = parse_args()
    job_id = args.job_id
    job_dir = args.job_dir
    gpu_id = str(args.gpu_id)
    config_toml_path = "/root/home/github/garouste_server/general_config.toml"
    dataset_toml_path = "/root/home/github/garouste_server/general_dataset.toml"
    train_script_path = "/root/home/github/sd-scripts/sdxl_train_network.py"
 
    # Get local paths
    job_dir, output_image_dir, face_image_dir, dataset_dir, face_lora_dir, face_lora_path = create_directories(
        job_dir=job_dir
    )   
 
    # Reading metadata file to get the prompt variable
    with open(os.path.join(job_dir, "metadata.txt"), 'r') as file:
        for line in file:
            if line.startswith("prompt: "):
                # Extracting the prompt variable
                prompt = line[len("prompt: "):].strip()
 
    # Update training configs
    job_config_toml_path, job_dataset_toml_path = prepare_config_tomls(config_toml_path=config_toml_path, dataset_toml_path=dataset_toml_path, job_dir=job_dir, face_lora_dir=face_lora_dir, dataset_dir=dataset_dir)
 
    segment_images(basedir=face_image_dir, newdir=dataset_dir, sizes=4, gpu_id=gpu_id)
    train_model(train_script_path=train_script_path, dataset_config=job_dataset_toml_path, config_file=job_config_toml_path, gpu_id=gpu_id)
    generate(prompt=prompt, face_lora_path=face_lora_path, output_image_dir=output_image_dir, gpu_id=gpu_id)
    url = f'https://ebb.global/_vh_api/aiapi/genValidator.php?id={job_id}'
    requests.get(url)

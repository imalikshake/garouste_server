from server_utils import segment_images, generate, train_model
import argparse
import os 
import requests
import toml

def override_toml_values(A, B):
    for section, values in B.items():
        if section in A:
            for key, value in values.items():
                A[section][key] = value
        else:
            A[section] = values
    return A

def prepare_config_tomls(custom_toml_path, config_toml_path, dataset_toml_path, job_dir, face_lora_dir, dataset_dir):
        # Load the TOML file
    with open(custom_toml_path, 'r') as f:
        custom_dict = toml.load(f)
    with open(config_toml_path, 'r') as f:
        config_dict = toml.load(f)
    with open(dataset_toml_path, 'r') as f:
        dataset_dict = toml.load(f)
    config_dict['model']['output_dir'] = face_lora_dir
    dataset_dict['datasets'][0]['subsets'][0]['image_dir'] = dataset_dir
    custom_dict = override_toml_values(config_dict, custom_dict)
    job_config_toml_path = os.path.join(job_dir, f"config.toml")
    job_dataset_toml_path = os.path.join(job_dir, f"dataset.toml")
    with open(job_config_toml_path, 'w') as f:
        toml.dump(custom_dict, f)
    with open(job_dataset_toml_path, 'w') as f:
        toml.dump(dataset_dict, f)
    return job_config_toml_path, job_dataset_toml_path

def create_directories(job_dir):
    # output_image_dir = os.path.join(job_dir, "out_images")
    face_image_dir = os.path.join(job_dir, "crop")
    dataset_dir = os.path.join(job_dir, "dataset")
    face_lora_dir = os.path.join(job_dir, "lora")
    face_lora_path = os.path.join(face_lora_dir, "last.safetensors")
 
    # if not os.path.exists(output_image_dir):
    #     os.mkdir(output_image_dir)
 
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
 
    # Create face_lora_dir if it doesn't exist
    if not os.path.exists(face_lora_dir):
        os.mkdir(face_lora_dir)
 
    return job_dir, face_image_dir, dataset_dir, face_lora_dir, face_lora_path
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="Single-pass execution.")
    parser.add_argument('--gpu_id', default=0, type=int, help="GPU ID")
    parser.add_argument('--face_image_dir', type=str, help="Data directory")
    parser.add_argument('--custom_toml_path', type=str, help="Custom toml path")
    return parser.parse_args()
 
def generate_ascending_ids(folder_path):
    # Get list of folders in the directory
    folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    if not folders:  # Check if there are no existing folders
        return str(0)
    
    # Sort folders by their IDs and get the last one
    folders.sort(key=lambda x: int(x))
    last_id = int(folders[-1])
    
    # Increment the last ID for the new folder
    new_id = last_id + 1
    
    return str(new_id)

args = parse_args()
 
if __name__ == '__main__':
    args = parse_args()
    face_image_dir = args.face_image_dir
    custom_toml_path = args.custom_toml_path
    gpu_id = str(args.gpu_id)
    
    job_root = "jobs/"
    config_toml_path = "/root/home/github/garouste_server/general_config.toml"
    dataset_toml_path = "/root/home/github/garouste_server/general_dataset.toml"
    train_script_path = "/root/home/github/sd-scripts/sdxl_train_network.py"
    job_dir = os.path.join(job_root,generate_ascending_ids(job_root))
    os.mkdir(job_dir)

    # Get local paths
    job_dir, _, dataset_dir, face_lora_dir, face_lora_path = create_directories(
        job_dir=job_dir
    )   
    print(job_dir)
    # Update training configs
    job_config_toml_path, job_dataset_toml_path = prepare_config_tomls(custom_toml_path=custom_toml_path, config_toml_path=config_toml_path, dataset_toml_path=dataset_toml_path, job_dir=job_dir, face_lora_dir=face_lora_dir, dataset_dir=dataset_dir)
 
    with open(custom_toml_path, 'r') as f:
        custom_dict = toml.load(f)
    segment_images(basedir=face_image_dir, newdir=dataset_dir, sizes=custom_dict["segment"]["sizes"], gpu_id=gpu_id, token=custom_dict["segment"]["token"])
    train_model(train_script_path=train_script_path, dataset_config=job_dataset_toml_path, config_file=job_config_toml_path, name=custom_dict["model"]["name"], gpu_id=gpu_id)

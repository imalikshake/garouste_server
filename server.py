from flask import Flask, request, jsonify
import sys
import torch
import logging
from server_utils import generate_paintings, ColoredFormatter, encode_images_from_directory, save_images_from_request,create_directories_for_job, train_model, prepare_config_tomls
import gc
from preproc import segment_images
import subprocess

logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO or the desired level.
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="server.log",  # Set the log file name.
    filemode="a"  # Append mode (use 'w' to overwrite existing log).
)


logger = logging.getLogger("my_server")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)
logger.info('This is a message that will be printed to stdout')
 

app = Flask(__name__)


@app.route('/generate_images', methods=['POST'])
def generate_images():
    data = request.get_json()
    job_id = data.get('job_id')
    prompt = data.get('prompt')
    images_dict = data.get('images')
    proj_path = "/home/paperspace/github/garouste_server/temp_dir"
    config_toml_path = "/home/paperspace/github/garouste_server/general_config.toml"
    dataset_toml_path = "/home/paperspace/github/garouste_server/general_dataset.toml"
    train_script_path = "/home/paperspace/github/sd-scripts/sdxl_train_network.py"

    job_dir, output_image_dir, face_image_dir, dataset_dir, face_lora_dir, face_lora_path = create_directories_for_job(
        job_id=job_id,
        proj_path=proj_path
    )
    job_config_toml_path, job_dataset_toml_path = prepare_config_tomls(config_toml_path=config_toml_path, dataset_toml_path=dataset_toml_path, job_dir=job_dir, face_lora_dir=face_lora_dir, dataset_dir=dataset_dir)
    save_images_from_request(images_dict=images_dict, face_image_dir=face_image_dir)
    
    segment_images(basedir=face_image_dir, newdir=dataset_dir)
    
    train_model(train_script_path=train_script_path, dataset_config=job_dataset_toml_path, config_file=job_config_toml_path)
    
    generate_paintings(job_id=job_id, prompt=prompt, face_lora_path=face_lora_path, output_image_dir=output_image_dir)
    painting_byte_images = encode_images_from_directory(output_image_dir=output_image_dir)
    
    response_data = {'job_id': 'your_job_id', 'images': painting_byte_images}
    torch.cuda.empty_cache()
    gc.collect()
    
    return jsonify(response_data)


# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
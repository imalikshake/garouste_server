import queue
import threading
import torch
import time
import gc
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import uuid
from server_utils import GPUManager, generate_paintings, encode_images_from_directory, save_images_from_request,create_directories_for_job, train_model, prepare_config_tomls
from preproc import segment_images
import argparse
import os 

app = Flask(__name__)

# Define the queue to store incoming image processing requests
request_queue = queue.Queue()

# Store job status and result
job_statuses = {}
job_results = {}

def generate_images(job_id, prompt, images_dict):

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
    
    segment_images(basedir=face_image_dir, newdir=dataset_dir, SIZES=4)
    
    # train_model_gpu(train_script_path=train_script_path, gpu_id=gpu_id,dataset_config=job_dataset_toml_path, config_file=job_config_toml_path)
    train_model(train_script_path=train_script_path, dataset_config=job_dataset_toml_path, config_file=job_config_toml_path)
    
    generate_paintings(job_id=job_id, prompt=prompt, face_lora_path=face_lora_path, output_image_dir=output_image_dir)
    # generate_paintings_gpu(job_id=job_id, gpu_id=gpu_id, prompt=prompt, face_lora_path=face_lora_path, output_image_dir=output_image_dir)

    return output_image_dir

def get_generated_results(job_id):
    generated_response_dir = job_results.get(job_id)
    painting_byte_images = encode_images_from_directory(output_image_dir=generated_response_dir)
    return painting_byte_images

# Implement a route to check job status
@app.route('/check_job_status', methods=['GET'])
def check_job_status():
    job_id = request.args.get('job_id')
    if job_id in job_statuses:
        status = job_statuses[job_id]
        if status == 'completed':
            painting_byte_images = get_generated_results(job_id)
            response_data = {'status': status, 'job_id': job_id, 'images': painting_byte_images}
        else:
            response_data = {'status': status, 'job_id': job_id}
    else:
        response_data = {'status': 'not_found', 'job_id': job_id}

    return jsonify(response_data)

# Define a function to process image requests
def process_image_request(gpu_id):
    # Continuously check the queue for new image requests
    while True:
        if not request_queue.empty():
            request_data = request_queue.get()

            job_id = request_data.get('job_id')  # Get the request ID from the client
            #REMOVE AFTter
            # images_dict = request_data.get('images_dict')
            prompt = request_data.get('prompt')


            # Store the request status as "processing"
            job_statuses[job_id] = 'processing'
            # print(images_dict)

            # Handle image processing
            try:
                # generated_response_dir = generate_images(job_id, prompt, images_dict)  # Simulate processing time
                generated_response_dir = generate_batches(job_id, prompt)  # Simulate processing time

                # Store the result and update the job status to "completed"
                job_results[job_id] = generated_response_dir
                job_statuses[job_id] = 'completed'
            except Exception as e:
                # If there's an error, store the status as "failed" and log the error
                job_statuses[job_id] = 'failed'
                print(f"Error processing image request with ID: {job_id}: {str(e)}")
            finally:
                request_queue.task_done()
                torch.cuda.empty_cache()
                gc.collect()

# Implement a route to submit image processing requests
@app.route('/submit_job', methods=['POST'])
def submit_job():
    data = request.get_json()
    job_id = data.get('job_id')
    prompt = data.get('prompt')
    images_dict = data.get('images')

    request_queue.put({'job_id': job_id, 'prompt': prompt, 'images_dict': images_dict})

    job_statuses[job_id] = "waiting"
    
    return jsonify({'message': 'Image processing request submitted successfully'})

def generate_batches(job_id, prompt, output_image_dir="/home/paperspace/garouste_server/tests/"):

    proj_path = "/home/paperspace/github/garouste_server/temp_dir"
    config_toml_path = "/home/paperspace/github/garouste_server/general_config.toml"
    dataset_toml_path = "/home/paperspace/github/garouste_server/general_dataset.toml"
    train_script_path = "/home/paperspace/github/sd-scripts/sdxl_train_network.py"
    out_dir = output_image_dir
    job_dir, output_image_dir, face_image_dir, dataset_dir, face_lora_dir, face_lora_path = create_directories_for_job(
        job_id=job_id,
        proj_path=proj_path
    )
    output_image_dir = os.path.join(output_image_dir, str(int(time.time() * 1000) % 100000000))
    os.mkdir(output_image_dir)
    print(output_image_dir)
    generate_paintings(job_id=job_id, prompt=prompt, face_lora_path=face_lora_path, output_image_dir=output_image_dir)

    return output_image_dir


# Define a function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Flask Server with Arguments")
    parser.add_argument('--gpus', default=1, type=int, help="Number of GPUs")
    return parser.parse_args()

args = parse_args()

# num_worker_threads = args.gpus  # You can adjust the number of worker threads
num_worker_threads = 1 # You can adjust the number of worker threads
GPU_MANAGER = GPUManager(num_worker_threads)
worker_threads = []

for _ in range(num_worker_threads):
    thread = threading.Thread(target=process_image_request, args=(GPU_MANAGER.assign_gpu(),))
    thread.start()
    worker_threads.append(thread)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

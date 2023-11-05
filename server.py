from flask import Flask, request, jsonify
import os
import sys
import torch
import base64
import logging
import queue
import threading
from server_utils import generate_job, ColoredFormatter


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
    job_dir = os.path.join("/home/paperspace/github/garouste_server/temp_dir", job_id)
    
    if not os.path.exists(job_dir):
        os.mkdir(job_dir)
    
    output_image_dir = os.path.join(job_dir, "images")
    if not os.path.exists(output_image_dir):
        os.mkdir(output_image_dir)
    
    face_lora_path = os.path.join(job_dir, "lora", "woman-epoch-000005.safetensors")

    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of available GPUs: {num_gpus}")

        # Get information about each GPU
        for i in range(num_gpus):
            gpu = torch.cuda.get_device_properties(i)
            available_memory = torch.cuda.memory_allocated(i)
            print(f"Available GPU memory: {available_memory} bytes")
            logger.info(f"GPU {i}: {gpu.name}, Available GPU memory: {available_memory / 1024**3} GB")
    else:
        logger.info("No GPU available. Using CPU.")

    generate_job(job_id, prompt, face_lora_path, output_image_dir)
    
    base64_encoded_images = []

    # # Replace this loop with actual image generation
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

    # Return the images as a JSON response
    response_data = {'job_id': 'your_job_id', 'images': base64_encoded_images}
    
    torch.cuda.empty_cache()
    
    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True)
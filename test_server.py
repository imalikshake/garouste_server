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

app = Flask(__name__)

# Define the queue to store incoming image processing requests
request_queue = queue.Queue()

# Store job status and result
job_statuses = {}
job_results = {}

class GPUManager:
    def __init__(self, num_gpus):
        self.available_gpus = list(range(num_gpus))
        self.gpus_in_use = []
        self.lock = threading.Lock()

    def assign_gpu(self):
        with self.lock:
            if self.available_gpus:
                gpu = self.available_gpus.pop(0)
                self.gpus_in_use.append(gpu)
                return gpu
            else:
                return None

    def release_gpu(self, gpu):
        with self.lock:
            self.gpus_in_use.remove(gpu)
            self.available_gpus.append(gpu)

# Implement a route to check job status
@app.route('/check_job_status', methods=['GET'])
def check_job_status():
    request_id = request.args.get('request_id')

    if request_id in job_statuses:
        status = job_statuses[request_id]
        if status == 'completed':
            processed_image = job_results.get(request_id)
            response_data = {'status': status, 'processed_image': processed_image, 'request_id': request_id}
        else:
            response_data = {'status': status, 'request_id': request_id}
    else:
        response_data = {'status': 'not_found', 'request_id': request_id}

    return jsonify(response_data)

# Define a function to process image requests
def process_image_request(n_gpu):
    # Continuously check the queue for new image requests
    while True:
        if not request_queue.empty():
            request_data = request_queue.get()

            request_id = request_data.get('request_id')  # Get the request ID from the client

            image_data = request_data.get('image')

            # Store the request status as "processing"
            job_statuses[request_id] = 'processing'

            # Handle image processing
            try:
                image = image_data
                time.sleep(10)  # Simulate processing time

                # Convert the processed image to base64-encoded format
                processed_image_data = base64.b64encode(image).decode('utf-8')

                # Store the result and update the job status to "completed"
                job_results[request_id] = processed_image_data
                job_statuses[request_id] = 'completed'
            except Exception as e:
                # If there's an error, store the status as "failed" and log the error
                job_statuses[request_id] = 'failed'
                print(f"Error processing image request with ID: {request_id}: {str(e)}")
            finally:
                request_queue.task_done()

# Implement a route to submit image processing requests
@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()

    image_data = base64.b64decode(data.get('image'))
    request_id = data.get('request_id')  # Get the request ID from the client

    # Add the request data to the queue
    request_queue.put({'request_id': request_id, 'image': image_data})
    job_statuses[request_id] = "waiting"
    
    return jsonify({'message': 'Image processing request submitted successfully'})



num_worker_threads = 2  # You can adjust the number of worker threads
GPU_MANAGER = GPUManager(num_worker_threads)
worker_threads = []

for _ in range(num_worker_threads):
    thread = threading.Thread(target=process_image_request, args=(GPU_MANAGER.assign_gpu(),))
    thread.start()
    worker_threads.append(thread)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

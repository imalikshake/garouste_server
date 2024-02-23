import mediapipe as mp
import cv2
import argparse
import os
# Define a function to crop the face as a square
def crop_face_square(image, landmarks):
    # Calculate the bounding box of the face using landmarks
    # print(landmarks)
    x_min = min(landmark.x for landmark in landmarks.landmark)
    y_min = min(landmark.y for landmark in landmarks.landmark)
    x_max = max(landmark.x for landmark in landmarks.landmark)
    y_max = max(landmark.y for landmark in landmarks.landmark)
    # print(x_min,y_min,x_max,y_max)
    # Calculate the side length of the square
    side_length = max(x_max - x_min, y_max - y_min)

    # Calculate the center of the bounding box
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    # Calculate the top-left corner of the square crop
    crop_x = center_x - side_length // 2
    crop_y = center_y - side_length // 2

    crop_x = int(max(0, crop_x))
    crop_y = int(max(0, crop_y))
    crop_x_end = int(min(crop_x + side_length, image.shape[1]))
    crop_y_end = int(min(crop_y + side_length, image.shape[0]))
    # print(crop_x, crop_y, crop_x_end, crop_y_end)
    # Crop the image
    cropped_face = image[crop_y:crop_y_end, crop_x:crop_x_end]
    return cropped_face

def largest_square_bounding_box(xs, ys, image_width, image_height):
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    # Calculate the center of the bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Calculate the maximum possible side length from the center to the image edges
    max_side_length = min(center_x, image_width - center_x, center_y, image_height - center_y) * 2

    # Calculate the new coordinates of the bounding box
    new_min_x = center_x - max_side_length / 2
    new_max_x = center_x + max_side_length / 2
    new_min_y = center_y - max_side_length / 2
    new_max_y = center_y + max_side_length / 2

    return int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
target_size = (1024,1024)

# Define command-line arguments
parser = argparse.ArgumentParser(description='Process images with face detection and cropping.')
parser.add_argument('--input_dir', type=str, help='Input directory containing images')
parser.add_argument('--output_dir', type=str, help='Output directory to save results')

args = parser.parse_args()

with mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True, min_detection_confidence=0
) as face_mesh:
    # Iterate over images in the input directory
    for img_path in os.listdir(args.input_dir):
        xs = []
        ys = []
        if img_path.endswith('.png') or img_path.endswith('.jpg'):
            image = cv2.imread(os.path.join(args.input_dir, img_path))
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Crop the face as a square
                # cropped_face = crop_face_square(image, face_landmarks)
                for id, lm in enumerate(face_landmarks.landmark):
                    # print(lm)
                    ih, iw, ic = image.shape
                    x,y,z= int(lm.x*iw),int(lm.y*ih),int(lm.z*ic)
                    # print(id,x,y,z)
                    xs.append(x)
                    ys.append(y)
        # print(image.shape)
        # print(len(xs))
        # print(len(ys))
        if (len(xs) == 0):
            print("ERROR: "+ img_path)
            continue
        resized_bbox = largest_square_bounding_box(xs, ys, image.shape[1], image.shape[0])
        x1, y1, x2, y2 = resized_bbox
        cropped_img = image[y1:y2, x1:x2]
        resized_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)
        # Extract filename and extension
        filename, extension = os.path.splitext(img_path)
        # Construct output file path
        output_path = os.path.join(args.output_dir, filename + '.jpg')
        # Save cropped image
        cv2.imwrite(output_path, resized_img)
        


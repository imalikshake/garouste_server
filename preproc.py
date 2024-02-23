#!/usr/bin/env python3

import numpy as np
import cv2
import torch
from transformers import SamModel, SamProcessor
from colors import COLOR_DICT
import os
import argparse
import gc

def create_caption(color, token):
    color = color.lower()
    if color[0] in "aeiou":
        an_color = "an " + color
    else:
        an_color = "a " + color
    return f"A photo of {token} person in front of {an_color} colored background."

def write_caption(img_path, caption):
    caption_file = f"{os.path.splitext(img_path)[0]}.caption"
    with open(caption_file, 'w') as f:
        f.write(caption)

def segment_images(basedir, newdir, colors=2, sizes=4, token="raff"):
    print(sizes)
    COLOR_LIST = list(COLOR_DICT.keys())
    color_counter = 0

    if not os.path.exists(newdir):
        os.mkdir(newdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    for i, filename in enumerate(os.listdir(basedir)):
        # load the image
        fname = filename.split(".")[0]
        image_bgr = cv2.imread(os.path.join(basedir, filename))
        # before doing anything else, let's resize the bgr image, and center-crop it to square
        w, h, _ = image_bgr.shape
        if w != h:
            min_size = min(w, h)
            xl = (w - min_size) / 2
            yl = (h - min_size) / 2
            image_bgr = image_bgr[xl:xl+min_size, yl:yl + min_size]
            w = min_size
            h = min_size
        if w != 1024:
            interp = cv2.INTER_AREA if w > 1024 else cv2.INTER_LANCZOS4
            image_bgr = cv2.resize(image_bgr, (1024, 1024), interpolation = cv2.INTER_AREA)
        max_size = 1024

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # use the center, and a bit below it, to segment
        # note that this requires images to be roughly aligned
        curr_points = [[[500, 500], [500, 400]]]

        # use sam to segment the input image with the specified target points
        # first preprocess
        inputs = processor(image_rgb, input_points=curr_points, return_tensors="pt").to("cuda")
        image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
        # pop the pixel_values as they are not neded
        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})
        # then run the segmentation model
        with torch.no_grad():
            outputs = model(**inputs)

        # use the processor to postprocess the masks
        masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        scores = outputs.iou_scores

        # find the mask with the largest area
        masks_t = torch.squeeze(masks[0])
        masks_batch = list(masks_t)
        masks_areas = [m.sum() for m in masks_batch]
        # show_masks_on_image(resized_image, masks_t, scores)
        # show_points_on_image(resized_image, curr_points[0])
        max_i = masks_areas.index(max(masks_areas))
        mask_final = masks_t[max_i]
        mask_final = mask_final.numpy()
        mask_final = mask_final.astype(np.uint8)
        mask_matrix = np.uint8(mask_final) * 255
        # Invert the mask to keep the object and exclude the background
        inverted_mask = cv2.bitwise_not(mask_matrix)

        for div in range(1, SIZES+1):
            frac_resize = div / SIZES
            size = int(max_size * frac_resize)

            # round-robin the images with the input colors
            # todo find a way of doing this that involves less nesting
            for color_i in range(COLORS_PER_IMAGE):
                color_name = COLOR_LIST[color_counter]
                color_rgb = COLOR_DICT[color_name]
                color_counter = (color_counter + 1) % len(COLOR_LIST)
                # Create a solid color background (e.g., blue)
                background_color = color_rgb  # BGR color (here, it's red)

                # Create a new image with the background color
                background = np.full_like(image_rgb, background_color)
                background = cv2.bitwise_and(background, background, mask=inverted_mask)

                # Extract the object using the original image and the mask
                object_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_final)

                # Combine the object image and the background
                result = cv2.add(object_image, background)
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                color_caption = create_caption(color_name, token)
                color_str = "_".join(color_name.split())
                img_path = f"{fname}-{size}-{color_str.lower()}.jpg"

                # now we actually resize
                resized_result = cv2.resize(result, (size, size), interpolation = cv2.INTER_AREA)

                cv2.imwrite(os.path.join(newdir, img_path), resized_result)
                write_caption(os.path.join(newdir, img_path), color_caption)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input directory", required=True)
    parser.add_argument("--output", type=str, help="output directory", default="output")
    parser.add_argument("--sizes", type=int, help="number of sizes to resize to", default=4)
    parser.add_argument("--colors", type=int, help="number of colors to use per image", default=2)
    parser.add_argument("--token", type=str, help="lora token", default="raff")
    args = parser.parse_args()

    basedir = args.input
    newdir = args.output
    COLORS_PER_IMAGE = args.colors
    SIZES = args.sizes
    token = args.token

    segment_images(basedir=basedir, newdir=newdir, colors=COLORS_PER_IMAGE, sizes=SIZES, token=token)
    torch.cuda.empty_cache()
    gc.collect()

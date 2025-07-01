from transformers import pipeline
from PIL import Image, ImageChops
import numpy as np
import os

# Initialize segmentation pipeline
segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")


def segment_clothing(img, category):
    """
    Segments clothing from an image and returns two images:
    1. A black and white mask of the clothing.
    2. The original image with the clothing parts turned to grayscale.
    """
    img_rgba = img.convert("RGBA")

    # Segment image
    segments = segmenter(img_rgba)
    if category == "upper_body":
        clothes = ["Upper-clothes", "Scarf", "Belt"]
    elif category == "dresses":
        clothes = ["Dress"]
    else:
        clothes = ["Pants", "Skirt"]
    # Create list of masks for clothes
    mask_list = []
    for s in segments:
        if s['label'] in clothes:
            mask_list.append(s['mask'])

    # Create a combined mask for all clothing items
    if mask_list:
        binary_mask = Image.new('L', img_rgba.size, 0)
        for mask in mask_list:
            binary_mask = ImageChops.lighter(binary_mask, mask)
    else:
        # If no clothes detected, use an empty (black) mask
        binary_mask = Image.new('L', img_rgba.size, 0)

    # Create the grayscaled clothes image
    img_rgb = img.convert('RGB')

    # Create a solid gray image to use as the mask overlay
    gray_mask_color = (128, 128, 128)  # A solid gray color
    gray_overlay = Image.new('RGB', img_rgb.size, gray_mask_color)

    # Create the final image by pasting the gray overlay onto the original using the clothing mask
    fine_mask_clothes = img_rgb.copy()
    fine_mask_clothes.paste(gray_overlay, mask=binary_mask)
    binary_mask.save("binary_mask.jpg")
    fine_mask_clothes.save("fine_mask_clothes.jpg")
    return binary_mask, fine_mask_clothes


def batch_segment_clothing(img_dir,
                           out_dir,
                           clothes=[
                               "Hat", "Upper-clothes", "Skirt", "Pants",
                               "Dress", "Belt", "Left-shoe", "Right-shoe",
                               "Scarf"
                           ]):
    # Create output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Iterate through each file in the input directory
    for filename in os.listdir(img_dir):
        if filename.lower().endswith((".jpg", ".png")):
            try:
                # Load image
                img_path = os.path.join(img_dir, filename)
                img = Image.open(img_path)

                # Segment clothing
                mask_img, grayed_img = segment_clothing(img, clothes)

                # Save segmented images to output directory as PNG
                base_filename = os.path.splitext(filename)[0]
                mask_out_path = os.path.join(out_dir,
                                             base_filename + "_mask.jpg")
                grayed_out_path = os.path.join(
                    out_dir, base_filename + "_fine_mask.jpg")

                mask_img.save(mask_out_path)
                grayed_img.save(grayed_out_path)

                print(f"Segmented {filename} successfully.")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        else:
            print(f"Skipping {filename} as it is not a supported image file.")

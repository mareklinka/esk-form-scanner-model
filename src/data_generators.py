from PIL import Image, ImageDraw
import random
import math
from os.path import join
import numpy as np

def generate_examples(original, target_folder, count):
    """
    Generates a batch of training examples
    
    Parameters
    ----------
    original : string
        Path to the image to use as the source image
    target_folder : string
        Path to the folder to store the generated examples in
    count : int
        Number of examples to generate
    """
    
    template = Image.open(original).convert("RGB")

    for i in range(count):
        example = template.copy()

        example, corners = __rotate_image(example)

        for x,y in corners:
            example.putpixel((x ,y), (0, 0, 0))

        example.save(join(target_folder, f"i{i}.jpg"), "JPEG")
        with open(join(target_folder, f"i{i}.txt"), "w") as label:
            for x,y in corners:
                label.write(f"{x},{y} ")

def __rotate_image(example):
    while True:
        angle = math.floor(random.uniform(0, 2)) * 180 + random.uniform(-10, 10)
        result = example.rotate(angle)

        corners = __find_corners(result)

        if len(corners) > 0:
            return result, corners

def __find_corners(image):
    a = np.array(image).T
    upper_left = np.argwhere((a[2, :, :] == 255) & (a[1, :, :] == 0) & (a[0, :, :] == 255))
    lower_left = np.argwhere((a[2, :, :] == 0) & (a[1, :, :] == 255) & (a[0, :, :] == 0))
    upper_right = np.argwhere((a[2, :, :] == 255) & (a[1, :, :] == 255) & (a[0, :, :] == 0))
    lower_right = np.argwhere((a[2, :, :] == 255) & (a[1, :, :] == 0) & (a[0, :, :] == 110))

    if ((upper_left.size == 0) | (lower_left.size == 0) | (upper_right.size == 0) | (lower_right.size == 0)):
        return ()

    return (upper_left[0], lower_left[0], upper_right[0], lower_right[0])
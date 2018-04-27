from PIL import Image, ImageDraw
from os.path import join, isfile, basename
from os import listdir
import re

def draw_bounding_boxes(folder, predictions, output_folder):
    images = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) & f.endswith('.jpg') ]
    labels = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) & f.endswith('.txt') ]

    tuples = list(zip(images, labels, predictions))

    for image_path, label_path, prediction in tuples:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        label = __decode_label_file(label_path)
        true_bounding_box = __to_coordinates(label)
        predicted_bounding_box = __to_coordinates(prediction)

        draw.polygon(true_bounding_box, fill=None, outline="green")
        draw.polygon(predicted_bounding_box, fill=None, outline="red")
        image.save(join(output_folder, basename(image_path)))

def __to_coordinates(label):
    return [label[0], label[1], label[2], label[3], label[6], label[7], label[4], label[5], label[0], label[1]]

def __decode_label_file(path):
    with open(path, "r") as file:
        array = [int(x) for x in re.split('[, ]', file.readline().strip())]

        return array
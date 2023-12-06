import math
import random

from PIL import Image, ImageDraw

from .image_perturbation import get_rotated_image

def change_proportion_to_zero(image, bounding_box):

    modified_image = image.copy()

    draw = ImageDraw.Draw(modified_image)
    draw.rectangle(bounding_box, fill=0)

    return modified_image

def shift_borders(image):

    sx = random.uniform(0, 0.4)
    x_bounding_box = [
        [0, 0, int(image.size[0] * sx), image.size[1]],
        [image.size[0] - int(image.size[0] * sx), 0, image.size[0], image.size[1]]
    ]

    sy = random.uniform(0, 0.4)
    y_bounding_box = [
        [0, 0, image.size[0], int(image.size[1] * sy)],
        [0, image.size[1] - int(image.size[1] * sy), image.size[0], image.size[1]]
    ]

    augmented_image = image.copy()
    augmented_image = change_proportion_to_zero(augmented_image, x_bounding_box[random.randrange(0, 2)])
    augmented_image = change_proportion_to_zero(augmented_image, y_bounding_box[random.randrange(0, 2)])

    return augmented_image

def zoom_out_borders(image):

    sz = random.uniform(0, 0.4) * 0.5
    x_bounding_box = [
        [0, 0, int(image.size[0] * sz), image.size[1]],
        [image.size[0] - int(image.size[0] * sz), 0, image.size[0], image.size[1]]
    ]
    y_bounding_box = [
        [0, 0, image.size[0], int(image.size[1] * sz)],
        [0, image.size[1] - int(image.size[1] * sz), image.size[0], image.size[1]]
    ]

    augmented_image = image.copy()
    augmented_image = change_proportion_to_zero(augmented_image, x_bounding_box[0])
    augmented_image = change_proportion_to_zero(augmented_image, x_bounding_box[1])
    augmented_image = change_proportion_to_zero(augmented_image, y_bounding_box[0])
    augmented_image = change_proportion_to_zero(augmented_image, y_bounding_box[1])

    return augmented_image

def rotation_borders(image):
    radian = random.uniform(-math.pi / 4, math.pi / 4)
    bounding_box = [0, 0, image.size[0], image.size[1]]

    rotated_image = get_rotated_image(image, bounding_box, allow_zero_pixel=True, radian=radian)
    augmented_image = get_rotated_image(rotated_image, [0, 0, rotated_image.size[0], rotated_image.size[1]], allow_zero_pixel=True, radian=-radian)

    return augmented_image

def get_augmented_image(image, type):
    if type == 'shift':
        return shift_borders(image)
    if type == 'zoom-out':
        return zoom_out_borders(image)
    if type == 'rotate':
        return rotation_borders(image)

if __name__ == '__main__':
    image_path = '../../VAPNet/data/sample.jpg'
    image = Image.open(image_path)
    image.show()

    augmented_image = get_augmented_image(image, 'shift')
    augmented_image.show()

    augmented_image = get_augmented_image(image, 'zoom-out')
    augmented_image.show()

    augmented_image = get_augmented_image(image, 'rotate')
    augmented_image.show()
    

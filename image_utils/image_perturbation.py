import math
import random

from PIL import Image
import numpy as np

def get_normalized_bounding_box(box, image_size):
    norm_box = box.copy()
    norm_box[0] = norm_box[0] / image_size[0]
    norm_box[2] = norm_box[2] / image_size[0]
    norm_box[1] = norm_box[1] / image_size[1]
    norm_box[3] = norm_box[3] / image_size[1]
    return norm_box

def update_operator(type):

    # ox, oy, oz, oa
    operator = [0.0, 0.0, 0.0, 0.0]

    if type == 'shift':
        operator[0] = random.uniform(-0.4, 0.4)
        operator[1] = random.uniform(-0.4, 0.4)
            
    elif type == 'zoom-out':
        operator[2] = random.uniform(0, 0.4)

    elif type == 'crop':
        operator[2] = random.uniform(math.sqrt(0.5), math.sqrt(0.8))
        operator[0] = random.uniform(-operator[2] / 2, operator[2] / 2)
        operator[1] = random.uniform(-operator[2] / 2, operator[2] / 2)

    elif type == 'rotate':
        operator[3] = random.uniform(-math.pi/4, math.pi/4)
    
    return operator

def is_not_in_image(boudning_box):
    x1, y1, x2, y2 = boudning_box
    if x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <=1:
        return False
    else:
        return True

def bounding_box_to_box_info(box):
    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    
    box_info = [cx, cy, w, h]
    return box_info

def perturb(box_info, operator):
    cx, cy, w, h = box_info
    ox, oy, oz, oa = operator

    cx_out = cx + w * ox
    cy_out = cy + h * oy
    w_out = w + w * oz
    h_out = h + h * oz

    output = [cx_out, cy_out, w_out, h_out]
    if oa != 0:
        perturbed_norm_box = rotate(output, oa)
    else:
        perturbed_norm_box = output_to_bounding_box(output)

    return perturbed_norm_box
    
def rotate(box_info, oa):
    cx, cy, w, h = box_info
    cos_a = math.cos(oa)
    sin_a = math.sin(oa)
    rotation_matrix = np.array([[cos_a, -sin_a],
                                [sin_a, cos_a]])
    c = np.array([cx, cy])
    v_list = np.array([[-w // 2, -h // 2], [w // 2, -h // 2], [w // 2, h // 2], [-w // 2, h // 2]])
    box = []
    for v in v_list:
        r = np.dot(rotation_matrix, v)
        u = (c + r)
        u = u.tolist()
        box.append(u)
    return box

def output_to_bounding_box(output):
    cx, cy, w, h = output

    x1 = cx - (w / 2)
    y1 = cy - (h / 2)
    x2 = cx + (w / 2)
    y2 = cy + (h / 2)
    
    box = [x1, y1, x2, y2]
    return box

def get_origin_bounding_box(norm_box, image_size):
    new_box = norm_box.copy()
    new_box[0] = norm_box[0] * image_size[0]
    new_box[2] = norm_box[2] * image_size[0]
    new_box[1] = norm_box[1] * image_size[1]
    new_box[3] = norm_box[3] * image_size[1]
    new_box = [int(x) for x in new_box]
    return new_box

def rotate_image(image, bounding_box, allow_zero_pixel, operator):

    def is_not_in_image_rotate(corners, image_size):
        for point in corners:
            if point[0] < 0 or point[1] < 0 or point[0] >= image_size[0] or point[1] >= image_size[1]:
                return True
        return False
    
    def rotate_dot(x, y, oa):
        cos_a = math.cos(oa)
        sin_a = math.sin(oa)
        rotation_matrix = np.array([[cos_a, -sin_a],
                                    [sin_a, cos_a]])
        x, y = np.round(np.dot(rotation_matrix, np.array([x, y])))
        x = int(x)
        y = int(y)
        return x, y

    radian = operator[3]
    box_info = bounding_box_to_box_info(bounding_box)
    rotated_box_corners = perturb(box_info, operator)

    # check the rotated image is in original image
    if allow_zero_pixel == False and is_not_in_image_rotate(rotated_box_corners, image.size):
        return None
    
    # make the rectangle cropped image which contains the rotated image
    rec_corners = (min(x[0] for x in rotated_box_corners), min(x[1] for x in rotated_box_corners), max(x[0] for x in rotated_box_corners), max(x[1] for x in rotated_box_corners))
    norm_rotated_box_corners = [[x[0] - min(y[0] for y in rotated_box_corners), x[1] - min(y[1] for y in rotated_box_corners)] for x in rotated_box_corners]
    rec_image = image.crop(rec_corners)

    # rotate the rectangle image
    rotated_rec_image = rec_image.rotate(radian * (180.0 / math.pi), expand=True)
    rec_centers = [rotated_rec_image.size[0] // 2, rotated_rec_image.size[1] // 2]

    # rotate the rotated image(we want) because the rectangle image was rotated
    rotated_box_corners = [rotate_dot(x[0], x[1], -radian) for x in norm_rotated_box_corners]
    rbc_centers = [(max(x[0] for x in rotated_box_corners) + min(x[0] for x in rotated_box_corners)) // 2, (max(x[1] for x in rotated_box_corners) + min(x[1] for x in rotated_box_corners)) // 2]
    rotated_box_corners = [[x[0] - rbc_centers[0] + rec_centers[0], x[1] - rbc_centers[1] + rec_centers[1]] for x in rotated_box_corners]

    # make bounding box and crop
    bounding_box = [min(x[0] for x in rotated_box_corners), min(x[1] for x in rotated_box_corners), max(x[0] for x in rotated_box_corners), max(x[1] for x in rotated_box_corners)]
    rotated_image = rotated_rec_image.crop(bounding_box)

    return rotated_image

def get_rotated_image(image, bounding_box, allow_zero_pixel, radian=None):

    operator = [0.0, 0.0, 0.0, 0.0]

    if radian == None:
        operator = update_operator('rotate')
    else:
        operator[3] = radian
    
    return rotate_image(image, bounding_box, allow_zero_pixel, operator)


def get_perturbed_image(image, bounding_box, allow_zero_pixel, type):

    if type == 'rotate':
        return get_rotated_image(image, bounding_box, allow_zero_pixel)
    
    norm_box = get_normalized_bounding_box(bounding_box, image.size)
    
    operator = update_operator(type)

    box_info = bounding_box_to_box_info(norm_box)
    perturbed_norm_box = perturb(box_info, operator)
    
    if allow_zero_pixel == False and is_not_in_image(perturbed_norm_box):
        return None

    perturbed_bounding_box = get_origin_bounding_box(perturbed_norm_box, image.size)
    perturbed_crop_image = image.crop(perturbed_bounding_box)

    return perturbed_crop_image

if __name__ == '__main__':
    image_path = '../../VAPNet/data/sample.jpg'
    bounding_box = [200, 100, 300, 250]
    image = Image.open(image_path)
    image.show()
    print(image.size)

    allow_zero_pixel = False

    box = image.crop(bounding_box)
    box.show()

    rotated_image = get_perturbed_image(image, bounding_box, allow_zero_pixel, type='rotate')
    rotated_image.show()

    shifted_image = get_perturbed_image(image, bounding_box, allow_zero_pixel, type='shift')
    shifted_image.show()

    zoomed_out_image = get_perturbed_image(image, bounding_box, allow_zero_pixel, type='zoom-out')
    zoomed_out_image.show()

    cropped_image = get_perturbed_image(image, bounding_box, allow_zero_pixel, type='crop')
    cropped_image.show()
    
import os
import random
import random
import numpy as np
import cv2
from utils import stitch, rotate_image, crop_and_resize, fill_right_bound
from set_params import photos_path, seed, params, intersection, intermediate_res, add_percent, rotate_angle, half_shape

# подготовим данные
bottom, top = [], []
dim_arr = np.array(list(map(lambda x: x.split('_')[1], os.listdir(photos_path))))
bottom_len, top_len = np.unique(dim_arr, return_counts=True)[1]
shape = max(bottom_len, top_len) + 1
for i in range(1, shape):
    if i <= bottom_len:
        bottom.append(os.path.join(photos_path, f'{i}_1.jpg'))
    if i <= top_len:
        top.append(os.path.join(photos_path, f'{i}_2.jpg'))

# зафиксируем сиды
random.seed(seed)
np.random.seed(seed)
cv2.setRNGSeed(seed)

# объединим фотографии по горизонтали разделив пополам
parts = []
for num, i in enumerate([bottom, top]):
    part = len(i) // 2
    for j in range(0, len(i), part):
        panorama = stitch(i[j: j+part+intersection], params)
        cv2.imwrite(f'{num}_{j}.png', panorama) if intermediate_res else 0
        panorama = cv2.rotate(panorama, cv2.ROTATE_90_CLOCKWISE)
        parts.append(panorama)

# объединим разделенные пополам части
images = []
for i in range(2):
    panorama = stitch([parts[i], parts[i+2]], params)
    panorama = cv2.rotate(panorama, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(f'{i}.png', panorama) if intermediate_res else 0
    images.append(panorama)

# объединим фотографии по вертикали
stitched_img = stitch(images, params)
cv2.imwrite(f'joined.png', stitched_img) if intermediate_res else 0

try:
    # попробуем соединить левую и правую части, нужно для точного совпадения краев
    left_img = cv2.flip(stitched_img[:, :half_shape, :], 1)
    right_img = cv2.flip(stitched_img[:, -half_shape:, :], 1)
    params['wave_correct_kind'] = 'no'
    bound = stitch([left_img, right_img], params).shape[1] - half_shape * 2
    stitched_img = stitched_img[:, :bound, :]
    cv2.imwrite(f'merged.png', stitched_img) if intermediate_res else 0
except:
    pass

# обработаем границы
stitched_img = rotate_image(stitched_img, rotate_angle)
cv2.imwrite(f'rotate.png', stitched_img) if intermediate_res else 0
stitched_img = crop_and_resize(stitched_img)
cv2.imwrite(f'cropped.png', stitched_img) if intermediate_res else 0
stitched_img = fill_right_bound(stitched_img)

# добавим наверх и вниз дополнительный цвет
add_part = int(stitched_img.shape[0] * add_percent)
add_arr = np.zeros((add_part, stitched_img.shape[1], 3))
add_arr_top, add_arr_bottom = add_arr.copy(), add_arr.copy()
add_arr_top[:] = stitched_img[:add_part].mean(axis=0)
add_arr_bottom[:] = stitched_img[add_part:].mean(axis=0)
stitched_img = np.concatenate((add_arr_top, stitched_img, add_arr_bottom))

# сохраним результат
cv2.imwrite(f'result_{photos_path}.png', stitched_img)
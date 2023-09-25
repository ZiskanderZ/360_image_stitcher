from stitching import Stitcher
import os
import cv2
import random
import numpy as np
import random

# считаем фотографии
photos_path = r'photos'
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
seed = 1
random.seed(seed)
np.random.seed(seed)
cv2.setRNGSeed(seed)

# объединим фотографии по горизонтали
images = []
for num, i in enumerate([bottom, top]):
    stitcher = Stitcher(medium_megapix=1, confidence_threshold=0.8, crop=True)
    panorama = stitcher.stitch(i)
    panorama = cv2.rotate(panorama, cv2.ROTATE_90_CLOCKWISE)
    images.append(panorama)

# обновим сиды
seed = 1000
random.seed(seed)
np.random.seed(seed)
cv2.setRNGSeed(seed)

# объединим фотографии по вертикали
stitcher = Stitcher(confidence_threshold=0.9, crop=False)
stitched_img = stitcher.stitch(images)
stitched_img = cv2.rotate(stitched_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# обработаем границы
clear_img = stitched_img.copy()
for k in range(3):
    for y, i in enumerate(stitched_img[:, :, k]):
        non_zeros = np.where(i != 0)[0]
        zeros = np.where(i == 0)[0]
        if len(non_zeros) < stitched_img.shape[1] * 0.5:
            if y < (stitched_img.shape[0] / 2):
                top_bound = y + 15
            else:
                bottom_bound = y - 15
            continue
        for x, j in enumerate(i):
            if j == 0:
                if x < (len(i) / 2):
                    clear_img[y, zeros[-1], k] = i[non_zeros[-1]]
                    zeros = zeros[:-1]
                    non_zeros = non_zeros[:-1]
            else:
                break

# удалим ненужные края
crop_img = clear_img[top_bound:bottom_bound, np.mean(clear_img[:, :, 0] == 0, axis=0) < 0.01, :]

# сохраним результат
cv2.imwrite('result.png', crop_img)
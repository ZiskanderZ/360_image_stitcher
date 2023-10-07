import cv2
from stitching import Stitcher
import imutils
import numpy as np

def stitch(imgs: list, params: dict) -> np.array:
    '''
    Функция объединения изображений

    :param imgs массив изображений или путей к иображениям
    :param params словарь параметров сшивателя

    :return объединенное изображение
    '''

    stitcher = Stitcher(**params)
    stitched_img = stitcher.stitch(imgs)
    return stitched_img

def rotate_image(image: np.array, angle: float) -> np.array:
  '''
  Функция поворота изоюражения относительно левой верхней точки
  
  :param image исходное изображение
  :param angle угол поворота

  :return повернутое изображение 
  '''
  
  image_point = (0, 0)
  rot_mat = cv2.getRotationMatrix2D(image_point, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def crop_and_resize(image: np.array) -> np.array:
    '''
    Функция обрезки изображения для удаления черных сегментов сверху и снизу

    :param image исходное изображение

    :return обрезанное изображение
    '''

    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    mask = np.zeros(thresh_img.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)

    minRectangle = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        minRectangle = cv2.erode(minRectangle, None)
        sub = cv2.subtract(minRectangle, thresh_img)

    contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    _, y, _, h = cv2.boundingRect(areaOI)

    return image[y:y + h, x:x + w]

def fill_right_bound(image: np.array) -> np.array:
    '''
    Функция заполнения правого края черных пикселей изображения пикселями левого края

    :param image исходное изображение

    :return изображение без черных пикселей 
    '''

    image = cv2.flip(image, 1)
    clear_img = image.copy()
    for k in range(3):
        for y, i in enumerate(image[:, :, k]):
            non_zeros = np.where(i != 0)[0]
            zeros = np.where(i == 0)[0]
            for x, j in enumerate(i):
                if j == 0:
                    if x < (len(i) / 2):
                        clear_img[y, zeros[-1], k] = i[non_zeros[-1]]
                        zeros = zeros[:-1]
                        non_zeros = non_zeros[:-1]
                else:
                    break
    clear_img = cv2.flip(clear_img, 1)
    return clear_img

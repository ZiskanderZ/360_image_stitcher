# Краткое описание

Решалась задача склейки изображений для получения панорамы в `360°`.

Были попробованы различные подходы и алгоритмы.

Все алгоритмы достаточно хорошо работают для `2-3` фотографий, но если необходимо склеить `10-20-30` фото, то либо они ломаются, либо появляются артефакты в изображениях. Ломаются в большинтсве своем т.к. камера для получения панорамы в `360°` постепенно поворачивается, что является для алгоритмов сложностью.

Была попробована для всех подходов склейка различными методами: последовательная склейка, склейка по частям и последующее объединение, склейка всех сразу.

# Технологии, которые были попробованы

## Классические методы

### 1. OpenCV

```
import cv2
imageStitcher = cv2.Stitcher_create()
error, stitched_img = imageStitcher.stitch(images)
```

Не получилось склеить, т.к. библиотека не гибкая, нельзя просто настроить склейку, по порогу, количеству точек совпадения и т.п. Зачастую получалась ошибка без объяснения причин. Также один из минусов - долгая обработка, в результате которой могут склеиться пару последних фотографий.

### 2. Stitching

```
from stitching import Stitcher
stitcher = Stitcher()
stitched_img = stitcher.stitch(images)
```

Библиотека очень гибкая, позволяет настраивать все необходимые параметры. Также есть возможность отслеживать результаты с помощью сохранения соответствий фотографий друг другу.

С помощью данной библиотеки был реализован основной [подход](https://github.com/ZiskanderZ/360_image_stitcher) и выложен на `git`.

Склейка довольно быстро работает, но появляются артефакты (затемнение, блюр). Также библиотека не умеет склеивать начало и конец полученной панорамы. Это необходимо для того чтобы получить `360°`. Не нашел реализаций в интернете и пришлось реализовать самому скрипт склейки начала и конца, из-за которого по краям изображения также появляются артефакты.

## Transformers

Я не нашел таких трансформеров, которые решают задачу склейки изображений, но некоторые из них можно настроить на это.

### 1. LOFTR

```
from kornia.contrib import ImageStitcher
IS = ImageStitcher(KF.LoFTR(pretrained="outdoor"), estimator="ransac")
out = IS(*imgs)
```

Очень крутой инструмент для поиска точек и склейки обычной панорамы из `2-3` фото. При большем количестве фотографий необходимо большие ресурсы, пробовал склеить `20` фотографий, но `24ГБ GPU` не хватило. Также ломается при большом количестве фото, начинает накладывать друг на друга фотографии.

### 2. Coarse_LoFTR_TRT

Облегченная версия `LOFTR`, но я не смог разобраться как его использовать для склейки фото.

### 3. Superpoint + superglue

Тоже крутые инструменты для поиска и матчинга точек, но также не смог настроить, чтобы они помогли в склейке панаромы `360°`.

## Библиотеки на С++

Не пробовал реализовывать

1. https://sourceforge.net/p/hugin/hugin/ci/default/tree/
2. https://sourceforge.net/p/panotools/libpano13/ci/default/tree/
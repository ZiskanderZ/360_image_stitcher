# путь к папке с фотографиями
photos_path = r'photos'

# параметр для воспроизводимости результатов
seed = 1

# параметры сшивателя
params = {'confidence_threshold': 0.7, # порог для выбора точек (уменьшать при ошибке "No match exceeds the given confidence threshold.", 
                                       #                         увеличивать при ошибке "Camera parameters adjusting failed.")
          'nfeatures': 1500000,        # количество совпадающих точек на фото (чем больше, тем лучше)
          'blend_strength': 0.3,       # число для степени блюра. Чем больше тем, больше блюрятся границы сшивания
          'crop': True}

# влияет на сшивание на строчке 34. При ошибке на данной строчке увеличивать/уменьшать данный параметр (0, 1, 2, 3 ..)
intersection = 2

# сохранять ли промежуточные результаты
intermediate_res = False

# параметр, показывающий какую долю дополнительного цвета добавить
add_percent = 0.15

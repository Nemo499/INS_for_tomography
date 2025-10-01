from onnxruntime import InferenceSession
from pathlib import Path
from PIL import Image
from sys import argv
import numpy as np
import pydicom
import os

# Загрузка модели ONNX
def loadModel(model_path, info = False):
    try:
        session = InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print("Модель загружена успешно из", model_path)

        if info:
            print("Вход модели:")
            for input_meta in session.get_inputs():
                print(f"Имя: {input_meta.name}, Размеры: {input_meta.shape}, Тип: {input_meta.type}")
            print("Выход модели:")
            for output_meta in session.get_outputs():
                print(f"Имя: {output_meta.name}, Размеры: {output_meta.shape}, Тип: {output_meta.type}")

        return session
    except Exception as e:
        print("Ошибка загрузки модели.", e)
        return None

# Загрузка изображения
def load_image_robust(file_path):
        """
        Загружает изображение, пытаясь сначала прочитать его как DICOM,
        а затем, в случае ошибки, как стандартный графический файл.
        """
        try:
            # Пытаемся прочитать как DICOM-файл
            ds = pydicom.dcmread(file_path, force=True)
            img = ds.pixel_array.astype(np.float32)

            # Применяем Rescale Slope / Intercept
            slope = getattr(ds, 'RescaleSlope', 1.0)
            intercept = getattr(ds, 'RescaleIntercept', 0.0)
            img = img * slope + intercept

            # Если есть Window Center / Width, применяем windowing
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                center = ds.WindowCenter
                width = ds.WindowWidth

                if isinstance(center, pydicom.multival.MultiValue):
                    center = float(center[0])
                if isinstance(width, pydicom.multival.MultiValue):
                    width = float(width[0])
                
                min_val = center - width 
                max_val = center + width
                img = np.clip(img, min_val, max_val)
            
            # Нормализация данных к диапазону 0-255
            img = (img - img.min()) / (img.max() - img.min()) * 255.0
            img = img.astype(np.uint8)

            return Image.fromarray(img).convert('RGB')
        
        except Exception as dicom_e:
            # Если не DICOM, пробуем как обычное изображение
            try:
                return Image.open(file_path).convert('RGB')
            except Exception as img_e:
                raise FileNotFoundError(f"Не удалось загрузить файл как DICOM или изображение: {file_path}. Ошибка: {img_e}")

# Парсинг папки с изображениями
def parseDataFolder(data_folder):
    import glob
    valid_extensions = ('.dcm', '.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    file_list = []
    
    for ext in valid_extensions:
        file_list.extend(glob.glob(os.path.join(data_folder, f'*{ext}')))
    
    if not file_list:
        raise FileNotFoundError("В папке нет изображений с поддерживаемыми расширениями.")
    
    return file_list

# Импорт набора изображений из директории
def importImageSet(img_paths, img_size):
    img_array = np.empty((0, img_size[0], img_size[1], img_size[2]), dtype=np.float32)
    try:
        for img_path in img_paths:
            img = load_image_robust(img_path)
            img = img.resize(img_size[1:])
            img = np.array(img).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img_array = np.append(img_array, [img], axis=0)

        print(f"Импортировано {len(img_array)} изображений.")
        return img_array
    except Exception as e:
        print("Ошибка загрузки изображений из директории:", e)
        return None

# Запуск модели с входными данными
def runModel(session, input_data):
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    # Проверка и подготовка входных данных
    if input_data is None:
        print("Ошибка загрузки данных, используется случайный набор данных.")
        input_data = np.random.rand(*input_shape).astype(np.float32)
    else:
        if input_data.shape[1:] != tuple(input_shape[1:]):
            raise ValueError(f"Размеры входных данных ({input_data.shape}) не соответствуют ожидаемым ({input_shape}).")
        else:
            # input_data = input_data.astype(np.float32)
            print("Входные данные загружены успешно.")

    # Запуск сессии анализа
    try:
        print(f"Запуск анализа... ")
        outputs = session.run(None, {input_name: input_data})
    except Exception as e:
        print("Ошибка во время выполнения анализа.", e)
        return None

    print("Анализ завершён успешно.")
    print(f"Вывод модели: {len(outputs[0])} элементов.")
    return outputs

# Обработка результатов анализа
def preprocessOutputs(outputs, paths):
    classes = len(outputs[0][0])
    print(f"Классов: {classes}")
    class_arr = [[] for _ in range(classes)]
    for i in range(len(outputs[0])):
        class_ = np.argmax(outputs[0][i])
        class_arr[class_].append(paths[i])
    return class_arr

def main(model_path, paths):
    try:
        session = loadModel(model_path)
        images = importImageSet(paths, session.get_inputs()[0].shape[1:])
        result = runModel(session, images)
        return preprocessOutputs(result, paths)
    except Exception as e:
        print(e, "\nЗавершение.")

# Копирование файлов в указанную директорию
def copyFiles(file_list, destination):
    import shutil
    for file in file_list:
        img_name = os.path.basename(file)
        dest_path = destination / img_name
        try:
            shutil.copy(file, dest_path)
        except Exception as e:
            print(f"Ошибка сохранения файла {file} в {dest_path}: {e}")

folder = ""
if len(argv) > 1:
    folder = argv[1]

ROOT = Path(__file__).parent.parent.parent
INPUT_DIR = ROOT / 'images' /folder
OUTPUT_DIR = ROOT / 'results' / str(INPUT_DIR).split('\\')[-1]

NORMAL = OUTPUT_DIR / 'normal'
CAP = OUTPUT_DIR / 'CAP'
COVID = OUTPUT_DIR / 'covid'

NORMAL.mkdir(parents=True, exist_ok=True)
CAP.mkdir(parents=True, exist_ok=True)
COVID.mkdir(parents=True, exist_ok=True)

m_is_chest = "models\ourneOurFinal.onnx"
m_norm_pathology = "models\CNN9.onnx"
m_pathology_type = "models\densenet_ct.onnx"


not_chest, chest = main(m_is_chest, parseDataFolder(INPUT_DIR))
print(f"Не грудные: {len(not_chest)}, Грудные: {len(chest)}")

pathology, norma = main(m_norm_pathology, chest)
print(f"Патология: {len(pathology)}, Норма: {len(norma)}")

norma_, cap, covid = main(m_pathology_type, pathology)
if len(norma_)>0:
    norma.append(norma_)
print(f"Норма: {len(norma)}, CAP: {len(cap)}, COVID: {len(covid)}")

copyFiles(norma, NORMAL)
copyFiles(cap, CAP)
copyFiles(covid, COVID)

print('Готово!')

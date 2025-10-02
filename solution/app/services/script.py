from onnxruntime import InferenceSession
from openpyxl import Workbook
from pathlib import Path
from PIL import Image
from sys import argv
import numpy as np
import os, shutil, pydicom


# Загрузка модели ONNX
def loadModel(model_path, info = False):
    """
    Загрузка модели ONNX с возможностью вывода информации о входах и выходах.
    Args:
        model_path (str): Путь к файлу модели ONNX.
        info (bool): Если True, выводит информацию о входах и выходах модели.
    Returns:
        InferenceSession: Сессия ONNX Runtime или None в случае ошибки.
    """
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
        Args:
            file_path (str): Путь к файлу изображения.
        Returns:
            PIL.Image: Загруженное изображение в формате RGB.
        Raises:
            FileNotFoundError: Если файл не удалось загрузить ни как DICOM, ни как изображение.
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
    """"
    Парсит папку и возвращает список путей к изображениям с поддерживаемыми расширениями.
    Поддерживаемые расширения: .dcm, .jpg, .jpeg, .png, .bmp, .tiff
    Args:
        data_folder (str): Путь к папке с изображениями.
    Returns:
        list of str: Список путей к изображениям.
    Raises:
        FileNotFoundError: Если в папке нет изображений с поддерживаемыми расширениями
    """
    import glob
    valid_extensions = ('','.dcm', '.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    file_list = []
    
    for ext in valid_extensions:
        file_list.extend(glob.glob(os.path.join(data_folder, f'*{ext}')))
    
    if not file_list:
        raise FileNotFoundError("В папке нет изображений с поддерживаемыми расширениями.")
    
    return file_list

def normalize(img, target_size=(224, 224)):
    """
    Нормализует и преобразовывает изображение для подачи на вход модели.
    Args:
        img (PIL.Image): Входное изображение.
        target_size (tuple): (H, W) — целевые высота и ширина изображения.
    Returns:
        np.ndarray: Нормализованное изображение формы (C, H, W), dtype=float32
    """
    H, W = target_size

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    w, h = img.size
    short_side = 256

    if w < h:
        new_w = short_side
        new_h = int(h * short_side / w)
    else:
        new_h = short_side
        new_w = int(w * short_side / h)

    img = img.resize((new_w, new_h))
    left = (new_w - W) // 2
    top = (new_h - H) // 2
    img = img.crop((left, top, left + W, top + H))

    # Преобразуем в numpy
    img = np.array(img).astype(np.float32) / 255.0  # [0,1]

    # Нормализация
    img = (img - mean) / std

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    return img

# Импорт набора изображений из директории
def importImageSet(img_paths, input_shape=(3, 224, 224)):
    """
    Загружает набор изображений (включая DICOM) и подготавливает их для DenseNet ONNX/PyTorch.

    Args:
        img_paths (list of str): Список путей к изображениям.
        target_size (tuple): (C, H, W) — размер входа модели.

    Returns:
        np.ndarray: Массив изображений формы (N, C, H, W), dtype=float32
    """
    imgs = []

    for img_path in img_paths:
        try:
            img = load_image_robust(img_path)
            img = normalize(img, input_shape[1:])
            imgs.append(img)

        except Exception as e:
            print(f"Ошибка при загрузке {img_path}: {e}")
    
    if not imgs:
        print("Не удалось загрузить ни одного изображения.")
        return None
    
    img_array = np.stack(imgs, axis=0)
    print(f"Импортировано {len(img_array)} изображений.")
    return img_array

# Запуск модели с входными данными
def runModel(session, input_data):
    """
    Запускает модель ONNX с заданными входными данными.
    Args:
        session (InferenceSession): Сессия ONNX Runtime.
        input_data (np.ndarray): Входные данные формы (N, C, H, W).
    Returns:
        tuple: (outputs, [times, proc_statuses])
            outputs (list of np.ndarray): Список выходов модели.
            times (list of float): Время обработки каждого изображения.
            proc_statuses (list of str): Статус обработки каждого изображения ("Success" или "Failure").
    """
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
            print("Входные данные загружены успешно.")

    times = []
    outputs = []
    confidences = []
    proc_statuses = []
    from time import perf_counter
    for i in range(len(input_data)):
        try:
            x = input_data[i].reshape(1, *input_data[i].shape).astype(np.float32)
            start = perf_counter()
            output = session.run(None, {input_name: x})
            end = perf_counter()
            
            out_array = output[0][0]  
            e_x = np.exp(out_array - np.max(out_array))
            prob = e_x / e_x.sum()
            conf = float(np.max(prob))

            times.append(end - start)
            outputs.append(output[0][0])
            confidences.append(conf)
            proc_statuses.append("Success")
            # print(f"Изображение {i+1}/{len(input_data)} обработано за {end - start:.4f} секунд.")
        except Exception as e:
            print(f"Ошибка при обработке изображения {i}: {e}")
            times.append(0)
            outputs.append(None)
            proc_statuses.append("Failure")

    print("Анализ завершён.")
    print(f"Вывод модели: {len(outputs)} элементов.")
    return outputs, [times, proc_statuses,confidences]

# Обработка результатов анализа
def preprocessOutputs(outputs, paths):
    """
    Обрабатывает выходы модели, группируя пути к изображениям по предсказанным классам.
    Args:
        outputs (list of np.ndarray): Список выходов модели.
        paths (list of str): Список путей к изображениям.
    Returns:
        list of list of str: Массив списков путей, сгруппированных по классам.
    """
    classes = len(outputs[0])
    print(f"Классов: {classes}")
    class_arr = [[] for _ in range(classes)]
    for i in range(len(outputs)):
        class_ = np.argmax(outputs[i])
        class_arr[class_].append(paths[i])
    return class_arr

def main(model_path, paths):
    """
    Запуск рабочего цикла модели.
    Args:
        model_path (str): Путь к файлу модели ONNX.
        paths (list of str): Список путей к изображениям.
    Returns:
        tuple: (classes_arr, [times, proc_statuses])
            classes_arr (list of list of str): Массив списков путей, сгруппированных по классам.
            times (list of float): Время обработки каждого изображения.
            proc_statuses (list of str): Статус обработки каждого изображения ("Success" или "Failure").
    """
    try:
        session = loadModel(model_path)
        images = importImageSet(paths, session.get_inputs()[0].shape[1:])
        result, meta = runModel(session, images)
        classes_arr = preprocessOutputs(result, paths)
        return (classes_arr, meta)
    except Exception as e:
        print(e, "\nЗавершение.")

# Извлечение метаданных DICOM
def parse_DICOM_Data(file_paths):
    """
    Извлекает StudyInstanceUID и SeriesInstanceUID из DICOM файлов.
    Args:
        file_paths (list of str): Список путей к DICOM файлам.
    Returns:
        tuple: (study_uids, series_uids)
            study_uids (list of str or None): Список StudyInstanceUID или None, если не удалось извлечь.
            series_uids (list of str or None): Список SeriesInstanceUID или None, если не удалось извлечь.
    """
    study_uids = []
    series_uids = []
    for file_path in file_paths:
        try:
            ds = pydicom.dcmread(file_path, force=True)
            study_uids.append(ds.StudyInstanceUID if 'StudyInstanceUID' in ds else None)
            series_uids.append(ds.SeriesInstanceUID if 'SeriesInstanceUID' in ds else None)
        except Exception as e:
            print(f"Ошибка чтения DICOM тэгов файла {file_path}: {e}")
            study_uids.append(None)
            series_uids.append(None)
    return study_uids, series_uids

# Копирование файлов в указанную директорию
def copyFiles(file_list, destination):
    """
    Копирует файлы из списка в памяти в директорию destination.
    Args:
        file_list (list of str): Список путей к файлам для копирования.
        destination (Path): Путь к директории назначения.
    """
    for file in file_list:
        img_name = os.path.basename(file)
        dest_path = destination / img_name
        try:
            shutil.copy(file, dest_path)
        except Exception as e:
            print(f"Ошибка сохранения файла {file} в {dest_path}: {e}")

# Создание Excel отчёта
def createExcelReport(report_dict, filename):
    """
    Создаёт Excel отчёт из словаря данных.
    Args:
        report_dict (dict): Словарь, где ключи — названия столбцов, а значения — списки данных.
        filename (str or Path): Путь к файлу для сохранения отчёта.
    """
    wb = Workbook()
    sheet = wb.active
    sheet.title = "Отчёт"

    headers = list(report_dict.keys())
    sheet.append(headers)

    rows = zip(*report_dict.values())
    for row in rows:
        sheet.append(row)

    wb.save(filename)
    print("Отчёт сохранён в", filename)

folder = ""
if len(argv) > 1:
    folder = argv[1]

# Подготовка директорий
ROOT = Path(__file__).parent.parent.parent # /solution
INPUT_DIR = ROOT / 'images' / folder
OUTPUT_DIR = ROOT / 'results' / folder
REPORTS = ROOT / 'reports'

NORMAL = OUTPUT_DIR / 'normal'
CAP = OUTPUT_DIR / 'CAP'
COVID = OUTPUT_DIR / 'covid'

NORMAL.mkdir(parents=True, exist_ok=True)
CAP.mkdir(parents=True, exist_ok=True)
COVID.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

m_is_chest = "models\ourneOurFinal.onnx"
m_norm_pathology = "models\CNN9.onnx"
m_pathology_type = "models\densenet_ct.onnx"

# Основной процесс
files = parseDataFolder(INPUT_DIR)
study_uids, series_uids = parse_DICOM_Data(files)
[not_chest, chest], [times1, statuses1, _] = main(m_is_chest, files)

print(f"Не грудные: {len(not_chest)}, Грудные: {len(chest)}")
chest_or_not = [True if f in chest else False for f in files]

[pathology, norma], [times2, statuses2,prob] = main(m_norm_pathology, chest)
print(f"Патология: {len(pathology)}, Норма: {len(norma)}")

if(len(pathology)>0):
    [norma_, cap, covid], [times3, statuses3,_] = main(m_pathology_type, pathology)
    if len(norma_)>0:
        norma.append(norma_)
    print(f"Норма: {len(norma)}, CAP: {len(cap)}, COVID: {len(covid)}")

# Составление наборов метаданных
times = []
statuses = []
p_types = []
norm_or_not = []

for i in range(len(files)):
    if files[i] in norma:
        p_types.append("-")
        norm_or_not.append(0)
    elif files[i] in cap:
        p_types.append("CAP")
        norm_or_not.append(1)
    elif files[i] in covid:
        p_types.append("COVID")
        norm_or_not.append(1)
    
    times.append(times1[i])
    times[i] += times2[chest.index(files[i])] if files[i] in chest else 0
    times[i] += times3[pathology.index(files[i])] if files[i] in pathology else 0

    statuses.append(statuses1[i])
    if (statuses[i] != 'Failure'):
        if (files[i] in chest):
            statuses[i] = statuses2[chest.index(files[i])]
            if (files[i] in pathology):
                statuses[i] = statuses3[pathology.index(files[i])]

# Создание отчёта
report = {
    "chest_study": chest_or_not,
    "path_to_study": files,
    "study_uid": study_uids,
    "series_uid": series_uids,
    "probability_of_pathology":prob,
    "pathology": norm_or_not,
    "pathology_type": p_types,
    "processing_status": statuses,
    "time_of_processing": times
}

createExcelReport(report, REPORTS / f'{folder}.xlsx')

copyFiles(norma, NORMAL)
copyFiles(cap, CAP)
copyFiles(covid, COVID)

print('Готово!')

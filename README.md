# [Задача №8] Решение от команды CADAI

## Описание решения
Веб-приложение для сортировки изображений с помощью нейросети.  
Стек технологий:  
- **Node.js** – бэкенд  
- **Vue.js** – фронтенд  
- **Docker + docker-compose** – контейнеризация  
- **NGINX** – обратный прокси и раздача статики  
- **Python + ONNX** – инференс сессии анализа и составление отчёта


## Возможности и ограничения



##  Системные требования

### На устройстве должны быть установлены:

- [Docker](https://www.docker.com/get-started) 
- [Node.js](https://nodejs.org/) (версии 18 или новее)
- [npm](https://www.npmjs.com/) (для управления зависимостями)

## Быстрый старт

### 1. Установите необходимые библиотеки
Перед первым запуском необходимо запустить файл install reqs.bat для установки необходимых модулей Python.

### 2. Build the Docker Image
```bash
docker build -t <image-name> .
```

### 3. Run the Docker Container
```bash
docker run -p 80:80 --name <container-name> <image-name>
```

Приложение запущено на `http://localhost:80`.


### 4. Development Setup
Для локального запуска без Docker:

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run serve
```

##  Структура проекта

```
├── Dockerfile           # Конфигурация Docker
├── app/                 # 
├── app/package.json     # Node.js dependencies and scripts
├── reports/             # Сформированные отчёты
├── results/             # Отсортированные данные
├── install_reqs.bat     # Установка модулей Python
├── Run.bat              # Запуск программы
└── README.md            # 
```

### Встроенные команды
- `npm run serve` - Start the Vue.js development server
- `npm run build` - Build the application for production
- `npm run lint`  - Run linting checks

### Команды Docker
- Stop the container: `docker stop <container-name>`
- Remove the container: `docker rm <container-name>`
- Rebuild the image: `docker build -t <image-name> .`
# Базовый образ с Python (если нужны зависимости)
FROM python:3.11-slim

# Рабочая директория
WORKDIR /app

# Копируем зависимости Python
COPY requirements.txt .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Устанавливаем Nginx
RUN apt-get update && apt-get install -y nginx

# Копируем твой конфиг Nginx
COPY solution/app/nginx/conf/nginx.conf /etc/nginx/nginx.conf

# Запуск Nginx в foreground
CMD ["nginx", "-g", "daemon off;"]

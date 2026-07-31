# Используем легковесный Python-образ
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Установка системных зависимостей, необходимых для работы OpenCV и YOLO
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код и папку с весами модели
COPY . .

# Открываем порт для приложения
EXPOSE 8000

# Запуск FastAPI через Uvicorn (подходит для продакшена)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
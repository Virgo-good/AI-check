# Car Damage Detection API

Микросервис компьютерного зрения для автоматизации диагностики автомобилей.

## Архитектура
* **Язык**: Python 3.10
* **Фреймворк**: FastAPI
* **Модель**: YOLOv8 (Ultralytics)

## Запуск через Docker

1. Поместите веса обученной модели в папку `weights/` под именем `best.pt`.
2. Соберите образ:
   `docker build -t car-damage-api .`
3. Запустите контейнер:
   `docker run -d --name cv-service -p 8000:8000 --restart always car-damage-api`

## API Эндпоинты

### `POST /predict`
Принимает изображение автомобиля и возвращает координаты повреждений.

**Запрос:**
* `Content-Type`: `multipart/form-data`
* `Body`: поле `image` (файл)

**Пример использования (cURL):**
`curl -X POST -F "image=@car_bumper.jpg" http://localhost:8000/predict`

**Успешный ответ (200 OK):**
```json
{
  "defects": [
    {
      "class": "scratch",
      "bbox": [120, 45, 230, 90],
      "confidence": 0.88
    }
  ]
}

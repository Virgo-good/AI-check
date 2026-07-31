from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image
import io
import time
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Car Damage Detection API",
    description="Микросервис для детекции повреждений кузова (YOLOv8)",
    version="1.0.0"
)

# Глобальная переменная для модели
model = None


@app.on_event("startup")
def load_model():
    """Загрузка модели при старте сервера в оперативную память."""
    global model
    try:
        # Укажите путь к вашим обученным весам
        model = YOLO("weights/best.pt")
        logger.info("Модель YOLOv8 успешно загружена.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Эндпоинт для инференса.
    Принимает фото автомобиля, возвращает JSON с дефектами.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением (jpg, png, webp)")

    start_time = time.time()

    try:
        # Чтение изображения
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Инференс модели
        results = model(img)

        # Постпроцессинг результатов
        defects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                # Фильтрация только нужных классов (если в модели их больше)
                if class_name in ["scratch", "dent", "crack"]:
                    defects.append({
                        "class": class_name,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": round(conf, 2)
                    })

        process_time = time.time() - start_time
        logger.info(f"Обработка завершена за {process_time:.2f} сек. Найдено дефектов: {len(defects)}")

        return {"defects": defects}

    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера при обработке изображения")
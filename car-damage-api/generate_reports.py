from ultralytics import YOLO


def generate_grant_reports():
    # Загружаем обученную модель
    model = YOLO("weights/best.pt")

    # Запускаем валидацию на тестовом датасете
    # Результаты (графики, матрицы) автоматически сохранятся в папку runs/detect/val
    metrics = model.val(data="dataset.yaml", split="test", save_json=True, plots=True)

    print("=== МЕТРИКИ ДЛЯ ОТЧЕТА ===")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print("Графики (Confusion Matrix, PR-curves) сохранены в директории 'runs/detect/val/'")


if __name__ == "__main__":
    generate_grant_reports()
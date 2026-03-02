import numpy as np
import cv2
import os
from datetime import datetime

def generate_good_paint_surface(width=640, height=640, variation=0):
    """
    Генерация серой текстурированной поверхности без дефектов
    
    Args:
        width: ширина изображения
        height: высота изображения
        variation: параметр для создания вариаций текстуры
    
    Returns:
        numpy array: изображение серой текстурированной поверхности
    """
    # Базовый серый цвет (с небольшими вариациями)
    base_color = 220 + variation * 3
    base_color = max(200, min(240, base_color))
    
    # Создаем базовое изображение
    img = np.full((height, width), base_color, dtype=np.uint8)
    
    # Добавляем текстуру "апельсиновой корки"
    for _ in range(3000):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        radius = np.random.randint(1, 3)
        intensity = np.random.randint(0, 20)
        cv2.circle(img, (x, y), radius, base_color + intensity - 10, -1)
    
    # Добавляем линии для имитации поверхности
    line_count = 150 + variation * 20
    for _ in range(line_count):
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        x2 = x1 + np.random.randint(50, 200)
        y2 = y1 + np.random.randint(-20, 20)
        thickness = np.random.randint(1, 2)
        intensity = np.random.randint(-15, 15)
        cv2.line(img, (x1, y1), (x2, y2), base_color + intensity, thickness)
    
    # Добавляем легкий градиент
    for y in range(height):
        alpha = 0.95 + 0.05 * np.sin(y / 50.0)
        for x in range(width):
            img[y, x] = int(img[y, x] * alpha)
    
    return img

def add_scratch_defect(img, intensity=0.8):
    """Добавляет царапину на поверхность"""
    height, width = img.shape
    
    # Случайный выбор точки начала царапины
    x1 = np.random.randint(100, width - 100)
    y1 = np.random.randint(100, height - 100)
    
    # Случайный выбор длины и направления царапины
    length = np.random.randint(80, 150)
    angle = np.random.uniform(-0.5, 0.5) * np.pi  # Угол в радианах
    
    x2 = int(x1 + length * np.cos(angle))
    y2 = int(y1 + length * np.sin(angle))
    
    # Случайная интенсивность царапины
    intensity = np.random.randint(10, 25)
    if np.random.random() > 0.5:
        intensity = -intensity
    
    # Добавляем царапину
    cv2.line(img, (x1, y1), (x2, y2), img.mean() + intensity, 2)
    
    return img

def add_spot_defect(img):
    """Добавляет пятно на поверхность"""
    height, width = img.shape
    
    # Случайный выбор центра пятна
    x = np.random.randint(150, width - 150)
    y = np.random.randint(150, height - 150)
    
    # Случайный выбор радиуса пятна
    radius = np.random.randint(30, 60)
    
    # Случайная интенсивность пятна
    intensity = np.random.randint(15, 35)
    if np.random.random() > 0.5:
        intensity = -intensity
    
    # Добавляем пятно
    cv2.circle(img, (x, y), radius, img.mean() + intensity, -1)
    
    return img

def add_flow_defect(img):
    """Добавляет потек краски на поверхность"""
    height, width = img.shape
    
    # Случайный выбор центра потека
    x = np.random.randint(200, width - 200)
    y = np.random.randint(200, height - 200)
    
    # Создаем потек
    pts = np.array([
        [x - 50, y],
        [x, y - 30],
        [x + 50, y],
        [x, y + 40]
    ], np.int32)
    
    # Случайная интенсивность потека
    intensity = np.random.randint(10, 25)
    if np.random.random() > 0.5:
        intensity = -intensity
    
    # Добавляем потек
    cv2.fillPoly(img, [pts], img.mean() + intensity)
    
    return img

# Создаем папки для сохранения изображений
os.makedirs("data/paint/good_samples", exist_ok=True)
os.makedirs("data/paint/defect_samples", exist_ok=True)

# Генерируем 50 хороших образцов
print("🔄 Генерация 50 хороших образцов без дефектов...")
for i in range(50):
    img = generate_good_paint_surface(variation=i)
    cv2.imwrite(f"data/paint/good_samples/good_{i:03d}.png", img)

# Генерируем 3 образца с дефектами
print("🔄 Генерация 3 образцов с дефектами...")
defect_types = ["scratch", "spot", "flow"]
for i, defect_type in enumerate(defect_types):
    # Создаем базовое изображение
    img = generate_good_paint_surface(variation=50 + i)
    
    # Добавляем дефект
    if defect_type == "scratch":
        img = add_scratch_defect(img)
    elif defect_type == "spot":
        img = add_spot_defect(img)
    else:  # flow
        img = add_flow_defect(img)
    
    # Сохраняем
    cv2.imwrite(f"data/paint/defect_samples/{defect_type}_{i:03d}.png", img)

# Генерируем демонстрационные изображения
print("🔄 Генерация демонстрационных изображений...")
demo_dir = "data/paint/demo_images"
os.makedirs(demo_dir, exist_ok=True)

# Хорошее изображение
good_demo = generate_good_paint_surface()
cv2.imwrite(f"{demo_dir}/good_demo.png", good_demo)

# Изображения с дефектами
for i, defect_type in enumerate(defect_types):
    img = generate_good_paint_surface(variation=60 + i)
    if defect_type == "scratch":
        img = add_scratch_defect(img)
    elif defect_type == "spot":
        img = add_spot_defect(img)
    else:
        img = add_flow_defect(img)
    
    cv2.imwrite(f"{demo_dir}/{defect_type}_demo.png", img)

print("✅ Генерация завершена!")
print(f"   - 50 хороших образцов сохранено в: data/paint/good_samples/")
print(f"   - 3 образца с дефектами сохранено в: data/paint/defect_samples/")
print(f"   - Демонстрационные изображения сохранено в: data/paint/demo_images/")
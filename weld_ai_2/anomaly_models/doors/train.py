"""
anomaly_models/doors/train.py
Обучение модели аномалий для дверей с генеративной аугментацией
(только на хороших данных + симуляция реальных условий)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from datetime import datetime
import cv2

# ==================== ПАРАМЕТРЫ ОБУЧЕНИЯ ====================
INPUT_SIZE = (640, 640)
BATCH_SIZE = 8  # Уменьшаем батч для стабильности на небольших датасетах
EPOCHS = 50     # Уменьшаем эпохи для демо (можно увеличить до 100 при наличии данных)
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
ANOMALY_THRESHOLD = 0.35  # Порог для обнаружения аномалии

# ==================== ГЕНЕРАТИВНАЯ АУГМЕНТАЦИЯ ДАННЫХ ====================
class GenerativeAugmentation:
    """
    Генеративная аугментация данных с симуляцией реальных условий производства
    
    Ключевые техники:
    1. Освещение: симуляция различных условий освещения на производстве
    2. Угол съемки: симуляция разных ракурсов без поворота изображения
    3. Гарантированный фиксированный размер 640x640 для всех изображений
    """
    
    @staticmethod
    def get_transforms():
        """Генеративная аугментация данных с гарантированным фиксированным размером"""
        return transforms.Compose([
            # 1. Гарантированное изменение размера до 700x700 для последующей обрезки
            transforms.Resize((700, 700)),
            
            # 2. Симуляция разных условий освещения (применяется всегда)
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.15),
            
            # 3. Симуляция разных углов съемки (без поворота) - ГАРАНТИРОВАННО для всех изображений
            transforms.RandomPerspective(
                distortion_scale=0.25, 
                p=0.7,  # Применяется к 70% изображений
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            
            # 4. Случайная обрезка до точного размера 640x640 (ГАРАНТИРОВАННО для всех изображений)
            transforms.RandomCrop(INPUT_SIZE),
            
            # 5. Дополнительные аугментации для устойчивости
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            
            # 6. Нормализация
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ==================== ПРЕДПРОЦЕССИНГ ИЗОБРАЖЕНИЙ ====================
class GoodSamplesDataset(Dataset):
    """Датасет только хороших образцов (без дефектов) с генеративной аугментацией"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        
        # Собираем пути ко всем изображениям
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(root, file))
        
        if not self.image_paths:
            raise ValueError(f"❌ Не найдено изображений в директории: {data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # Загружаем изображение в RGB (гарантируем 3 канала)
            img = Image.open(img_path).convert('RGB')
            
            # Применяем генеративную аугментацию
            if self.transform:
                img = self.transform(img)
            
            # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: гарантируем фиксированный размер
            if img.shape != (3, INPUT_SIZE[0], INPUT_SIZE[1]):
                # Если размер неверный, применяем финальный ресайз
                from torchvision.transforms.functional import resize
                img = resize(img, INPUT_SIZE)
            
            return img
            
        except Exception as e:
            print(f"⚠️ Ошибка загрузки изображения {img_path}: {str(e)}")
            # Возвращаем заглушку правильного размера при ошибке
            return torch.zeros(3, INPUT_SIZE[0], INPUT_SIZE[1])

# ==================== АРХИТЕКТУРА АВТОЭНКОДЕРА ====================
class Autoencoder(nn.Module):
    """Автоэнкодер для обнаружения аномалий"""
    
    def __init__(self, input_size=INPUT_SIZE):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ==================== ОБУЧЕНИЕ МОДЕЛИ ====================
def train_anomaly_model(data_dir, output_dir="anomaly_models/doors"):
    """Обучение модели аномалий для дверей с генеративной аугментацией"""
    print("🚀 Загрузка хороших образцов дверей с генеративной аугментацией...")
    
    # Трансформации с генеративной аугментацией
    transform = GenerativeAugmentation.get_transforms()
    
    # Создаем датасет
    dataset = GoodSamplesDataset(data_dir, transform=transform)
    print(f"✅ Найдено {len(dataset)} хороших образцов с генеративной аугментацией")
    
    # Разделение на обучающую и валидационную выборки
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Даталоадеры (ИСПРАВЛЕНО: num_workers=0 для Windows)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Создаем модель
    print("🚪 Создаем автоэнкодер для дверей с генеративной аугментацией...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    print(f"✅ Используем устройство: {device}")
    
    # Оптимизатор и функция потерь
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Обучение
    print(f"🔥 Начинаем обучение модели ({EPOCHS} эпох) с генеративной аугментацией...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # Обучение
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for inputs in train_loader:
            # Пропускаем батчи с нулевыми тензорами (ошибки загрузки)
            if torch.all(inputs == 0):
                continue
                
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        if batch_count == 0:
            print("⚠️ Пропущена эпоха: все батчи содержали ошибки загрузки")
            continue
            
        # Валидация
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            for inputs in val_loader:
                if torch.all(inputs == 0):
                    continue
                    
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
                val_batch_count += 1
        
        # Вычисление средних потерь
        train_loss = train_loss / batch_count if batch_count > 0 else 0.0
        val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0.0
        
        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        
        # Логирование
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Оценка на валидационной выборке
    print("🔍 Оценка на валидационной выборке с генеративной аугментацией...")
    model.eval()
    mse_values = []
    
    with torch.no_grad():
        for inputs in val_loader:
            if torch.all(inputs == 0):
                continue
                
            inputs = inputs.to(device)
            outputs = model(inputs)
            mse = torch.mean((inputs - outputs) ** 2, dim=[1, 2, 3])
            mse_values.extend(mse.cpu().numpy())
    
    # Определение порога аномалии
    if len(mse_values) == 0:
        raise ValueError("❌ Не удалось вычислить MSE: валидационная выборка пуста")
    
    mse_values = np.array(mse_values)
    anomaly_threshold = np.mean(mse_values) + 2 * np.std(mse_values)
    print(f"🛡️ Порог аномалии: {anomaly_threshold:.4f}")
    
    # Сохранение модели и конфигурации
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем финальную модель
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    
    # Сохраняем метрики
    config = {
        "anomaly_threshold": float(anomaly_threshold),
        "input_size": INPUT_SIZE,
        "training_history": {
            "train_loss": train_losses,
            "val_loss": val_losses
        },
        "validation_metrics": {
            "mean_mse": float(np.mean(mse_values)),
            "std_mse": float(np.std(mse_values)),
            "sample_count": len(mse_values)
        },
        "training_date": datetime.now().isoformat(),
        "augmentation_info": "Генеративная аугментация: освещение и угол съемки (без поворота)",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "transform_pipeline": "Resize(700) → ColorJitter → RandomPerspective → RandomCrop(640) → Blur → Normalize"
    }
    
    with open(os.path.join(output_dir, "config.json"), "w", encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Модель сохранена в: {output_dir}")
    print("🎉 Обучение модели аномалий для дверей завершено!")
    
    return model, config

# ==================== ЗАПУСК ОБУЧЕНИЯ ====================
if __name__ == "__main__":
    # Путь к данным (только хорошие образцы)
    data_dir = "data/doors/good_samples"
    
    # Проверка существования директории с данными
    if not os.path.exists(data_dir):
        print(f"⚠️ Директория с данными не найдена: {data_dir}")
        print("Создаём демо-данные для демонстрации...")
        
        # Создаём демо-данные
        os.makedirs(data_dir, exist_ok=True)
        
        # Генерируем 10 синтетических изображений "хороших дверей"
        for i in range(10):
            # Создаём изображение двери с текстурой
            img = np.ones((640, 640, 3), dtype=np.uint8) * 240
            
            # Добавляем текстуру двери
            cv2.rectangle(img, (100, 50), (540, 590), (220, 220, 220), -1)
            cv2.rectangle(img, (120, 70), (520, 570), (200, 200, 200), -1)
            
            # Добавляем ручку
            cv2.circle(img, (480, 320), 25, (180, 180, 180), -1)
            
            # Добавляем текстуру дерева/металла
            for y in range(70, 570, 30):
                cv2.line(img, (120, y), (520, y), (190, 190, 190), 1)
            
            # Сохраняем изображение
            cv2.imwrite(os.path.join(data_dir, f"door_{i:03d}.jpg"), img)
        
        print(f"✅ Создано 10 демо-изображений в: {data_dir}")
    
    # Обучение модели
    try:
        model, config = train_anomaly_model(data_dir)
        
        # Проверка работы модели
        print("\n🔍 Тестирование модели на демо-изображении...")
        
        # Создаём демо-изображение для теста
        test_img = np.ones((640, 640, 3), dtype=np.uint8) * 240
        cv2.rectangle(test_img, (100, 50), (540, 590), (220, 220, 220), -1)
        test_img_pil = Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        
        # Применяем трансформации
        transform = GenerativeAugmentation.get_transforms()
        test_image_tensor = transform(test_img_pil).unsqueeze(0)
        
        # Проверка размера тензора
        print(f"   - Размер тензора: {test_image_tensor.shape}")
        
        # Переносим на устройство модели
        device = next(model.parameters()).device
        test_image_tensor = test_image_tensor.to(device)
        
        # Предсказание
        with torch.no_grad():
            reconstruction = model(test_image_tensor)
            mse = torch.mean((test_image_tensor - reconstruction) ** 2)
        
        print(f"   - Входное изображение: {test_image_tensor.shape}")
        print(f"   - Восстановленное: {reconstruction.shape}")
        print(f"   - MSE: {mse.item():.4f}")
        print(f"   - Порог аномалии: {config['anomaly_threshold']:.4f}")
        print(f"   - Вывод: {'АНОМАЛИЯ' if mse.item() > config['anomaly_threshold'] else 'НОРМА'}")
        
        print("\n✅ Обучение и тестирование завершены успешно!")
        
    except Exception as e:
        print(f"\n❌ Критическая ошибка при обучении: {str(e)}")
        import traceback
        traceback.print_exc()
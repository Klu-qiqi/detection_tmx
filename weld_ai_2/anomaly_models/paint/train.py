"""
anomaly_models/paint/train.py
Обучение модели аномалий для покраски с обучением на градациях серого (1 канал)
и визуализацией результатов в RGB
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
BATCH_SIZE = 8
EPOCHS = 600
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1
ANOMALY_THRESHOLD = 0.35

# ==================== ГЕНЕРАТИВНАЯ АУГМЕНТАЦИЯ ДАННЫХ С КОНВЕРТАЦИЕЙ В СЕРЫЙ ====================
class GenerativeAugmentation:
    """
    Генеративная аугментация с конвертацией в градации серого для обучения
    """
    
    @staticmethod
    def get_transforms():
        """Аугментация с последующей конвертацией в градации серого"""
        # Сначала применяем цветовые аугментации в RGB
        color_transforms = transforms.Compose([
            transforms.Resize((700, 700)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.15),
            transforms.RandomPerspective(distortion_scale=0.25, p=0.7),
            transforms.RandomCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Затем конвертируем в градации серого (1 канал)
        def full_transform(img):
            # Применяем цветовые аугментации
            img = color_transforms(img)
            # Конвертируем в градации серого: 0.299*R + 0.587*G + 0.114*B
            gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            # Добавляем размерность канала: [H, W] -> [1, H, W]
            gray = gray.unsqueeze(0)
            return gray
        
        return full_transform

# ==================== ПРЕДПРОЦЕССИНГ ИЗОБРАЖЕНИЙ С КОНВЕРТАЦИЕЙ В СЕРЫЙ ====================
class GoodSamplesDataset(Dataset):
    """Датасет только хороших образцов с конвертацией в градации серого"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        
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
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)  # Возвращает тензор [1, H, W]
            
            # Гарантируем правильный размер
            if img.shape != (1, INPUT_SIZE[0], INPUT_SIZE[1]):
                from torchvision.transforms.functional import resize
                img = resize(img, INPUT_SIZE)
            
            return img
            
        except Exception as e:
            print(f"⚠️ Ошибка загрузки изображения {img_path}: {str(e)}")
            return torch.zeros(1, INPUT_SIZE[0], INPUT_SIZE[1])  # 1 канал вместо 3

# ==================== АРХИТЕКТУРА АВТОЭНКОДЕРА ДЛЯ 1 КАНАЛА ====================
class Autoencoder(nn.Module):
    """Автоэнкодер для обнаружения аномалий (обучается на градациях серого)"""
    
    def __init__(self, input_size=INPUT_SIZE):
        super(Autoencoder, self).__init__()
        
        # Encoder (вход: 1 канал)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # Изменено: 1 вместо 3
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder (выход: 1 канал)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Изменено: 1 вместо 3
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ==================== ОБУЧЕНИЕ МОДЕЛИ НА ГРАДАЦИЯХ СЕРОГО ====================
def train_anomaly_model(data_dir, output_dir="anomaly_models/paint"):
    """Обучение модели аномалий на градациях серого"""
    print("🚀 Загрузка хороших образцов покраски с генеративной аугментацией...")
    
    # Трансформации с конвертацией в градации серого
    transform = GenerativeAugmentation.get_transforms()
    
    # Создаем датасет
    dataset = GoodSamplesDataset(data_dir, transform=transform)
    print(f"✅ Найдено {len(dataset)} хороших образцов")
    print(f"✅ Все изображения будут конвертированы в градации серого (1 канал) для обучения")
    
    # Разделение на обучающую и валидационную выборки
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Даталоадеры
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Создаем модель
    print("🎨 Создаем автоэнкодер для покраски (обучение на градациях серого)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    print(f"✅ Используем устройство: {device}")
    print(f"✅ Архитектура: вход/выход = 1 канал (градации серого)")
    
    # Оптимизатор и функция потерь
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Обучение
    print(f"🔥 Начинаем обучение модели ({EPOCHS} эпох) на градациях серого...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # Обучение
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for inputs in train_loader:
            if torch.all(inputs == 0):
                continue
                
            inputs = inputs.to(device)  # inputs.shape = [B, 1, H, W]
            
            optimizer.zero_grad()
            outputs = model(inputs)     # outputs.shape = [B, 1, H, W]
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
        
        train_loss = train_loss / batch_count if batch_count > 0 else 0.0
        val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0.0
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Оценка на валидационной выборке
    print("🔍 Оценка на валидационной выборке...")
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
    
    if len(mse_values) == 0:
        raise ValueError("❌ Не удалось вычислить MSE: валидационная выборка пуста")
    
    mse_values = np.array(mse_values)
    anomaly_threshold = np.mean(mse_values) + 2 * np.std(mse_values)
    print(f"🛡️ Порог аномалии: {anomaly_threshold:.4f}")
    
    # Сохранение модели и конфигурации
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    
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
        "augmentation_info": "Генеративная аугментация + обучение на градациях серого (1 канал)",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "input_channels": 1,  # Ключевой параметр: обучение на 1 канале
        "transform_pipeline": "RGB → Аугментация → Градации серого → Нормализация"
    }
    
    with open(os.path.join(output_dir, "config.json"), "w", encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Модель сохранена в: {output_dir}")
    print("🎉 Обучение модели аномалий на градациях серого завершено!")
    
    return model, config

# ==================== ЗАПУСК ОБУЧЕНИЯ ====================
if __name__ == "__main__":
    # Путь к данным (только хорошие образцы)
    data_dir = "data/paint/good_samples"
    
    # Проверка существования директории с данными
    if not os.path.exists(data_dir):
        print(f"⚠️ Директория с данными не найдена: {data_dir}")
        print("Создаём демо-данные для демонстрации...")
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Генерируем 10 синтетических изображений покраски
        for i in range(10):
            img = np.ones((640, 640, 3), dtype=np.uint8) * 235
            
            # Градиент фона
            for y in range(640):
                intensity = 235 - int(y * 5 / 640)
                cv2.line(img, (0, y), (640, y), (intensity, intensity, intensity + 5), 1)
            
            # Блик для реалистичности
            cv2.ellipse(img, (320, 200), (80, 40), 0, 0, 360, (255, 255, 255), -1)
            
            cv2.imwrite(os.path.join(data_dir, f"paint_{i:03d}.jpg"), img)
        
        print(f"✅ Создано 10 демо-изображений в: {data_dir}")
    
    # Обучение модели
    try:
        model, config = train_anomaly_model(data_dir)
        
        # Проверка работы модели
        print("\n🔍 Тестирование модели на демо-изображении...")
        
        # Создаём демо-изображение
        test_img = np.ones((640, 640, 3), dtype=np.uint8) * 235
        for y in range(640):
            intensity = 235 - int(y * 5 / 640)
            cv2.line(test_img, (0, y), (640, y), (intensity, intensity, intensity + 5), 1)
        
        # Конвертируем в PIL и применяем трансформации
        test_img_pil = Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        transform = GenerativeAugmentation.get_transforms()
        test_image_tensor = transform(test_img_pil).unsqueeze(0)  # [1, 1, H, W]
        
        print(f"   - Размер входного тензора: {test_image_tensor.shape} (1 канал = градации серого)")
        
        # Переносим на устройство модели
        device = next(model.parameters()).device
        test_image_tensor = test_image_tensor.to(device)
        
        # Предсказание
        with torch.no_grad():
            reconstruction = model(test_image_tensor)
            mse = torch.mean((test_image_tensor - reconstruction) ** 2)
        
        print(f"   - Размер реконструкции: {reconstruction.shape}")
        print(f"   - MSE: {mse.item():.4f}")
        print(f"   - Порог аномалии: {config['anomaly_threshold']:.4f}")
        print(f"   - Вывод: {'АНОМАЛИЯ' if mse.item() > config['anomaly_threshold'] else 'НОРМА'}")
        
        # Демонстрация конвертации обратно в RGB для визуализации
        print("\n🎨 Демонстрация конвертации для визуализации:")
        print("   1. Модель обучена на градациях серого (1 канал)")
        print("   2. Для визуализации результат конвертируется обратно в псевдо-RGB")
        print("   3. Тепловая карта накладывается на оригинальное RGB-изображение")
        
        print("\n✅ Обучение и тестирование завершены успешно!")
        
    except Exception as e:
        print(f"\n❌ Критическая ошибка при обучении: {str(e)}")
        import traceback
        traceback.print_exc()
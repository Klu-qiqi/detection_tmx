"""
anomaly_detector.py
Интеграция обученных моделей аномалий (на PyTorch) с поддержкой 1 канала (градации серого)
Полностью рабочая версия без ошибок нормализации
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import json
from PIL import Image

class Autoencoder(nn.Module):
    """
    Автоэнкодер для обнаружения аномалий (обучен на градациях серого - 1 канал)
    """
    
    def __init__(self, input_channels=1, input_size=(640, 640)):
        super(Autoencoder, self).__init__()
        
        # Encoder (вход: 1 канал для градаций серого)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
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
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AnomalyDetector:
    """
    Детектор аномалий для разных категорий
    Использует обученные модели, обученные ТОЛЬКО на хороших данных в градациях серого
    """
    
    def __init__(self):
        self.models = {}
        self.configs = {}
        self.categories = ["paint"]#, "doors", "glass", "interior"]
        
        # Загружаем все модели
        for category in self.categories:
            model_dir = f"anomaly_models/{category}"
            model_path = os.path.join(model_dir, "best_model.pth")
            config_path = os.path.join(model_dir, "config.json")
            
            if os.path.exists(model_path) and os.path.exists(config_path):
                try:
                    # Загружаем конфигурацию
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    # Определяем количество входных каналов (по умолчанию 1)
                    input_channels = config.get("input_channels", 1)
                    
                    # Создаем модель с правильным количеством каналов
                    model = Autoencoder(input_channels=input_channels)
                    
                    # Загружаем веса
                    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                    model.load_state_dict(state_dict)
                    model.eval()
                    
                    self.models[category] = model
                    self.configs[category] = config
                    
                    # Создаем трансформации для этой категории (1 канал)
                    self._create_transforms(category, input_channels)
                    
                    print(f"✅ Модель аномалий загружена для категории: {category} (каналов: {input_channels})")
                except Exception as e:
                    print(f"❌ Ошибка загрузки модели для {category}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️ Модель аномалий не найдена для категории: {category} (путь: {model_path})")
    
    def _create_transforms(self, category, input_channels):
        """
        Создает трансформации для конкретной категории в зависимости от количества каналов
        """
        if input_channels == 1:
            # Трансформации для градаций серого (1 канал)
            transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),  # Для 'L' изображения вернет [1, H, W]
                # Нормализация для градаций серого: усредненные параметры из ImageNet
                transforms.Normalize(mean=[0.449], std=[0.226])
            ])
        else:
            # Трансформации для RGB (3 канала) - для совместимости
            transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Сохраняем трансформации в отдельный атрибут для каждой категории
        if not hasattr(self, 'transforms'):
            self.transforms = {}
        self.transforms[category] = transform
    
    def detect_anomaly(self, image, category="paint"):
        """
        Детекция аномалии для заданной категории с обучением на градациях серого
        
        Args:
            image: BGR изображение (numpy array)
            category: Категория анализа (paint, doors, glass, interior)
            
        Returns:
            annotated_img: Изображение с визуализацией аномалии в RGB
            anomaly_score: Степень аномалии (0.0 - 1.0)
            is_anomaly: Булево значение наличия аномалии
        """
        # Проверяем наличие модели
        if category not in self.models:
            print(f"⚠️ Модель не найдена для категории: {category}")
            return image.copy(), 0.0, False
        
        try:
            # Сохраняем исходный размер изображения
            original_size = (image.shape[1], image.shape[0])  # (width, height)
            
            # Конвертируем BGR -> Grayscale напрямую (без промежуточного RGB)
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Создаем PIL Image в режиме 'L' (градации серого)
            img_pil = Image.fromarray(img_gray).convert('L')
            
            # Применяем трансформации для 1 канала
            if category in self.transforms:
                transform = self.transforms[category]
            else:
                # Резервная трансформация для 1 канала
                transform = transforms.Compose([
                    transforms.Resize((640, 640)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.449], std=[0.226])
                ])
            
            img_tensor = transform(img_pil).unsqueeze(0)  # [1, 1, H, W]
            
            # Проверка размера тензора
            if img_tensor.shape[1] != 1:
                raise ValueError(f"Ожидался 1 канал, получено: {img_tensor.shape[1]}")
            
            # Получаем устройство модели
            device = next(self.models[category].parameters()).device
            img_tensor = img_tensor.to(device)
            
            # Предсказание
            with torch.no_grad():
                reconstruction = self.models[category](img_tensor)  # [1, 1, H, W]
                mse = torch.mean((img_tensor - reconstruction) ** 2)
                mse = mse.item()
            
            # Определение аномалии
            anomaly_threshold = self.configs[category]["anomaly_threshold"]
            is_anomaly = mse > anomaly_threshold
            
            # Создаем аннотированное изображение (копия оригинала в BGR)
            annotated_img = image.copy()
            
            # Цвета в зависимости от уровня аномалии
            if is_anomaly:
                if mse > anomaly_threshold * 1.5:
                    color = (0, 0, 255)  # Красный - критическая аномалия
                else:
                    color = (0, 165, 255)  # Оранжевый - средняя аномалия
            else:
                color = (0, 255, 0)  # Зеленый - норма
            
            # Добавляем текст на изображение
            status_text = "АНОМАЛИЯ" if is_anomaly else "НОРМА"
            cv2.putText(annotated_img, status_text, (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # ===== ВИЗУАЛИЗАЦИЯ: Конвертация разницы в тепловую карту =====
            # 1. Вычисляем разницу в пространстве градаций серого
            diff = np.abs(img_tensor.cpu().numpy()[0, 0] - reconstruction.cpu().numpy()[0, 0])  # [H, W]
            diff = (diff * 255).astype(np.uint8)  # Нормализация в диапазон 0-255
            
            # 2. Создаем тепловую карту из разницы
            diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)  # [H, W, 3] в BGR
            
            # 3. Изменяем размер тепловой карты до исходного размера изображения
            diff_colored_resized = cv2.resize(diff_colored, original_size, interpolation=cv2.INTER_LINEAR)
            
            # 4. Накладываем тепловую карту на оригинальное BGR-изображение
            if diff_colored_resized.shape[:2] == annotated_img.shape[:2]:
                annotated_img = cv2.addWeighted(annotated_img, 0.7, diff_colored_resized, 0.3, 0)
                
                # Добавляем легенду тепловой карты
                legend_height = 30
                legend = np.zeros((legend_height, original_size[0], 3), dtype=np.uint8)
                
                # Генерируем градиент для легенды
                for x in range(original_size[0]):
                    ratio = x / original_size[0]
                    color_val = int(ratio * 255)
                    color = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
                    legend[:, x] = color
                
                # Добавляем текст легенды
                cv2.putText(legend, "Низкая", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(legend, "Высокая", (original_size[0] - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Накладываем легенду на нижнюю часть изображения
                annotated_img[-legend_height:] = legend
            else:
                print(f"⚠️ Размеры не совпадают: {annotated_img.shape} vs {diff_colored_resized.shape}")
            # ============================================================
            
            return annotated_img, float(mse), is_anomaly
            
        except Exception as e:
            print(f"❌ Ошибка при детекции аномалии: {str(e)}")
            import traceback
            traceback.print_exc()
            return image.copy(), 0.0, False
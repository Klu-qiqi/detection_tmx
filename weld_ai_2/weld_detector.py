"""
weld_detector.py
Модуль детекции дефектов сварных швов на основе вашей существующей модели
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

class WeldDefectDetector:
    """Детектор дефектов сварных швов на основе обученной модели YOLO"""
    
    def __init__(self, model_path="models/best_weld_ai.pt"):
        self.model_path = model_path
        self.model = self._load_model()
        
        # Маппинг классов в категории и критичность
        self.class_mapping = {
            "Bad Welding": {"category": "Сварные швы", "severity": "Критичный", "impact": "Высокое — нарушение прочности соединения"},
            "Crack": {"category": "Сварные швы", "severity": "Критичный", "impact": "Критическое — риск разрушения конструкции"},
            "Excess Reinforcement": {"category": "Сварные швы", "severity": "Средний", "impact": "Среднее — нарушение геометрии шва"},
            "Good Welding": {"category": "Сварные швы", "severity": "Низкий", "impact": "Нет — соответствие стандартам"},
            "Porosity": {"category": "Сварные швы", "severity": "Средний", "impact": "Среднее — снижение прочности шва"},
            "Spatters": {"category": "Сварные швы", "severity": "Низкий", "impact": "Низкое — косметический дефект"}
        }
    
    def _load_model(self):
        """Загрузка модели с обработкой ошибок"""
        try:
            if os.path.exists(self.model_path):
                model = YOLO(self.model_path)
                print(f"✅ Модель сварки загружена: {self.model_path}")
                return model
            else:
                print(f"⚠️ Модель не найдена: {self.model_path}. Используется yolov8n-seg.pt")
                return YOLO("yolov8n-seg.pt")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return None
    
    def detect(self, image):
        """
        Детекция дефектов сварных швов
        
        Args:
            image: BGR изображение
            
        Returns:
            annotated_img: Изображение с аннотациями
            defects: Список обнаруженных дефектов
        """
        if self.model is None:
            return image, []
        
        # Конвертация в RGB для YOLO
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Детекция
        results = self.model(image_rgb, conf=0.3)
        annotated_img = results[0].plot()
        
        # Извлечение дефектов
        defects = []
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = results[0].names[cls_id]
            conf = float(box.conf[0])
            
            # Пропускаем "хорошие" швы при анализе дефектов
            if cls_name == "Good Welding" and conf < 0.9:
                continue
            
            # Получение метаданных класса
            class_info = self.class_mapping.get(cls_name, {
                "category": "Сварные швы",
                "severity": "Средний",
                "impact": "Среднее"
            })
            
            # Определение цвета для визуализации
            color_map = {
                "Критичный": (0, 0, 255),    # Красный
                "Средний": (0, 165, 255),    # Оранжевый
                "Низкий": (255, 0, 0)        # Синий
            }
            color = color_map.get(class_info["severity"], (255, 0, 0))
            
            defects.append({
                "id": i + 1,
                "bbox": [x1, y1, x2, y2],
                "class": cls_name,
                "category": class_info["category"],
                "confidence": conf,
                "severity": class_info["severity"],
                "impact": class_info["impact"],
                "color": color
            })
        
        return annotated_img, defects
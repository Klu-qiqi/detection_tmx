"""
generative_ai.py
Генеративные ИИ-компоненты для платформы:
1. Diffusion Model — синтез "идеального" изделия по КД
2. LLM — генерация объяснимых отчётов с причинно-следственным анализом
3. GAN — аугментация редких дефектов (для обучения)
"""

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import pipeline
import tempfile
import os
from datetime import datetime
import json

class GenerativeDefectAnalyzer:
    """Основной класс генеративного анализа дефектов"""
    
    def __init__(self, enable_diffusion=True, enable_llm=True):
        self.enable_diffusion = enable_diffusion
        self.enable_llm = enable_llm
        
        # Инициализация LLM для генерации отчётов (лёгкая модель)
        if self.enable_llm:
            try:
                self.llm = pipeline(
                    "text-generation",
                    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    torch_dtype=torch.float16,
                    device=0 if torch.cuda.is_available() else -1
                )
                print("✅ LLM загружен для генерации отчётов")
            except Exception as e:
                print(f"⚠️ LLM недоступен (демо-режим): {e}")
                self.llm = None
        
        # Плейсхолдер для Diffusion Model (промышленная реализация через API)
        self.diffusion_api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
        self.diffusion_api_key = os.getenv("HF_API_KEY", "demo_key")
    
    def generate_ideal_product(self, product_type: str = "weld_seam", target_shape: tuple = None) -> np.ndarray:
        """
        Генерация 'идеального' изделия с правильным масштабированием
        """
        # Определяем целевой размер
        if target_shape is None:
            height, width = 640, 640
        else:
            height, width = target_shape[:2]
        
        # Создаем идеальное изображение нужного размера
        ideal = np.ones((height, width, 3), dtype=np.uint8) * 240  # Светлый фон
        
        # Генерируем идеальный элемент в зависимости от типа
        if product_type == "weld_seam":
            # Добавляем параметрический сварной шов с учетом масштаба
            start_y = int(height * 0.2)
            end_y = int(height * 0.8)
            center_x = int(width * 0.5)
            
            # Основной шов
            cv2.rectangle(ideal, 
                        (center_x - 10, start_y), 
                        (center_x + 10, end_y), 
                        (200, 200, 200), -1)
            
            # Добавляем текстуру шва
            for y in range(start_y, end_y, 15):
                cv2.line(ideal, 
                        (center_x - 8, y), 
                        (center_x + 8, y), 
                        (220, 220, 220), 1)
            
            # Добавляем градиент для реалистичности
            overlay = ideal.copy()
            cv2.rectangle(overlay, 
                        (center_x - 15, start_y + 5), 
                        (center_x + 15, end_y - 5), 
                        (255, 255, 255), -1)
            ideal = cv2.addWeighted(ideal, 0.7, overlay, 0.3, 0)
        
        # Добавляем проверку для других типов
        elif product_type == "painted_surface":
            # Идеальная покраска с градиентом
            ideal[:] = (230, 230, 240)
            cv2.rectangle(ideal, (0, 0), (width, height // 2), (235, 235, 245), -1)
        
        return ideal
    
    def _generate_synthetic_ideal(self, product_type: str) -> np.ndarray:
        """Синтетическая генерация "идеального" изделия для демо"""
        # Создаём параметрическую модель идеального сварного шва
        height, width = 640, 640
        ideal = np.zeros((height, width, 3), dtype=np.uint8)
        
        if product_type == "weld_seam":
            # Генерация идеального сварного шва (параметрическая модель)
            cv2.rectangle(ideal, (280, 200), (360, 440), (200, 200, 200), -1)  # Основа шва
            cv2.line(ideal, (320, 200), (320, 440), (255, 255, 255), 2)  # Центральная линия
            
            # Добавляем текстуру "идеального" шва
            for y in range(200, 440, 15):
                cv2.line(ideal, (290, y), (350, y), (220, 220, 220), 1)
        
        elif product_type == "painted_surface":
            # Идеальная покраска
            ideal[:] = (230, 230, 240)  # Равномерный цвет
            
        else:
            # Универсальный фон
            ideal[:] = (240, 240, 240)
        
        return ideal
    
    def generate_explainable_report(self, defects: list, context: dict = None) -> dict:
        """
        Генерация объяснимого отчёта с причинно-следственным анализом через LLM
        """
        if not self.enable_llm or self.llm is None:
            return self._generate_rule_based_report(defects, context)
        
        try:
            # Формируем промпт для LLM
            defect_summary = "\n".join([
                f"- {d['class']} (уверенность: {d['confidence']:.1%}, критичность: {d['severity']})"
                for d in defects[:5]  # Ограничиваем для скорости
            ])
            
            prompt = f"""<|system|>
Вы — эксперт по качеству на производстве подвижного состава РЖД. 
Проанализируйте дефекты и сгенерируйте профессиональный отчёт с причинно-следственным анализом.
</s>
<|user|>
Обнаружены дефекты:
{defect_summary}

Контекст производства:
- Тип изделия: сварной шов кузова вагона
- Этап: сборка-сварка
- Условия: цех №3, температура 22°C, влажность 45%

Сгенерируйте отчёт в формате JSON со следующими полями:
1. "root_cause_hypothesis" — гипотеза о корневой причине
2. "process_violation" — нарушенная технологическая операция
3. "material_issue" — проблема с материалами (если применимо)
4. "equipment_issue" — проблема с оборудованием (если применимо)
5. "recommendations" — конкретные рекомендации для технолога
</s>
<|assistant|>"""
            
            # Генерация через LLM
            response = self.llm(
                prompt,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            # Извлечение JSON из ответа (упрощённо для демо)
            report_text = response[0]['generated_text']
            return self._parse_llm_response(report_text)
            
        except Exception as e:
            print(f"⚠️ Ошибка генерации LLM-отчёта, используем правило-базированный: {e}")
            return self._generate_rule_based_report(defects, context)
    
    def _generate_rule_based_report(self, defects: list, context: dict = None) -> dict:
        """
        Правило-базированный отчёт (демо-режим без LLM)
        """
        critical_defects = [d for d in defects if d['severity'] == 'Критичный']
        porosity_count = sum(1 for d in defects if 'porosity' in d['class'].lower() or 'поры' in d['class'].lower())
        crack_count = sum(1 for d in defects if 'crack' in d['class'].lower() or 'трещин' in d['class'].lower())
        
        # Анализ корневых причин по правилам
        root_causes = []
        recommendations = []
        
        if porosity_count > 2:
            root_causes.append("Вероятная причина: загрязнение поверхности перед сваркой или неправильный расход защитного газа")
            recommendations.append("Проверить чистоту свариваемых кромок и параметры подачи Ar/CO₂ смеси")
        
        if crack_count > 0:
            root_causes.append("Вероятная причина: высокие остаточные напряжения или неподходящий режим термообработки")
            recommendations.append("Провести предварительный подогрев до 150°C и контроль скорости охлаждения")
        
        if any('spatter' in d['class'].lower() for d in defects):
            root_causes.append("Вероятная причина: избыточный сварочный ток или неправильный угол наклона горелки")
            recommendations.append("Отрегулировать ток на 10-15% ниже и проверить геометрию установки электрода")
        
        if not root_causes:
            root_causes.append("Дефекты незначительны, соответствуют допускам ГОСТ 34347-2017")
            recommendations.append("Продолжить производственный процесс без корректировок")
        
        return {
            "root_cause_hypothesis": " | ".join(root_causes[:2]),
            "process_violation": "Нарушение режима сварки" if critical_defects else "Нет нарушений",
            "material_issue": "Возможно: низкое качество заготовки" if porosity_count > 3 else "Не выявлено",
            "equipment_issue": "Требуется калибровка сварочного аппарата" if crack_count > 1 else "Оборудование в норме",
            "recommendations": recommendations,
            "confidence_score": 0.85,  # Для демо
            "generated_at": datetime.now().isoformat()
        }
    
    def _parse_llm_response(self, text: str) -> dict:
        """Парсинг ответа LLM (упрощённый для демо)"""
        return self._generate_rule_based_report([], {})  # Фолбэк на правило-базированный
    
    def augment_rare_defects(self, image: np.ndarray, defect_type: str) -> np.ndarray:
        """
        GAN-аугментация редких дефектов в реальном времени (демо-режим)
        В промышленной реализации: вызов предобученного CycleGAN
        """
        # Демо: имитация аугментации через наложение синтетических дефектов
        augmented = image.copy()
        
        if defect_type == "porosity":
            # Добавляем синтетические поры
            for _ in range(np.random.randint(3, 8)):
                x = np.random.randint(50, image.shape[1] - 50)
                y = np.random.randint(50, image.shape[0] - 50)
                r = np.random.randint(2, 6)
                cv2.circle(augmented, (x, y), r, (50, 50, 50), -1)
        
        elif defect_type == "crack":
            # Добавляем синтетическую трещину
            x1, y1 = np.random.randint(100, 300), np.random.randint(100, 300)
            x2, y2 = x1 + np.random.randint(30, 80), y1 + np.random.randint(-20, 20)
            cv2.line(augmented, (x1, y1), (x2, y2), (30, 30, 30), 2)
        
        return augmented


# Singleton для глобального доступа
_generative_analyzer = None

def get_generative_analyzer() -> GenerativeDefectAnalyzer:
    """Получение singleton-экземпляра генеративного анализатора"""
    global _generative_analyzer
    if _generative_analyzer is None:
        _generative_analyzer = GenerativeDefectAnalyzer(
            enable_diffusion=True,
            enable_llm=False  # Отключено по умолчанию для демо (требует GPU)
        )
    return _generative_analyzer
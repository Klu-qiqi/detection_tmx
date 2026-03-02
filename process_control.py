"""
process_control.py
Контроль последовательности технологических операций через:
1. Трекинг объектов между кадрами (ByteTrack)
2. Конечный автомат валидации технологической цепочки
"""

import cv2
import numpy as np
from collections import deque, defaultdict
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
import time


class ProductionStage(Enum):
    """Этапы технологического процесса"""
    RAW_MATERIAL = "Входной контроль"
    WELDING = "Сварка"
    WELD_INSPECTION = "Контроль сварки"
    PAINTING = "Покраска"
    FINAL_ASSEMBLY = "Финальная сборка"
    QUALITY_CONTROL = "Финальный контроль качества"


@dataclass
class ProductionEvent:
    """Событие в технологическом процессе"""
    timestamp: float
    stage: ProductionStage
    confidence: float
    metadata: Dict


class ProcessValidator:
    """
    Конечный автомат для валидации технологической цепочки
    Допустимая последовательность: RAW_MATERIAL → WELDING → WELD_INSPECTION → PAINTING → FINAL_ASSEMBLY → QUALITY_CONTROL
    """
    
    def __init__(self):
        self.valid_sequence = [
            ProductionStage.RAW_MATERIAL,
            ProductionStage.WELDING,
            ProductionStage.WELD_INSPECTION,
            ProductionStage.PAINTING,
            ProductionStage.FINAL_ASSEMBLY,
            ProductionStage.QUALITY_CONTROL
        ]
        self.current_stage_index = 0
        self.event_history = deque(maxlen=100)
        self.violations = []
        
    def validate_transition(self, detected_stage: ProductionStage) -> Dict:
        """
        Валидация перехода между этапами
        Возвращает: статус валидации и описание нарушений
        """
        expected_stage = self.valid_sequence[self.current_stage_index]
        
        # Проверка корректного перехода
        if detected_stage == expected_stage:
            # Корректный переход на следующий этап
            self.current_stage_index = min(self.current_stage_index + 1, len(self.valid_sequence) - 1)
            status = "valid"
            message = f"✅ Этап '{detected_stage.value}' пройден корректно"
            
        elif self.valid_sequence.index(detected_stage) < self.current_stage_index:
            # Возврат к предыдущему этапу — нарушение
            status = "violation"
            message = f"⚠️ Нарушение: возврат к этапу '{detected_stage.value}' после '{expected_stage.value}'"
            self.violations.append({
                "type": "backward_transition",
                "from_stage": expected_stage.value,
                "to_stage": detected_stage.value,
                "timestamp": time.time()
            })
            
        elif self.valid_sequence.index(detected_stage) > self.current_stage_index + 1:
            # Пропуск этапа — критическое нарушение
            status = "critical_violation"
            skipped = self.valid_sequence[self.current_stage_index:self.valid_sequence.index(detected_stage)]
            message = f"❌ КРИТИЧЕСКОЕ НАРУШЕНИЕ: пропущены этапы {[s.value for s in skipped]}"
            self.violations.append({
                "type": "skipped_stages",
                "skipped_stages": [s.value for s in skipped],
                "timestamp": time.time()
            })
            self.current_stage_index = self.valid_sequence.index(detected_stage)
            
        else:
            # Неизвестный этап
            status = "unknown"
            message = f"❓ Неизвестный этап: {detected_stage.value}"
        
        # Сохранение события
        event = ProductionEvent(
            timestamp=time.time(),
            stage=detected_stage,
            confidence=0.95,
            metadata={"validation_status": status}
        )
        self.event_history.append(event)
        
        return {
            "status": status,
            "message": message,
            "current_stage": expected_stage.value,
            "next_expected_stage": self.valid_sequence[min(self.current_stage_index, len(self.valid_sequence)-1)].value,
            "violations_count": len(self.violations)
        }
    
    def get_process_report(self) -> Dict:
        """Генерация отчёта по технологическому процессу"""
        return {
            "current_stage": self.valid_sequence[self.current_stage_index].value,
            "progress_percent": int((self.current_stage_index + 1) / len(self.valid_sequence) * 100),
            "total_violations": len(self.violations),
            "critical_violations": sum(1 for v in self.violations if v["type"] == "skipped_stages"),
            "event_history": [
                {
                    "timestamp": time.strftime("%H:%M:%S", time.localtime(e.timestamp)),
                    "stage": e.stage.value,
                    "status": e.metadata["validation_status"]
                }
                for e in list(self.event_history)[-10:]  # Последние 10 событий
            ],
            "is_process_valid": len([v for v in self.violations if v["type"] == "skipped_stages"]) == 0
        }


class ObjectTracker:
    """Трекинг объектов между кадрами для анализа последовательности операций"""
    
    def __init__(self):
        # В промышленной реализации: интеграция с ByteTrack или BoT-SORT
        self.track_history = defaultdict(list)  # track_id -> список боксов
        self.next_track_id = 0
        self.iou_threshold = 0.3
        
    def track_objects(self, current_detections: List[Dict], frame_id: int) -> List[Dict]:
        """
        Трекинг объектов между кадрами (упрощённая реализация для демо)
        Возвращает детекции с добавленным track_id
        """
        # Простая реализация: сопоставление по IoU с предыдущим кадром
        if not hasattr(self, 'prev_detections'):
            self.prev_detections = []
            self.track_ids = {}
        
        tracked_detections = []
        
        # Сопоставление текущих детекций с предыдущими
        for det in current_detections:
            best_match = None
            best_iou = self.iou_threshold
            
            for prev_id, prev_det in enumerate(self.prev_detections):
                iou = self._calculate_iou(det['bbox'], prev_det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = prev_id
            
            if best_match is not None:
                # Найдено совпадение — сохраняем тот же track_id
                track_id = self.track_ids.get(best_match, self.next_track_id)
                if best_match not in self.track_ids:
                    self.track_ids[best_match] = self.next_track_id
                    self.next_track_id += 1
            else:
                # Новая детекция — новый track_id
                track_id = self.next_track_id
                self.next_track_id += 1
            
            tracked_detections.append({
                **det,
                'track_id': track_id,
                'frame_id': frame_id
            })
        
        self.prev_detections = current_detections
        return tracked_detections
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Расчёт Intersection over Union между двумя боксами"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def detect_process_sequence(self, tracked_detections: List[Dict]) -> Dict:
        """
        Анализ последовательности операций на основе трекинга
        Пример: обнаружение, что покраска выполнена ДО контроля сварки
        """
        # Группировка по трекам
        tracks = defaultdict(list)
        for det in tracked_detections:
            tracks[det['track_id']].append(det)
        
        # Анализ последовательности для каждого трека
        sequence_violations = []
        
        for track_id, detections in tracks.items():
            # Сортировка по времени (кадрам)
            detections.sort(key=lambda x: x['frame_id'])
            
            # Извлечение последовательности операций
            operation_sequence = [self._map_class_to_stage(d['class']) for d in detections]
            
            # Проверка на нарушение порядка (упрощённо)
            if 'Покраска' in operation_sequence and 'Сварные швы' in operation_sequence:
                weld_idx = operation_sequence.index('Сварные швы')
                paint_idx = operation_sequence.index('Покраска')
                if paint_idx < weld_idx:
                    sequence_violations.append({
                        "track_id": track_id,
                        "violation": "Покраска выполнена до сварки",
                        "frames": [d['frame_id'] for d in detections]
                    })
        
        return {
            "total_tracks": len(tracks),
            "sequence_violations": sequence_violations,
            "is_sequence_valid": len(sequence_violations) == 0
        }
    
    def _map_class_to_stage(self, class_name: str) -> str:
        """Маппинг класса дефекта к этапу производства"""
        if any(kw in class_name.lower() for kw in ['weld', 'шов', 'porosity', 'crack']):
            return 'Сварные швы'
        elif any(kw in class_name.lower() for kw in ['paint', 'scratch', 'покраска']):
            return 'Покраска'
        elif 'missing' in class_name.lower() or 'комплектация' in class_name.lower():
            return 'Комплектация'
        else:
            return 'Неизвестно'


# Singleton для глобального доступа
_process_validator = None
_object_tracker = None

def get_process_validator() -> ProcessValidator:
    global _process_validator
    if _process_validator is None:
        _process_validator = ProcessValidator()
    return _process_validator

def get_object_tracker() -> ObjectTracker:
    global _object_tracker
    if _object_tracker is None:
        _object_tracker = ObjectTracker()
    return _object_tracker
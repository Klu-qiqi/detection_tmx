"""
process_control.py
Контроль технологических процессов через трекинг объектов и конечный автомат
"""

from enum import Enum
from collections import deque
from dataclasses import dataclass
from typing import Dict, List
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
    
    Допустимая последовательность:
    Входной контроль → Сварка → Контроль сварки → Покраска → Финальная сборка → Финальный контроль
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
        """Валидация перехода между этапами"""
        expected_stage = self.valid_sequence[self.current_stage_index]
        
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
            skipped_indices = range(self.current_stage_index, self.valid_sequence.index(detected_stage))
            skipped = [self.valid_sequence[i] for i in skipped_indices]
            message = f"❌ КРИТИЧЕСКОЕ НАРУШЕНИЕ: пропущены этапы {[s.value for s in skipped]}"
            self.violations.append({
                "type": "skipped_stages",
                "skipped_stages": [s.value for s in skipped],
                "timestamp": time.time()
            })
            self.current_stage_index = self.valid_sequence.index(detected_stage)
            
        else:
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
                for e in list(self.event_history)[-10:]
            ],
            "is_process_valid": len([v for v in self.violations if v["type"] == "skipped_stages"]) == 0
        }
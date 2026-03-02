"""
generative_ai.py
Генерация объяснимых отчётов с причинно-следственным анализом
Имитация работы LLM через правило-базированный подход с генеративной логикой
"""

import numpy as np
from datetime import datetime

class GenerativeReportGenerator:
    """
    Генератор объяснимых отчётов с причинно-следственным анализом
    
    Принцип работы:
    1. Анализ типа и критичности дефектов
    2. Генерация гипотез о корневых причинах
    3. Формирование рекомендаций с учётом технологического контекста
    4. Оценка уверенности анализа
    """
    
    def __init__(self):
        # База знаний для генерации отчётов
        self.knowledge_base = {
            "weld": {
                "porosity": {
                    "causes": [
                        "Загрязнение поверхности перед сваркой",
                        "Неправильный расход защитного газа (низкий)",
                        "Высокая влажность в зоне сварки"
                    ],
                    "recommendations": [
                        "Проверить чистоту свариваемых кромок",
                        "Увеличить расход защитного газа на 15%",
                        "Проверить герметичность газовых магистралей"
                    ],
                    "process_violation": "Нарушение подготовки поверхности к сварке"
                },
                "crack": {
                    "causes": [
                        "Высокие остаточные напряжения",
                        "Отсутствие предварительного подогрева",
                        "Неподходящий режим термообработки"
                    ],
                    "recommendations": [
                        "Провести предварительный подогрев до 150°C",
                        "Контролировать скорость охлаждения после сварки",
                        "Проверить химический состав основного металла"
                    ],
                    "process_violation": "Нарушение режима термической обработки"
                },
                "spatter": {
                    "causes": [
                        "Избыточный сварочный ток",
                        "Неправильный угол наклона горелки",
                        "Некачественные сварочные материалы"
                    ],
                    "recommendations": [
                        "Снизить сварочный ток на 10-15%",
                        "Отрегулировать угол наклона горелки до 80°",
                        "Проверить сертификаты сварочных материалов"
                    ],
                    "process_violation": "Несоответствие параметров сварки технологической карте"
                }
            },
            "paint": {
                "anomaly": {
                    "causes": [
                        "Недостаточная подготовка поверхности",
                        "Нарушение режима сушки ЛКП",
                        "Загрязнение краскораспылителя"
                    ],
                    "recommendations": [
                        "Проверить качество абразивной обработки",
                        "Контролировать температуру и влажность в камере сушки",
                        "Очистить систему подачи краски"
                    ],
                    "process_violation": "Нарушение технологии нанесения ЛКП"
                }
            },
            "assembly": {
                "anomaly": {
                    "causes": [
                        "Некомплект поставки",
                        "Ошибка оператора при сборке",
                        "Нарушение последовательности операций"
                    ],
                    "recommendations": [
                        "Провести инвентаризацию комплектующих",
                        "Усилить визуальный контроль на этапе сборки",
                        "Проверить корректность технологической документации"
                    ],
                    "process_violation": "Нарушение последовательности сборочных операций"
                }
            },
            "damage": {
                "anomaly": {
                    "causes": [
                        "Механическое воздействие при транспортировке",
                        "Нарушение режима хранения",
                        "Дефект оборудования цеха"
                    ],
                    "recommendations": [
                        "Проверить состояние транспортных приспособлений",
                        "Контролировать параметры складских помещений",
                        "Провести диагностику оборудования цеха"
                    ],
                    "process_violation": "Нарушение правил транспортировки и хранения"
                }
            }
        }
    
    def generate_report(self, defects, anomaly_score, category, context=None):
        """
        Генерация объяснимого отчёта с причинно-следственным анализом
        
        Args:
            defects: Список обнаруженных дефектов
            anomaly_score: Степень аномалии (для не-сварочных категорий)
            category: Категория анализа (weld, paint, assembly, damage)
            context: Контекст производства (линия, смена, оператор)
            
        Returns:
            report: Словарь с генеративным отчётом
        """
        # Определение типа дефекта для поиска в базе знаний
        defect_type = "anomaly"
        if category == "weld" and defects:
            # Для сварки определяем тип дефекта по классу
            cls_name = defects[0]["class"].lower()
            if "porosity" in cls_name or "пор" in cls_name:
                defect_type = "porosity"
            elif "crack" in cls_name or "трещ" in cls_name:
                defect_type = "crack"
            elif "spatter" in cls_name or "брызг" in cls_name:
                defect_type = "spatter"
        
        # Получение знаний из базы
        category_kb = self.knowledge_base.get(category, {})
        defect_kb = category_kb.get(defect_type, {
            "causes": ["Неизвестная причина"],
            "recommendations": ["Требуется дополнительный анализ в лаборатории"],
            "process_violation": "Нарушение не определено"
        })
        
        # Генерация корневой причины
        root_cause = defect_kb["causes"][0] if defect_kb["causes"] else "Причина не определена"
        
        # Генерация рекомендаций
        recommendations = defect_kb["recommendations"][:3]
        
        # Добавление контекстно-зависимых рекомендаций
        if context:
            if "смена" in context.get("shift", "").lower() and "ночн" in context.get("shift", "").lower():
                recommendations.append("Усилить контроль качества в ночной смене из-за повышенной утомляемости персонала")
            
            if anomaly_score and anomaly_score > 0.7:
                recommendations.insert(0, "НЕОБХОДИМО: Остановить производственную линию для выявления корневой причины")
        
        # Расчёт уверенности отчёта
        confidence = 0.95
        if not defects and anomaly_score is not None:
            confidence = max(0.7, 1.0 - anomaly_score)  # Для аномалий
        elif defects:
            avg_conf = np.mean([d["confidence"] for d in defects])
            confidence = 0.7 + (avg_conf * 0.25)
        
        # Формирование отчёта
        report = {
            "root_cause": root_cause,
            "process_violation": defect_kb["process_violation"],
            "recommendations": recommendations,
            "confidence": min(confidence, 0.98),
            "analysis_timestamp": datetime.now().isoformat(),
            "technology": "Generative AI (причинно-следственный анализ)",
            "category": category,
            "defect_count": len(defects)
        }
        
        return report
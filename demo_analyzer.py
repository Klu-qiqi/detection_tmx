# demo_analyzer.py
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

class WeldQualityInspector:
    def __init__(self, model_path="best_weld_ai.pt"):
        self.model = YOLO(model_path)
        self.class_names = [
            'Bad Welding', 'Crack', 'Excess Reinforcement', 
            'Good Welding', 'Porosity', 'Spatters'
        ]
        self.critical_classes = [0, 1, 4]  # критические дефекты
    
    def analyze_image(self, image_path, conf_threshold=0.35):
        """Полный анализ качества сварного шва с отчётом"""
        # Загрузка изображения
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Не найдено изображение: {image_path}")
        
        # Инференс
        results = self.model(img, conf=conf_threshold, iou=0.65, imgsz=640)[0]
        
        # Анализ дефектов
        defects = []
        good_weld_area = 0
        total_weld_area = 0
        
        for i, (box, cls, conf) in enumerate(zip(results.boxes.xyxy, 
                                                 results.boxes.cls, 
                                                 results.boxes.conf)):
            cls_id = int(cls.item())
            bbox = box.cpu().numpy()
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Сегментация для точного расчёта площади (если доступна)
            mask_area = 0
            if results.masks is not None:
                mask = results.masks.data[i].cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                mask_area = np.sum(mask > 0.5)
            
            defect = {
                "id": i + 1,
                "class_id": cls_id,
                "class_name": self.class_names[cls_id],
                "confidence": float(conf),
                "bbox": bbox.tolist(),
                "area_pixels": int(mask_area) if mask_area > 0 else int(area),
                "is_critical": cls_id in self.critical_classes
            }
            
            # Статистика по качеству
            if cls_id == 3:  # Good Welding
                good_weld_area += defect["area_pixels"]
            total_weld_area += defect["area_pixels"]
            
            defects.append(defect)
        
        # Расчёт итоговых метрик
        defect_ratio = 1.0 - (good_weld_area / total_weld_area) if total_weld_area > 0 else 1.0
        critical_count = sum(1 for d in defects if d["is_critical"])
        
        # Итоговая оценка качества (0-100)
        quality_score = 100.0
        quality_score -= defect_ratio * 50  # штраф за долю дефектов
        quality_score -= critical_count * 15  # штраф за критические дефекты
        quality_score = max(0, min(100, quality_score))
        
        # Формирование отчёта
        report = {
            "image": str(image_path),
            "total_defects": len(defects),
            "critical_defects": critical_count,
            "defect_ratio": defect_ratio,
            "quality_score": quality_score,
            "defects": defects,
            "annotated_image": results.plot()
        }
        
        return report
    
    def generate_quality_report(self, report, save_path="quality_report.txt"):
        """Генерация человекочитаемого отчёта для инженера"""
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write("ОТЧЁТ ПО КОНТРОЛЮ КАЧЕСТВА СВАРНОГО ШВА\n")
            f.write("="*60 + "\n\n")
            f.write(f"Изображение: {report['image']}\n")
            f.write(f"Дата анализа: {Path(report['image']).stat().st_mtime}\n\n")
            f.write(f"ИТОГОВАЯ ОЦЕНКА КАЧЕСТВА: {report['quality_score']:.1f}/100\n")
            f.write(f"Статус: {'✅ ГОДЕН' if report['quality_score'] >= 80 else '⚠️ ТРЕБУЕТ ПЕРЕДЕЛКИ' if report['quality_score'] >= 60 else '❌ БРАК'}\n\n")
            f.write(f"Обнаружено дефектов: {report['total_defects']}\n")
            f.write(f"Критических дефектов: {report['critical_defects']}\n")
            f.write(f"Доля дефектной зоны: {report['defect_ratio']:.1%}\n\n")
            
            if report['defects']:
                f.write("ДЕТАЛИЗАЦИЯ ДЕФЕКТОВ:\n")
                f.write("-"*60 + "\n")
                for d in report['defects']:
                    severity = "🔴 КРИТИЧЕСКИЙ" if d["is_critical"] else "🟡 СРЕДНИЙ"
                    f.write(f"ID {d['id']:2d} | {severity:15s} | {d['class_name']:25s} | "
                           f"уверенность: {d['confidence']:5.1%} | площадь: {d['area_pixels']} px\n")
            else:
                f.write("✅ Дефекты не обнаружены. Шов соответствует требованиям.\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("Рекомендации:\n")
            if report['quality_score'] >= 80:
                f.write("  • Продукция допущена к следующей операции\n")
            elif report['quality_score'] >= 60:
                f.write("  • Требуется ручная проверка инженером ОТК\n")
                f.write("  • Возможна локальная доработка дефектных зон\n")
            else:
                f.write("  • Брак. Требуется полная переделка сварного соединения\n")
                f.write("  • Проверить настройки сварочного оборудования\n")
            f.write("="*60 + "\n")
        
        print(f"📄 Отчёт сформирован: {save_path}")
        return save_path
    
    def visualize_results(self, report, output_dir="results"):
        """Визуализация результатов для демо"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1. Аннотированное изображение
        cv2.imwrite(f"{output_dir}/annotated.jpg", report["annotated_image"])
        
        # 2. График распределения дефектов
        if report["defects"]:
            classes = [d["class_name"] for d in report["defects"]]
            confs = [d["confidence"] for d in report["defects"]]
            
            plt.figure(figsize=(10, 5))
            colors = ["red" if self.class_names.index(c) in self.critical_classes else "orange" 
                     for c in classes]
            bars = plt.barh(classes, confs, color=colors)
            plt.xlabel("Уверенность модели")
            plt.title(f"Дефекты сварного шва (Итоговая оценка: {report['quality_score']:.0f}/100)")
            plt.xlim(0, 1)
            
            # Добавляем значения на бары
            for bar, conf in zip(bars, confs):
                plt.text(conf + 0.02, bar.get_y() + bar.get_height()/2, 
                        f'{conf:.1%}', va='center')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/defects_chart.png", dpi=150)
            plt.close()
        
        print(f"🖼️  Визуализация сохранена в: {output_dir}/")

# Демонстрация работы
if __name__ == "__main__":
    inspector = WeldQualityInspector("best_weld_ai.pt")
    
    # Анализ тестового изображения (возьмём первое из test/images)
    test_images = list(Path("test/images").glob("*.*"))
    if not test_images:
        print("⚠️  Папка test/images пуста. Используем синтетическое изображение...")
        # Создаём тестовое изображение
        img = np.random.randint(100, 130, (640, 640, 3), dtype=np.uint8)
        cv2.line(img, (100, 320), (540, 320), (200, 200, 200), 12)  # шов
        cv2.line(img, (250, 310), (350, 330), (30, 30, 30), 3)      # трещина
        cv2.imwrite("test_sample.jpg", img)
        test_image = "test_sample.jpg"
    else:
        test_image = test_images[0]
    
    print(f"\n🔍 Анализ изображения: {test_image}")
    report = inspector.analyze_image(test_image)
    
    # Вывод результатов в консоль
    print(f"\nРЕЗУЛЬТАТЫ АНАЛИЗА:")
    print(f"  Итоговая оценка: {report['quality_score']:.1f}/100")
    print(f"  Дефектов обнаружено: {report['total_defects']}")
    print(f"  Критических: {report['critical_defects']}")
    
    if report['defects']:
        print("\nДЕТЕКТИРОВАННЫЕ ДЕФЕКТЫ:")
        for d in report['defects']:
            mark = "🔴" if d["is_critical"] else "🟡"
            print(f"  {mark} {d['class_name']:25s} | {d['confidence']:5.1%} | {d['area_pixels']} px")
    
    # Генерация отчётов
    inspector.generate_quality_report(report, "weld_quality_report.txt")
    inspector.visualize_results(report, "demo_results")
    
    print("\n✅ Демо-анализ завершён!")
    print("   - Аннотированное изображение: demo_results/annotated.jpg")
    print("   - График дефектов: demo_results/defects_chart.png")
    print("   - Текстовый отчёт: weld_quality_report.txt")
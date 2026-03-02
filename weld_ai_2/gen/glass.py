# steel_paint_generator.py
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image
import argparse
import os
from pathlib import Path

class SteelPaintGenerator:
    """
    Генератор изображений идеально окрашенных стальных элементов.
    Не требует обучения — использует предобученные модели.
    """
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Загрузка моделей на устройство: {self.device}")
        
        # ControlNet для сохранения геометрии металлического элемента
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # SDXL с улучшенной детализацией металлов/поверхностей
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            variant="fp16" if self.device == "cuda" else None,
            use_safetensors=True
        ).to(self.device)
        
        # Оптимизации для скорости
        if self.device == "cuda":
            self.pipe.enable_xformers_memory_efficient_attention()
            self.pipe.enable_model_cpu_offload()
        
        # Промпты для идеальной покраски (настроены экспериментально)
        self.base_prompt = (
            "perfectly painted steel surface, uniform glossy coating, "
            "industrial quality, no defects, no scratches, no drips, "
            "smooth finish, metallic sheen, professional automotive paint, "
            "studio lighting, photorealistic, 8k"
        )
        self.negative_prompt = (
            "defects, scratches, dust, fingerprints, uneven paint, drips, "
            "bubbles, orange peel texture, rust, corrosion, dirt, smudges, "
            "poor lighting, blurry, low quality, jpeg artifacts"
        )
    
    def _preprocess_canny(self, image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
        """Преобразование изображения заготовки в Canny edges для контроля геометрии"""
        image = np.array(image)
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)
    
    def generate_from_sketch(self, 
                           sketch_path: str = None,
                           num_images: int = 4,
                           guidance_scale: float = 7.5,
                           num_inference_steps: int = 30,
                           seed: int = None) -> list:
        """
        Генерация изображений чистой покраски.
        
        Args:
            sketch_path: путь к изображению стального элемента (заготовка без покраски).
                         Если None — генерация случайных форм.
            num_images: сколько изображений сгенерировать
            guidance_scale: сила следования промпту (7-9 оптимально)
            seed: для воспроизводимости
        
        Returns:
            Список PIL.Image с изображениями идеально окрашенного металла
        """
        # Если нет скетча — создаём случайную "заготовку" металла
        if sketch_path is None:
            print("Создание случайной геометрии стального элемента...")
            sketch = Image.new("RGB", (1024, 1024), (120, 120, 120))  # серый металл
            draw = ImageDraw.Draw(sketch)
            # Рисуем простую геометрию (прямоугольник как заготовка)
            draw.rectangle((200, 200, 824, 824), fill=(80, 80, 80), outline=(50, 50, 50), width=8)
        else:
            sketch = load_image(sketch_path).convert("RGB").resize((1024, 1024))
        
        # Преобразуем в Canny edges для ControlNet
        canny_image = self._preprocess_canny(sketch)
        
        # Генерация
        generator = torch.manual_seed(seed) if seed is not None else None
        
        print(f"Генерация {num_images} изображений идеальной покраски...")
        images = self.pipe(
            prompt=[self.base_prompt] * num_images,
            negative_prompt=[self.negative_prompt] * num_images,
            image=[canny_image] * num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            controlnet_conditioning_scale=0.8,
        ).images
        
        print("✓ Генерация завершена!")
        return images
    
    def save_images(self, images: list, output_dir: str = "generated_paint"):
        """Сохранение изображений в папку"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, img in enumerate(images):
            path = Path(output_dir) / f"painted_steel_{i:03d}.png"
            img.save(path)
            print(f"Сохранено: {path}")
        
        # Сохраняем также скетч для сравнения
        if hasattr(self, 'last_sketch'):
            self.last_sketch.save(Path(output_dir) / "input_sketch.png")

# ======================
# CLI ИНТЕРФЕЙС (запуск одной командой)
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генератор изображений идеальной покраски стальных элементов")
    parser.add_argument("--sketch", type=str, default=None, 
                       help="Путь к изображению заготовки (стальной элемент без покраски)")
    parser.add_argument("--num_images", type=int, default=4, 
                       help="Количество генерируемых изображений")
    parser.add_argument("--output", type=str, default="generated_paint", 
                       help="Папка для сохранения результатов")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Seed для воспроизводимости")
    parser.add_argument("--cpu", action="store_true", 
                       help="Принудительно использовать CPU (медленнее)")
    
    args = parser.parse_args()
    
    # Инициализация генератора
    device = "cpu" if args.cpu else None
    generator = SteelPaintGenerator(device=device)
    
    # Генерация
    images = generator.generate_from_sketch(
        sketch_path=args.sketch,
        num_images=args.num_images,
        seed=args.seed
    )
    
    # Сохранение
    generator.save_images(images, args.output)
    
    print(f"\n🎉 Готово! Изображения сохранены в: {args.output}/")
    print("Пример использования в коде:")
    print("  from steel_paint_generator import SteelPaintGenerator")
    print("  gen = SteelPaintGenerator()")
    print("  images = gen.generate_from_sketch('my_steel_part.jpg')")
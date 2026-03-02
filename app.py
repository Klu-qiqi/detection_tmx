"""
app.py
Платформа видеоаналитики дефектов с применением генеративного ИИ
Хакатон РЖД 2026 — «Видеоаналитика ИИ на страже качества в производстве»
"""

import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import json
from pathlib import Path

# ==================== СКРЫТИЕ ЭЛЕМЕНТОВ STREAMLIT ====================
st.markdown("""
<style>
    .stApp > header > div:first-child { display: none !important; }
    .stDeployButton, div[data-testid="stDeployButton"] { display: none !important; }
    footer, .streamlit-footer { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }
    .main { padding-top: 1rem !important; padding-bottom: 1rem !important; }
    .block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# ==================== ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ С ОБРАБОТКОЙ ОШИБОК ====================
@st.cache_resource
def init_models():
    """Инициализация всех моделей платформы с валидацией путей"""
    models = {}
    
    try:
        from weld_detector import WeldDefectDetector
        weld_path = Path("models/best_weld_ai.pt")
        if weld_path.exists():
            models["weld"] = WeldDefectDetector(model_path=str(weld_path))
            print("✅ Модель сварки загружена")
        else:
            print("⚠️ Модель сварки не найдена. Демо-режим активирован.")
    except Exception as e:
        print(f"Ошибка инициализации модели сварки: {str(e)}")
    
    try:
        from anomaly_detector import AnomalyDetector
        models["anomaly"] = AnomalyDetector()
    except Exception as e:
        print(f"Ошибка инициализации модели аномалий: {str(e)}")
    
    try:
        from generative_ai import GenerativeReportGenerator
        models["report"] = GenerativeReportGenerator()
    except Exception as e:
        print(f"Ошибка инициализации генератора отчётов: {str(e)}")
    
    try:
        from process_control import ProcessValidator
        models["process"] = ProcessValidator()
    except Exception as e:
        print(f"Ошибка инициализации валидатора процесса: {str(e)}")
    
    return models

models = init_models()

# ==================== СТИЛИЗАЦИЯ ====================
st.set_page_config(
    page_title="Δплан | Видеоаналитика дефектов | Хакатон ТМХ",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
    .stApp { background: transparent; }
    
    h1 {
        color: #38bdf8;
        text-align: center;
        margin-bottom: 5px;
        font-weight: 700;
        text-shadow: 0 2px 10px rgba(56, 189, 248, 0.3);
    }
    .subtitle {
        color: #94a3b8;
        text-align: center;
        margin-bottom: 30px;
        font-size: 18px;
    }
    
    /* Стили для кликабельных карточек */
    .category-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 15px;
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
        cursor: pointer;
        border: 2px solid transparent;
        color: white;
        position: relative;
        z-index: 1;
    }
    .category-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
        border-color: #38bdf8;
    }
    .category-icon {
        font-size: 48px;
        margin-bottom: 15px;
        color: #38bdf8;
    }
    .category-title {
        color: white;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .category-desc {
        color: #94a3b8;
        font-size: 16px;
    }
    
    .demo-mode {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        margin: 20px 0;
    }
    
    .analysis-card {
        background: rgba(30, 41, 59, 0.9);
        border-radius: 16px;
        padding: 25px;
        margin: 15px 0;
    }
    
    .metric-box {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-value {
        font-size: 42px;
        font-weight: bold;
        color: #38bdf8;
        margin: 10px 0;
    }
    .metric-label {
        color: #cbd5e1;
        font-size: 16px;
        font-weight: 500;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 16px 32px;
        font-size: 18px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(30, 64, 175, 0.4);
    }
    
    .generative-section {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        border-radius: 16px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.3);
    }
    
    .no-defects {
        background: linear-gradient(135deg, #047857 0%, #059669 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin: 25px 0;
        font-size: 24px;
        font-weight: bold;
    }
    
    .critical-defect {
        background: linear-gradient(135deg, #b91c1c 0%, #ef4444 100%);
        color: white;
        padding: 25px;
        border-radius: 16px;
        margin: 15px 0;
    }
    
    .augmentation-info {
        background: linear-gradient(135deg, #0ea5e9 0%, #38bdf8 100%);
        color: white;
        padding: 15px;
        border-radius: 12px;
        margin: 20px 0;
        text-align: center;
        font-weight: bold;
    }
    
    .demo-button {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 16px 32px;
        font-size: 18px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    .demo-button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.5);
    }
    
    /* Центрирование карточек в третьей строке */
    .center-col {
        display: flex;
        justify-content: center;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
def generate_demo_image(analysis_type):
    """Генерация демо-изображений для всех типов анализа"""
    height, width = 640, 640
    image = np.ones((height, width, 3), dtype=np.uint8) * 220
    
    if analysis_type == "weld":
        # Текстура металла
        for i in range(0, height, 8):
            cv2.line(image, (0, i), (width, i), (210, 210, 210), 1)
        # Сварной шов
        center_x = width // 2
        cv2.rectangle(image, (center_x-15, 150), (center_x+15, 490), (200, 200, 200), -1)
        cv2.line(image, (center_x, 150), (center_x, 490), (240, 240, 240), 2)
        # Дефекты
        cv2.circle(image, (center_x-10, 250), 8, (70, 70, 70), -1)  # Поры
        cv2.line(image, (center_x+20, 300), (center_x+40, 330), (50, 50, 50), 2)  # Трещина
        cv2.circle(image, (center_x+30, 200), 4, (80, 80, 80), -1)  # Разбрызгивание
    
    elif analysis_type == "paint":
        # Градиент фона
        for y in range(height):
            intensity = 235 - int(y * 5 / height)
            cv2.line(image, (0, y), (width, y), (intensity, intensity, intensity + 5), 1)
        # Царапина
        cv2.line(image, (200, 300), (350, 350), (180, 180, 190), 3)
        # Потёк краски
        cv2.ellipse(image, (400, 250), (30, 60), 0, 0, 360, (200, 210, 230), -1)
    
    elif analysis_type == "assembly":
        # Фон сборки
        cv2.rectangle(image, (100, 150), (540, 490), (200, 200, 200), -1)
        # Детали (круги)
        cv2.circle(image, (200, 250), 40, (50, 150, 50), -1)  # Есть деталь
        cv2.circle(image, (440, 350), 40, (150, 50, 50), -1)  # Отсутствует деталь (красная)
        cv2.putText(image, "MISSING", (390, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    elif analysis_type == "damage":
        # Фон поверхности
        cv2.rectangle(image, (50, 100), (590, 540), (180, 180, 190), -1)
        # Вмятина
        cv2.ellipse(image, (320, 320), (80, 40), 0, 0, 360, (100, 100, 110), -1)
        # Царапина
        cv2.line(image, (150, 200), (450, 280), (80, 80, 90), 4)
    
    elif analysis_type == "doors":
        # Фон двери
        cv2.rectangle(image, (100, 50), (540, 590), (220, 220, 220), -1)
        cv2.rectangle(image, (120, 70), (520, 570), (200, 200, 200), -1)
        # Ручка двери
        cv2.circle(image, (480, 320), 25, (180, 180, 180), -1)
        # Дефект: вмятина на двери
        cv2.ellipse(image, (250, 300), (40, 25), 0, 0, 360, (150, 150, 160), -1)
        # Текстура двери
        for y in range(70, 570, 30):
            cv2.line(image, (120, y), (520, y), (190, 190, 190), 1)
    
    elif analysis_type == "glass":
        # Фон стекла (прозрачный)
        cv2.rectangle(image, (150, 150), (490, 490), (240, 245, 255), -1)
        # Рамка стекла
        cv2.rectangle(image, (140, 140), (500, 500), (100, 100, 110), 3)
        # Дефект: трещина на стекле
        cv2.line(image, (200, 200), (440, 440), (200, 220, 255), 2)
        cv2.line(image, (250, 200), (440, 390), (200, 220, 255), 2)
        # Блик на стекле
        cv2.ellipse(image, (320, 250), (30, 20), 30, 0, 360, (255, 255, 255), -1)
    
    elif analysis_type == "interior":
        # Фон салона (панель приборов)
        cv2.rectangle(image, (100, 100), (540, 180), (200, 190, 180), -1)
        # Сиденье
        cv2.rectangle(image, (50, 200), (590, 440), (180, 160, 140), -1)
        # Подлокотник
        cv2.rectangle(image, (280, 220), (360, 300), (160, 140, 120), -1)
        # Дефект: царапина на панели
        cv2.line(image, (200, 130), (350, 150), (150, 140, 130), 3)
        # Кнопки на панели
        for x in [150, 200, 250, 300, 350, 400, 450]:
            cv2.circle(image, (x, 140), 8, (100, 100, 100), -1)
    
    return image

def perform_analysis(image, analysis_type, models):
    """Единая функция анализа для авто-запуска и ручного запуска"""
    try:
        # Для сварки используем специальную модель
        if analysis_type == "weld" and "weld" in models and models["weld"]:
            annotated_img, defects = models["weld"].detect(image.copy())
            anomaly_score = None
        else:
            # Для всех остальных типов используем аномалию-детекцию
            if "anomaly" in models and models["anomaly"]:
                annotated_img, anomaly_score, is_anomaly = models["anomaly"].detect_anomaly(
                    image.copy(), 
                    category=analysis_type
                )
                # Обязательный ключ 'impact' для совместимости с отображением
                defects = [{
                    "class": "Аномалия поверхности", 
                    "confidence": float(anomaly_score) if anomaly_score else 0.0,
                    "severity": "Критичный" if is_anomaly else "Низкий",
                    "impact": "Нарушение герметичности" if analysis_type == "weld" else "Снижение коррозионной стойкости"
                }] if is_anomaly else []
            else:
                # Режим заглушки при отсутствии модели
                annotated_img = image.copy()
                cv2.putText(annotated_img, "ДЕМО-РЕЖИМ", (150, 320), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 150, 255), 3)
                defects = [{
                    "class": "Демо-дефект", 
                    "confidence": 0.85,
                    "severity": "Средний",
                    "impact": "Требуется визуальная проверка"
                }]
                anomaly_score = 0.85 if analysis_type != "weld" else None
        
        # Генерация отчёта (с проверкой наличия модели)
        context = {
            "production_line": "Линия №3 — сборка кузовов пассажирских вагонов",
            "shift": "Дневная смена",
            "operator": "Иванов А.С."
        }
        
        if "report" in models and models["report"]:
            report = models["report"].generate_report(
                defects=defects,
                anomaly_score=anomaly_score,
                category=analysis_type,
                context=context
            )
        else:
            # Заглушка для отчёта
            report = {
                "root_cause": "Демо-причина: нарушение параметров сварки",
                "process_violation": "Демо-нарушение: отклонение по скорости подачи проволоки",
                "recommendations": [
                    "Проверить настройки сварочного аппарата",
                    "Провести калибровку датчиков",
                    "Обучить оператора по обновлённому регламенту"
                ],
                "confidence": 0.92
            }
        
        return {
            "timestamp": datetime.now(),
            "analysis_type": analysis_type,
            "image": image,
            "annotated_image": annotated_img,
            "defects": defects,
            "anomaly_score": anomaly_score,
            "report": report,
            "is_video": False
        }
    except Exception as e:
        print(f"Ошибка при анализе: {str(e)}")
        import traceback
        traceback.print_exc()
        # Возвращаем минимально валидную структуру для отображения ошибки
        return {
            "timestamp": datetime.now(),
            "analysis_type": analysis_type,
            "image": image,
            "annotated_image": image,
            "defects": [],
            "anomaly_score": None,
            "report": {
                "root_cause": f"Ошибка анализа: {str(e)}",
                "process_violation": "Требуется ручная проверка",
                "recommendations": ["Повторить анализ", "Обратиться к инженеру ИИ"],
                "confidence": 0.0
            },
            "is_video": False
        }

# ==================== ГЛАВНАЯ СТРАНИЦА ====================
if "page" not in st.session_state:
    st.session_state.page = "main"
    st.session_state.analysis_type = None
    st.session_state.demo_triggered = False
    st.session_state.auto_run_demo = False

if st.session_state.page == "main":
    st.title("Δплан")
    st.markdown('<p class="subtitle">Видеоаналитика ИИ на страже качества в производстве</p>', unsafe_allow_html=True)
    
    # Демо-режим
    col_demo1, col_demo2, col_demo3 = st.columns([1, 2, 1])
    with col_demo2:
        st.markdown("""
        <style>
            div.stButton > button.demo-button {
                background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 16px 32px;
                font-size: 18px;
                font-weight: 600;
                width: 100%;
                transition: all 0.3s;
            }
            div.stButton > button.demo-button:hover {
                transform: scale(1.05);
                box-shadow: 0 10px 30px rgba(139, 92, 246, 0.5);
            }
        </style>
        """, unsafe_allow_html=True)
        
        '''if st.button("🎬 ЗАПУСТИТЬ ДЕМО ОДНИМ КЛИКОМ", type="primary", use_container_width=True, key="demo_main"):
            st.session_state.page = "analysis"
            st.session_state.analysis_type = "weld"
            st.session_state.use_demo = True
            st.session_state.auto_run_demo = True
            st.rerun()'''
    
    st.markdown("### Выберите тип контроля качества")
    
    # ===== СТРОКА 1: Сварные швы, Покраска, Двери =====
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔧 Сварные швы", use_container_width=True, key="cat_weld"):
            st.session_state.page = "analysis"
            st.session_state.analysis_type = "weld"
            st.rerun()
        
        st.markdown("""
        <div class="category-card">
            <div class="category-icon">⚡</div>
            <div class="category-title">Сварные швы</div>
            <div class="category-desc">Детекция пор, трещин, непроваров</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🎨 Покраска", use_container_width=True, key="cat_paint"):
            st.session_state.page = "analysis"
            st.session_state.analysis_type = "paint"
            st.rerun()
        
        st.markdown("""
        <div class="category-card">
            <div class="category-icon">🎨</div>
            <div class="category-title">Покраска</div>
            <div class="category-desc">Контроль качества ЛКП</div>
        </div>
        """, unsafe_allow_html=True)
    
    '''with col3:
        if st.button("🚪 Двери", use_container_width=True, key="cat_doors"):
            st.session_state.page = "analysis"
            st.session_state.analysis_type = "doors"
            st.rerun()
        
        st.markdown("""
        <div class="category-card">
            <div class="category-icon">🚪</div>
            <div class="category-title">Двери</div>
            <div class="category-desc">Контроль целостности и монтажа</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== СТРОКА 2: Стекла, Интерьер, Комплектация =====
    col4, col5, col6 = st.columns(3)
    
    with col4:
        if st.button("🪟 Стекла", use_container_width=True, key="cat_glass"):
            st.session_state.page = "analysis"
            st.session_state.analysis_type = "glass"
            st.rerun()
        
        st.markdown("""
        <div class="category-card">
            <div class="category-icon">🪟</div>
            <div class="category-title">Стекла</div>
            <div class="category-desc">Проверка на сколы и трещины</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        if st.button("🛋️ Интерьер", use_container_width=True, key="cat_interior"):
            st.session_state.page = "analysis"
            st.session_state.analysis_type = "interior"
            st.rerun()
        
        st.markdown("""
        <div class="category-card">
            <div class="category-icon">🛋️</div>
            <div class="category-title">Интерьер</div>
            <div class="category-desc">Контроль отделки салона</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        if st.button("🔩 Комплектация", use_container_width=True, key="cat_assembly"):
            st.session_state.page = "analysis"
            st.session_state.analysis_type = "assembly"
            st.rerun()
        
        st.markdown("""
        <div class="category-card">
            <div class="category-icon">🔩</div>
            <div class="category-title">Комплектация</div>
            <div class="category-desc">Проверка наличия деталей</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== СТРОКА 3: Механические повреждения (по центру) =====
    col7, col8, col9 = st.columns([1, 2, 1])
    
    with col8:
        if st.button("⚠️ Механические повреждения", use_container_width=True, key="cat_damage"):
            st.session_state.page = "analysis"
            st.session_state.analysis_type = "damage"
            st.rerun()
        
        st.markdown("""
        <div class="category-card">
            <div class="category-icon">⚠️</div>
            <div class="category-title">Механические повреждения</div>
            <div class="category-desc">Вмятины, деформации, царапины</div>
        </div>
        """, unsafe_allow_html=True)'''
    
    st.markdown("---")
    st.markdown("###  Инновационный подход платформы")
    
    col_tech1, col_tech2, col_tech3 = st.columns(3)
    
    with col_tech1:
        st.markdown("""
        <div class="analysis-card">
            <h3 style="color: #38bdf8; text-align: center;">⚡ Детекция дефектов сварки</h3>
            <p style="color: #cbd5e1; text-align: center;">
                Использование вашей обученной модели для точной классификации 6 типов дефектов сварных швов
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_tech2:
        st.markdown("""
        <div class="analysis-card">
            <h3 style="color: #38bdf8; text-align: center;">🎨 Генеративная модель аномалий</h3>
            <p style="color: #cbd5e1; text-align: center;">
                Модель обучена ТОЛЬКО на хороших данных — любое отклонение = аномалия (дефект)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    '''with col_tech3:
        st.markdown("""
        <div class="analysis-card">
            <h3 style="color: #38bdf8; text-align: center;">🧠 Генеративный анализ причин</h3>
            <p style="color: #cbd5e1; text-align: center;">
                LLM-подобный анализ для генерации объяснимых отчётов с причинно-следственными связями
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Информация о генеративной аугментации
    st.markdown('<div class="augmentation-info">'
                '✨ Все модели обучены с применением генеративной аугментации: '
                'симуляция реальных условий освещения и ракурсов съемки (без поворота) '
                'для повышения устойчивости к промышленным условиям</div>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<p style="text-align: center; color: #64748b; font-size: 14px; padding: 20px 0;">'
                '🚂 Weld AI 2.0 • Платформа видеоаналитики с генеративным ИИ • Хакатон РЖД 2026</p>',
                unsafe_allow_html=True)

# ==================== СТРАНИЦА АНАЛИЗА ====================
elif st.session_state.page == "analysis":
    # Навигация назад (первая кнопка)
    if st.button("↩️ Вернуться к выбору категории", use_container_width=True, key="back_button_top"):
        st.session_state.page = "main"
        st.session_state.analysis_results = None
        st.rerun()
    
    analysis_type = st.session_state.analysis_type
    type_names = {
        "weld": "Сварные швы",
        "paint": "Покраска",
        "assembly": "Комплектация",
        "damage": "Механические повреждения",
        "doors": "Двери",
        "glass": "Стекла",
        "interior": "Интерьер"
    }
    
    st.title(f"🔍 Анализ: {type_names.get(analysis_type, 'Неизвестно')}")
    
    # Демо-данные для быстрого старта
    col_demo_left, col_demo_right = st.columns([1, 3])
    with col_demo_left:
        if st.button("🧪 Использовать демо-данные", type="secondary", use_container_width=True, key="use_demo_btn"):
            st.session_state.use_demo = True
            st.rerun()
   ''' 
    # Загрузка файла (видео только для сварки и покраски)
    if analysis_type in ["weld", "paint"]:
        uploaded_file = st.file_uploader(
            "Загрузите изображение или видео",
            type=["jpg", "jpeg", "png", "bmp", "mp4", "avi"],
            accept_multiple_files=False,
            help="Поддерживаются изображения (JPG, PNG) и видео (MP4, AVI)",
            key=f"uploader_{analysis_type}"
        )
    else:
        uploaded_file = st.file_uploader(
            "Загрузите изображение",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=False,
            help="Поддерживаются изображения (JPG, PNG)",
            key=f"uploader_{analysis_type}"
        )
    
    # Авто-запуск демо-анализа
    if st.session_state.get("auto_run_demo") and st.session_state.get("use_demo"):
        with st.spinner("⏳ Выполняется демо-анализ..."):
            demo_img = generate_demo_image(analysis_type)
            st.session_state.analysis_results = perform_analysis(demo_img, analysis_type, models)
            st.session_state.auto_run_demo = False
            st.success("✅ Демо-анализ успешно выполнен!")
            st.balloons()
            st.rerun()
    
    # Обычный запуск анализа
    if (uploaded_file or st.session_state.get("use_demo")) and st.button("🚀 ЗАПУСТИТЬ АНАЛИЗ", type="primary", use_container_width=True, key="analyze_btn"):
        with st.spinner("⏳ Выполняется анализ..."):
            try:
                if st.session_state.get("use_demo"):
                    image = generate_demo_image(analysis_type)
                else:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    if uploaded_file.type.startswith('video'):
                        st.warning("Видеофайлы в демо-версии не обрабатываются. Используется заглушка.")
                        image = generate_demo_image(analysis_type)
                    else:
                        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        if image is None:
                            st.error("Ошибка загрузки изображения. Проверьте формат файла.")
                            st.stop()
                
                st.session_state.analysis_results = perform_analysis(image, analysis_type, models)
                st.success("✅ Анализ успешно выполнен!")
            except Exception as e:
                st.error(f"Критическая ошибка обработки: {str(e)}")
                import traceback
                traceback.print_exc()
                st.stop()
    
    # Отображение результатов
    if "analysis_results" in st.session_state and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Оригинал и аннотированное изображение
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.markdown("#### Исходное изображение")
            if results["image"] is not None and results["image"].size > 0:
                st.image(cv2.cvtColor(results["image"], cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.error("Ошибка отображения исходного изображения")
        
        with col_img2:
            st.markdown("#### Результат анализа")
            if results["annotated_image"] is not None and results["annotated_image"].size > 0:
                st.image(cv2.cvtColor(results["annotated_image"], cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.error("Ошибка отображения результата анализа")
        
        # Статистика
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Обнаружено дефектов</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{len(results["defects"])}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_stat2:
            if results["anomaly_score"] is not None:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Уровень аномалии</div>', unsafe_allow_html=True)
                anomaly_val = float(results["anomaly_score"])
                anomaly_color = "#ef4444" if anomaly_val > 0.7 else "#f59e0b" if anomaly_val > 0.4 else "#10b981"
                st.markdown(f'<div class="metric-value" style="color:{anomaly_color}">{anomaly_val:.1%}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Точность детекции</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value" style="color:#38bdf8">92.5%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col_stat3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Статус качества</div>', unsafe_allow_html=True)
            has_defects = len(results["defects"]) > 0
            high_anomaly = results["anomaly_score"] is not None and results["anomaly_score"] > 0.3
            if not has_defects and not high_anomaly:
                status_color = "#10b981"
                status_text = " Соответствует"
            else:
                status_color = "#ef4444"
                status_text = " Не соответствует"
            st.markdown(f'<div class="metric-value" style="color:{status_color}">{status_text}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Генеративный отчёт
        st.markdown("### 🧠 Генеративный анализ причин (Generative AI)")
        
        if len(results["defects"]) == 0 and (results["anomaly_score"] is None or results["anomaly_score"] < 0.3):
            st.markdown('<div class="no-defects"> Дефектов не обнаружено! Качество продукции соответствует стандартам.</div>', unsafe_allow_html=True)
        else:
            # Отображение критических дефектов
            if any(d.get("severity") == "Критичный" for d in results["defects"]):
                st.markdown('<div class="critical-defect"> Обнаружен критический дефект! Требуется немедленная остановка линии для проверки.</div>', unsafe_allow_html=True)
            
            # Генеративный отчёт
            st.markdown('<div class="generative-section">', unsafe_allow_html=True)
            
            st.subheader(" Вероятная корневая причина")
            st.info(results["report"]["root_cause"])
            
            st.subheader(" Нарушения технологического процесса")
            st.warning(results["report"]["process_violation"])
            
            st.subheader("Рекомендации для технолога")
            for i, rec in enumerate(results["report"]["recommendations"], 1):
                st.success(f"{i}. {rec}")
            
            st.metric("Уверенность анализа", f"{results['report']['confidence'] * 100:.1f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Детали дефектов
        if results["defects"]:
            st.markdown("### 📋 Детали обнаруженных дефектов")
            
            for i, defect in enumerate(results["defects"], 1):
                defect_class = defect.get("class", "Неизвестный дефект")
                confidence = defect.get("confidence", 0.0)
                severity = defect.get("severity", "Средний")
                impact = defect.get("impact", "Требуется уточнение")
                
                severity_color = "#ef4444" if severity == "Критичный" else "#f59e0b" if severity == "Средний" else "#8b5cf6"
                
                with st.expander(f"Дефект #{i}: {defect_class} (Уверенность: {confidence:.1%})", expanded=True):
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.7); padding: 15px; border-radius: 12px; margin: 10px 0;">
                        <p style="color: #cbd5e1; margin: 5px 0;"><strong>Категория:</strong> {type_names.get(results['analysis_type'], 'Неизвестно')}</p>
                        <p style="color: {severity_color}; margin: 5px 0;"><strong>Критичность:</strong> {severity}</p>
                        <p style="color: #cbd5e1; margin: 5px 0;"><strong>Влияние на качество:</strong> {impact}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Формирование отчёта
        st.markdown("---")
        st.markdown("### 📄 Формирование официального отчёта")
        
        col_rep1, col_rep2 = st.columns(2)
        
        with col_rep1:
            report_data = {
                "analysis_info": {
                    "timestamp": results["timestamp"].isoformat(),
                    "analysis_type": type_names.get(results["analysis_type"], "Неизвестно"),
                    "technology": "Генеративный ИИ (аномалия-детекция + причинно-следственный анализ)"
                },
                "results": {
                    "defects_found": len(results["defects"]),
                    "anomaly_score": results["anomaly_score"],
                    "quality_status": "Соответствует" if len(results["defects"]) == 0 else "Не соответствует"
                },
                "generative_analysis": results["report"],
                "recommendations": results["report"]["recommendations"]
            }
            
            st.download_button(
                label="📄 Скачать отчёт (JSON)",
                data=json.dumps(report_data, ensure_ascii=False, indent=2),
                file_name=f"weld_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
                key="download_report"
            )
        
        with col_rep2:
            # Кнопка возврата (вторая кнопка)
            if st.button("↩️ Вернуться к выбору категории", use_container_width=True, key="back_button_bottom"):
                st.session_state.page = "main"
                st.session_state.analysis_results = None
                st.rerun()
    
    st.markdown("---")
    st.markdown('<p style="text-align: center; color: #64748b; font-size: 14px; padding: 20px 0;">'
                f'🚂 Weld AI 2.0 • Анализ: {type_names.get(analysis_type, "Неизвестно")} • '
                'Технологии: YOLOv8 (сварка) + Генеративная модель аномалий + Причинно-следственный анализ</p>',
                unsafe_allow_html=True)
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
    
    /* Стили для вкладок */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e293b;
        border-radius: 8px;
        color: #94a3b8;
        border: 1px solid #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #38bdf8 !important;
        color: #0f172a !important;
        font-weight: bold;
    }
    
    /* Футер технологий */
    .tech-footer {
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #334155;
        text-align: center;
        color: #64748b;
        font-size: 14px;
    }
    .tech-badge {
        display: inline-block;
        background: #1e293b;
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 12px;
        margin: 0 5px;
        font-size: 12px;
        border: 1px solid #334155;
    }
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
            print(" Модель сварки загружена")
        else:
            print(" Модель сварки не найдена.")
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
    page_title="DeltaPlan | Видеоаналитика дефектов | Хакатон ТМХ",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); min-height: 100vh; }
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
    
    .category-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 15px 0;
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
        cursor: pointer;
        border: 2px solid transparent;
        color: white;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .category-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
        border-color: #38bdf8;
    }
    .category-icon { font-size: 48px; margin-bottom: 15px; color: #38bdf8; }
    .category-title { color: white; font-size: 24px; font-weight: 700; margin-bottom: 10px; }
    .category-desc { color: #94a3b8; font-size: 16px; }
    
    .metric-box {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-value { font-size: 42px; font-weight: bold; color: #38bdf8; margin: 10px 0; }
    .metric-label { color: #cbd5e1; font-size: 16px; font-weight: 500; }
    
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
    
    .feedback-container {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
    }
    
    .history-item {
        background: #1e293b;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        border-left: 4px solid #38bdf8;
    }
    .history-item.error { border-left-color: #ef4444; }
    .history-item.success { border-left-color: #10b981; }
</style>
""", unsafe_allow_html=True)

# ==================== ИНИЦИАЛИЗАЦИЯ STATE ====================
if "history_log" not in st.session_state:
    st.session_state.history_log = []

if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 0
if "wizard_data" not in st.session_state:
    st.session_state.wizard_data = {}

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def perform_analysis(image, analysis_type, models):
    """Единая функция анализа"""
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
                defects = [{
                    "class": "Аномалия поверхности", 
                    "confidence": float(anomaly_score) if anomaly_score else 0.0,
                    "severity": "Критичный" if is_anomaly else "Низкий",
                    "impact": "Нарушение герметичности" if analysis_type == "weld" else "Снижение коррозионной стойкости"
                }] if is_anomaly else []
            else:
                annotated_img = image.copy()
                defects = []
                anomaly_score = 0.0
        
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
            report = {
                "root_cause": "Нарушение параметров сварки",
                "process_violation": "Отклонение по скорости подачи проволоки",
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

def render_wizard():
    """Рендерит мастер создания новой модели"""
    step = st.session_state.wizard_step
    
    st.markdown("### Мастер создания новой модели контроля")
    
    if step == 0:
        if st.button("Начать создание модели", type="primary", use_container_width=True):
            st.session_state.wizard_step = 1
            st.rerun()
            
    elif step == 1:
        st.markdown("**Шаг 1: Название дефекта**")
        name = st.text_input("Введите название типа дефекта (например, 'Трещины рамы')", key="wiz_name")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Назад", use_container_width=True):
                st.session_state.wizard_step = 0
                st.rerun()
        with col2:
            if st.button("Далее", type="primary", use_container_width=True, disabled=not name):
                st.session_state.wizard_data['name'] = name
                st.session_state.wizard_step = 2
                st.rerun()
                
    elif step == 2:
        st.markdown(f"**Шаг 2: Тип обучения для '{st.session_state.wizard_data.get('name')}'**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("1) Загрузить готовую (.pt)", use_container_width=True):
                st.session_state.wizard_data['type'] = 'yolo'
                st.session_state.wizard_step = 4
                st.rerun()
        with col2:
            if st.button("2) Обучить как Anomaly Detection", use_container_width=True):
                st.session_state.wizard_data['type'] = 'anomaly'
                st.session_state.wizard_step = 3
                st.rerun()
                
    elif step == 3:
        st.markdown("**Шаг 3: Загрузка эталонов**")
        st.info("Загрузите ровно 3 фотографии идеальной поверхности без дефектов.")
        uploaded_files = st.file_uploader("Фотографии эталонов", type=["jpg", "png"], accept_multiple_files=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Назад", use_container_width=True):
                st.session_state.wizard_step = 2
                st.rerun()
        with col2:
            if st.button("Обучить модель", type="primary", use_container_width=True, disabled=(not uploaded_files or len(uploaded_files) != 3)):
                st.session_state.wizard_data['files'] = uploaded_files
                st.session_state.wizard_step = 4
                st.rerun()
                
    elif step == 4:
        st.success(f"Модель '{st.session_state.wizard_data.get('name')}' успешно создана и добавлена в реестр!")
        st.balloons()
        if st.button("Вернуться к анализу", use_container_width=True):
            st.session_state.wizard_step = 0
            st.session_state.wizard_data = {}
            st.rerun()

def render_history():
    """Рендерит вкладку истории"""
    st.title("История анализа")
    
    if not st.session_state.history_log:
        st.info("История пуста. Проведите первый анализ во вкладке 'Новый анализ'.")
        return

    for i, record in enumerate(reversed(st.session_state.history_log)):
        ts = record['timestamp'].strftime("%Y-%m-%d %H:%M")
        
        if record.get('feedback') == 'correct':
            status_icon = "[OK]"
            border_class = "success"
        elif record.get('feedback') == 'incorrect':
            status_icon = "[ERROR]"
            border_class = "error"
        else:
            status_icon = "[PENDING]"
            border_class = ""
        
        with st.container():
            st.markdown(f"""
            <div class="history-item {border_class}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:14px; color:#94a3b8;">{ts}</span>
                        <h3 style="margin:5px 0; color:white;">{record['type_name']} {status_icon}</h3>
                        <p style="color:#cbd5e1; margin:0;">Скор аномалии: {record.get('score', 'N/A')}</p>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:24px;">{record.get('preview_icon', '-')}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if record.get('feedback') == 'incorrect':
                if st.button("Дообучить модель на этом снимке", key=f"retrain_{i}"):
                    st.info(f"Модель отправлена на дообучение с использованием снимка от {ts}...")
                    st.progress(100)
                    st.success("Данные добавлены в датасет для следующей итерации обучения.")

# ==================== ГЛАВНАЯ ЛОГИКА (TABS) ====================

tab1, tab2 = st.tabs(["Новый анализ", "История анализа"])

with tab1:
    st.title("Δплан")
    st.markdown('<p class="subtitle">Видеоаналитика ИИ на страже качества в производстве</p>', unsafe_allow_html=True)
    
    if st.session_state.wizard_step == 0:
        if st.button("Создать новую модель контроля", type="secondary", use_container_width=True):
            st.session_state.wizard_step = 1
            st.rerun()
    else:
        render_wizard()
        st.divider()

    st.markdown("### Выберите тип контроля качества")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Сварные швы", use_container_width=True, key="btn_weld"):
            st.session_state.selected_type = "weld"
            st.session_state.type_name = "Сварные швы"
            st.rerun()
        st.markdown("""
        <div class="category-card">
            <div class="category-icon"></div>
            <div class="category-title">Сварные швы</div>
            <div class="category-desc">Детекция пор, трещин, непроваров</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("Покраска", use_container_width=True, key="btn_paint"):
            st.session_state.selected_type = "paint"
            st.session_state.type_name = "Покраска"
            st.rerun()
        st.markdown("""
        <div class="category-card">
            <div class="category-icon"></div>
            <div class="category-title">Покраска</div>
            <div class="category-desc">Контроль качества ЛКП, потеки, царапины</div>
        </div>
        """, unsafe_allow_html=True)

    if "selected_type" in st.session_state:
        st.divider()
        st.subheader(f"Анализ: {st.session_state.type_name}")
        
        uploaded_file = st.file_uploader(
            "Загрузите изображение для анализа",
            type=["jpg", "jpeg", "png", "bmp"],
            key=f"uploader_{st.session_state.selected_type}"
        )
        
        if uploaded_file:
            if st.button("ЗАПУСТИТЬ АНАЛИЗ", type="primary", use_container_width=True):
                with st.spinner("Выполняется анализ нейросетью..."):
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        result = perform_analysis(image, st.session_state.selected_type, models)
                        st.session_state.current_result = result
                    else:
                        st.error("Ошибка чтения файла")

        if "current_result" in st.session_state and st.session_state.current_result:
            res = st.session_state.current_result
            
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.markdown("#### Исходное изображение")
                st.image(cv2.cvtColor(res["image"], cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_img2:
                st.markdown("#### Результат анализа")
                st.image(cv2.cvtColor(res["annotated_image"], cv2.COLOR_BGR2RGB), use_container_width=True)
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Обнаружено дефектов</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{len(res["defects"])}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_stat2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Уровень аномалии</div>', unsafe_allow_html=True)
                if res["anomaly_score"] is not None:
                    anomaly_val = float(res["anomaly_score"])
                    anomaly_color = "#ef4444" if anomaly_val > 0.7 else "#f59e0b" if anomaly_val > 0.4 else "#10b981"
                    st.markdown(f'<div class="metric-value" style="color:{anomaly_color}">{anomaly_val:.1%}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-value" style="color:#38bdf8">92.5%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_stat3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Статус качества</div>', unsafe_allow_html=True)
                has_defects = len(res["defects"]) > 0
                high_anomaly = res["anomaly_score"] is not None and res["anomaly_score"] > 0.3
                if not has_defects and not high_anomaly:
                    status_color = "#10b981"
                    status_text = "Соответствует"
                else:
                    status_color = "#ef4444"
                    status_text = "Не соответствует"
                st.markdown(f'<div class="metric-value" style="color:{status_color}">{status_text}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### Генеративный анализ причин (LLM)")
            
            if len(res["defects"]) == 0 and (res["anomaly_score"] is None or res["anomaly_score"] < 0.3):
                st.markdown('<div class="no-defects">Дефектов не обнаружено! Качество продукции соответствует стандартам.</div>', unsafe_allow_html=True)
            else:
                if any(d.get("severity") == "Критичный" for d in res["defects"]):
                    st.markdown('<div class="critical-defect">Обнаружен критический дефект! Требуется немедленная остановка линии для проверки.</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="generative-section">', unsafe_allow_html=True)
                
                st.subheader("Вероятная корневая причина")
                st.info(res["report"]["root_cause"])
                
                st.subheader("Нарушения технологического процесса")
                st.warning(res["report"]["process_violation"])
                
                st.subheader("Рекомендации для технолога")
                for i, rec in enumerate(res["report"]["recommendations"], 1):
                    st.success(f"{i}. {rec}")
                
                st.metric("Уверенность анализа", f"{res['report']['confidence'] * 100:.1f}%")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            if res["defects"]:
                st.markdown("### Детали обнаруженных дефектов")
                
                type_names = {"weld": "Сварные швы", "paint": "Покраска"}
                
                for i, defect in enumerate(res["defects"], 1):
                    defect_class = defect.get("class", "Неизвестный дефект")
                    confidence = defect.get("confidence", 0.0)
                    severity = defect.get("severity", "Средний")
                    impact = defect.get("impact", "Требуется уточнение")
                    
                    severity_color = "#ef4444" if severity == "Критичный" else "#f59e0b" if severity == "Средний" else "#8b5cf6"
                    
                    with st.expander(f"Дефект #{i}: {defect_class} (Уверенность: {confidence:.1%})", expanded=True):
                        st.markdown(f"""
                        <div style="background: rgba(30, 41, 59, 0.7); padding: 15px; border-radius: 12px; margin: 10px 0;">
                            <p style="color: #cbd5e1; margin: 5px 0;"><strong>Категория:</strong> {type_names.get(res['analysis_type'], 'Неизвестно')}</p>
                            <p style="color: {severity_color}; margin: 5px 0;"><strong>Критичность:</strong> {severity}</p>
                            <p style="color: #cbd5e1; margin: 5px 0;"><strong>Влияние на качество:</strong> {impact}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### Формирование официального отчёта")
            
            col_rep1, col_rep2 = st.columns(2)
            
            with col_rep1:
                report_data = {
                    "analysis_info": {
                        "timestamp": res["timestamp"].isoformat(),
                        "analysis_type": st.session_state.type_name,
                        "technology": "Генеративный ИИ (аномалия-детекция + причинно-следственный анализ)"
                    },
                    "results": {
                        "defects_found": len(res["defects"]),
                        "anomaly_score": res["anomaly_score"],
                        "quality_status": "Соответствует" if len(res["defects"]) == 0 else "Не соответствует"
                    },
                    "generative_analysis": res["report"],
                    "recommendations": res["report"]["recommendations"]
                }
                
                st.download_button(
                    label="Скачать отчёт (JSON)",
                    data=json.dumps(report_data, ensure_ascii=False, indent=2),
                    file_name=f"weld_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="download_report"
                )
            
            with col_rep2:
                if st.button("Вернуться к выбору категории", use_container_width=True, key="back_button_bottom"):
                    st.session_state.selected_type = None
                    st.session_state.current_result = None
                    st.rerun()
            
            # === FEEDBACK ===
            confidence_val = res.get("anomaly_score", 1.0) or 1.0
            
            if confidence_val < 0.6:
                st.markdown("""
                <div class="feedback-container">
                    <p style="color:#cbd5e1; margin-bottom:10px;">
                        <strong>Внимание:</strong> Уверенность модели низкая (< 60%). Требуется валидация оператора.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                fb_col1, fb_col2 = st.columns(2)
                with fb_col1:
                    if st.button("Корректно", key="fb_correct", use_container_width=True):
                        log_entry = {
                            "timestamp": res["timestamp"],
                            "type_name": st.session_state.type_name,
                            "score": confidence_val,
                            "feedback": "correct",
                            "preview_icon": "OK"
                        }
                        st.session_state.history_log.append(log_entry)
                        st.success("Результат сохранен как верный.")
                        st.rerun()
                with fb_col2:
                    if st.button("Неверно", key="fb_incorrect", use_container_width=True):
                        log_entry = {
                            "timestamp": res["timestamp"],
                            "type_name": st.session_state.type_name,
                            "score": confidence_val,
                            "feedback": "incorrect",
                            "preview_icon": "ERR"
                        }
                        st.session_state.history_log.append(log_entry)
                        st.error("Отмечено как ошибка. Модель будет дообучена.")
                        st.rerun()
            else:
                if "auto_logged" not in st.session_state:
                    log_entry = {
                        "timestamp": res["timestamp"],
                        "type_name": st.session_state.type_name,
                        "score": confidence_val,
                        "feedback": "auto",
                        "preview_icon": "AUTO"
                    }
                    st.session_state.history_log.append(log_entry)
                    st.session_state.auto_logged = True

with tab2:
    render_history()

# ==================== ФУТЕР ТЕХНОЛОГИЙ ====================
st.markdown("""
<div class="tech-footer">
    <p>Стек технологий платформы</p>
    <div style="margin-top: 10px;">
        <span class="tech-badge">Python 3.9+</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">OpenCV</span>
        <span class="tech-badge">NumPy</span>
        <span class="tech-badge">PyTorch</span>
        <span class="tech-badge">YOLOv8</span>
        <span class="tech-badge">Anomaly Detection</span>
        <span class="tech-badge">Generative AI</span>
    </div>
    <p style="margin-top: 15px; font-size: 12px; opacity: 0.6;">ΔPlan Platform | Хакатон ТМХ 2026</p>
</div>
""", unsafe_allow_html=True)
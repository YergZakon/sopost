import streamlit as st
import os
import tempfile
import subprocess
from pydub import AudioSegment
from openai import OpenAI
from datetime import datetime

# ----------------------- Настройка страницы и CSS -----------------------
st.set_page_config(
    page_title="Система транскрибации следственных действий",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_css():
    st.markdown("""
        <style>
        /* Основной фон */
        .stApp {
            background-color: #E5E7EB;
        }
        
        /* Контейнеры */
        .element-container, div.stButton, div.stDownloadButton {
            background-color: #F3F4F6;
            padding: 0.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        /* Кнопки */
        .stButton>button {
            width: 100%;
            background-color: #1e3a8a;
            color: white;
            border: none;
            padding: 0.75rem 1rem;
            border-radius: 0.375rem;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #1e40af;
        }

        /* Заголовок */
        .header-container {
            display: flex;
            align-items: center;
            padding: 1.5rem;
            background-color: #1e3a8a;
            color: white;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Загрузка файлов */
        .uploadedFile {
            border: 2px solid #D1D5DB;
            border-radius: 0.5rem;
            padding: 1rem;
            background-color: #F9FAFB;
        }

        /* Текстовые области */
        .stTextArea textarea {
            background-color: #F9FAFB;
            border: 1px solid #D1D5DB;
            border-radius: 0.375rem;
        }

        /* Информационные блоки */
        .stAlert {
            background-color: #F9FAFB;
            border: 1px solid #D1D5DB;
            border-radius: 0.375rem;
        }
        
        /* Статус бар */
        .stStatusWidget {
            background-color: #F9FAFB;
            border: 1px solid #D1D5DB;
            border-radius: 0.375rem;
            padding: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

# ----------------------- Инициализация OpenAI -----------------------
def init_openai():
    try:
        # Инициализация клиента OpenAI. API-ключ должен быть указан в st.secrets.
        return OpenAI(api_key=st.secrets["openai_api_key"])
    except Exception as e:
        st.error("Ошибка инициализации OpenAI API: " + str(e))
        return None

# ----------------------- Генерация номера материала -----------------------
def generate_case_number():
    now = datetime.now()
    return f"М-{now.strftime('%Y%m%d')}-{now.strftime('%H%M%S')}"

# ----------------------- Проверка наличия FFmpeg -----------------------
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# ----------------------- Извлечение аудио из видео -----------------------
def extract_audio(uploaded_file):
    # Сохраняем временный видеофайл
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(uploaded_file.read())
        video_path = tmpfile.name

    audio_path = os.path.splitext(video_path)[0] + '.mp3'
    
    try:
        subprocess.run([
            "ffmpeg", 
            "-i", video_path, 
            "-q:a", "0",
            "-map", "a", 
            audio_path
        ], check=True, capture_output=True)
        
        audio = AudioSegment.from_mp3(audio_path)
        duration = len(audio) / 1000  # длительность в секундах
        
        return audio_path, duration
    except Exception as e:
        st.error(f"Ошибка при извлечении аудио: {str(e)}")
        raise
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

# ----------------------- Транскрибация аудио -----------------------
def transcribe_audio(client, audio_file, language='ru'):
    try:
        with open(audio_file, "rb") as file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=file,
                language=language
            )
        return transcript.text
    except Exception as e:
        st.error(f"Ошибка при транскрибации: {str(e)}")
        raise
    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)

# ----------------------- Функции для обработки текста -----------------------
# Здесь мы добавляем контекст следственных действий и роль следствия при установлении достоверности показаний.

def summarize_text(client, text, language='ru'):
    prompt = (
        f"В рамках следственных действий установите достоверность показаний. Суммируйте следующий текст на языке {language}:\n\n"
        f"{text}\n\n"
        "Дайте краткий вывод основных моментов, с учетом роли следствия в установлении достоверности показаний."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": "Вы действуете как опытный следователь, оценивающий достоверность показаний."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        st.error(f"Ошибка при суммаризации: {str(e)}")
        return ""

def check_sequence(client, text):
    prompt = (
        "Проанализируйте следующий текст показаний с точки зрения следственных действий. Проверьте логическую последовательность изложения и "
        "выделите несоответствия или пропущенные шаги, важные для установления достоверности показаний:\n\n" + text
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": "Вы следователь, оценивающий последовательность изложения показаний."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        analysis = response.choices[0].message.content.strip()
        return analysis
    except Exception as e:
        st.error(f"Ошибка при проверке последовательности: {str(e)}")
        return ""

def extract_key_facts(client, text):
    prompt = (
        "Извлеките из следующего текста ключевые факты, имеющие значение в следственном деле, которые помогут установить достоверность показаний:\n\n"
        + text
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": "Вы следователь, выделяющий существенные факты для установления истины."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        key_facts = response.choices[0].message.content.strip()
        return key_facts
    except Exception as e:
        st.error(f"Ошибка при извлечении ключевых фактов: {str(e)}")
        return ""

def check_contradictions(client, text1, text2):
    prompt = (
        "Сравните следующие два показания и определите противоречия или расхождения между ними, которые могут повлиять на достоверность показаний:\n\n"
        "Показания лица №1:\n" + text1 + "\n\n"
        "Показания лица №2:\n" + text2
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": "Вы следователь, сопоставляющий показания для выявления противоречий."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        contradictions = response.choices[0].message.content.strip()
        return contradictions
    except Exception as e:
        st.error(f"Ошибка при проверке противоречий: {str(e)}")
        return ""

def formulate_questions(client, contradictions):
    prompt = (
        "На основе следующих противоречий, выявленных в показаниях, сформулируйте вопросы для уточнения и проверки достоверности сведений:\n\n"
        + contradictions
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": "Вы следователь, формирующий вопросы для уточнения показаний."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        questions = response.choices[0].message.content.strip()
        return questions
    except Exception as e:
        st.error(f"Ошибка при формировании вопросов: {str(e)}")
        return ""

# ----------------------- Основная функция приложения -----------------------
def main():
    load_css()

    # Проверяем наличие FFmpeg
    if not check_ffmpeg():
        st.error("FFmpeg не установлен. Пожалуйста, установите FFmpeg для работы с видео.")
        st.stop()

    # Заголовок
    st.markdown("""
        <div class="header-container">
            <h1 style="margin:0; color:white;">🛡️ Система транскрибации следственных действий</h1>
        </div>
    """, unsafe_allow_html=True)

    # Инициализация OpenAI клиента
    client = init_openai()
    if not client:
        st.error("⚠️ Введите API ключ OpenAI в боковой панели для начала работы")
        return

    # Основной интерфейс
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📤 Загрузка материалов")
        uploaded_file_1 = st.file_uploader(
            "Выберите видео или аудио для показаний лица №1",
            key="face1",
            type=["mp4", "avi", "mov", "mp3", "wav"],
        )
        uploaded_file_2 = st.file_uploader(
            "Выберите видео или аудио для показаний лица №2",
            key="face2",
            type=["mp4", "avi", "mov", "mp3", "wav"],
        )
    with col2:
        st.markdown("### 📋 Информация о материале")
        case_number = st.text_input(
            "Номер материала",
            value=generate_case_number(),
            disabled=True
        )
        language = st.selectbox(
            "Язык материала",
            options=["ru", "kk", "en"],
            format_func=lambda x: {
                "ru": "🇷🇺 Русский",
                "kk": "🇰🇿 Қазақ тілі",
                "en": "🇬🇧 English"
            }[x]
        )

    st.markdown("---")
    st.info(
        "ℹ️ Все материалы обрабатываются с соблюдением требований информационной безопасности. "
        "Файлы автоматически удаляются после обработки."
    )

    if st.button("🚀 Начать обработку"):
        transcription1 = transcription2 = ""
        summary1 = summary2 = ""
        sequence_check1 = sequence_check2 = ""
        key_facts1 = key_facts2 = ""

        # Обработка материала для лица №1
        if uploaded_file_1:
            st.markdown("### Обработка показаний лица №1")
            try:
                with st.spinner("Извлечение аудио для лица №1..."):
                    audio_path1, duration1 = extract_audio(uploaded_file_1)
                    st.write(f"Длительность аудио: {int(duration1)} сек.")
                with st.spinner("Транскрибация показаний лица №1..."):
                    transcription1 = transcribe_audio(client, audio_path1, language)
                with st.spinner("Суммаризация показаний лица №1..."):
                    summary1 = summarize_text(client, transcription1, language)
                with st.spinner("Проверка логической последовательности для лица №1..."):
                    sequence_check1 = check_sequence(client, transcription1)
                with st.spinner("Извлечение ключевых фактов для лица №1..."):
                    key_facts1 = extract_key_facts(client, transcription1)
            except Exception as e:
                st.error("Ошибка при обработке материала лица №1: " + str(e))
        else:
            st.warning("Материал для лица №1 не загружен.")

        # Обработка материала для лица №2
        if uploaded_file_2:
            st.markdown("### Обработка показаний лица №2")
            try:
                with st.spinner("Извлечение аудио для лица №2..."):
                    audio_path2, duration2 = extract_audio(uploaded_file_2)
                    st.write(f"Длительность аудио: {int(duration2)} сек.")
                with st.spinner("Транскрибация показаний лица №2..."):
                    transcription2 = transcribe_audio(client, audio_path2, language)
                with st.spinner("Суммаризация показаний лица №2..."):
                    summary2 = summarize_text(client, transcription2, language)
                with st.spinner("Проверка логической последовательности для лица №2..."):
                    sequence_check2 = check_sequence(client, transcription2)
                with st.spinner("Извлечение ключевых фактов для лица №2..."):
                    key_facts2 = extract_key_facts(client, transcription2)
            except Exception as e:
                st.error("Ошибка при обработке материала лица №2: " + str(e))
        else:
            st.warning("Материал для лица №2 не загружен.")

        # Отображение результатов для лица №1
        if transcription1:
            st.markdown("#### Результаты для лица №1")
            st.text_area("Транскрипция лица №1", value=transcription1, height=200)
            st.text_area("Суммаризация лица №1", value=summary1, height=100)
            st.text_area("Анализ последовательности лица №1", value=sequence_check1, height=100)
            st.text_area("Ключевые факты лица №1", value=key_facts1, height=100)
            st.download_button(
                "⬇️ Скачать транскрипцию лица №1",
                data=transcription1,
                file_name=f"Протокол_лицо1_{case_number}.txt",
                mime="text/plain"
            )

        # Отображение результатов для лица №2
        if transcription2:
            st.markdown("#### Результаты для лица №2")
            st.text_area("Транскрипция лица №2", value=transcription2, height=200)
            st.text_area("Суммаризация лица №2", value=summary2, height=100)
            st.text_area("Анализ последовательности лица №2", value=sequence_check2, height=100)
            st.text_area("Ключевые факты лица №2", value=key_facts2, height=100)
            st.download_button(
                "⬇️ Скачать транскрипцию лица №2",
                data=transcription2,
                file_name=f"Протокол_лицо2_{case_number}.txt",
                mime="text/plain"
            )

        # Сопоставление показаний и формирование вопросов, если оба материала загружены
        if transcription1 and transcription2:
            st.markdown("## Сопоставление показаний и установление достоверности")
            with st.spinner("Проверка противоречий между показаниями..."):
                contradictions = check_contradictions(client, transcription1, transcription2)
            with st.spinner("Формирование вопросов для уточнения показаний..."):
                questions = formulate_questions(client, contradictions)
            st.text_area("Найденные противоречия", value=contradictions, height=150)
            st.text_area("Сформированные вопросы", value=questions, height=150)

    # Футер
    st.markdown("---")
    st.markdown("""
        <div style="text-align:center; padding:1rem; background-color:#F3F4F6; 
                    border-radius:0.5rem; margin-top:2rem;">
            <p style="color:#6B7280; margin:0;">
                © 2024 Система автоматизированной обработки процессуальных материалов. 
                Версия 1.0.0
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

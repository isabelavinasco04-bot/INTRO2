import streamlit as st
import whisper
import tempfile
from PIL import Image

# 🌸 Estilos personalizados (CSS)
st.markdown("""
    <style>
    body {
        background-color: #fff6fb;
        color: #333333;
        font-family: 'Poppins', sans-serif;
    }
    .stButton>button {
        background-color: #ff66b3;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff3385;
        transform: scale(1.05);
    }
    .css-1d391kg {
        background-color: #ffe0ef !important;
    }
    h1, h2, h3 {
        color: #ff3385;
    }
    </style>
""", unsafe_allow_html=True)

# 🌷 Título
st.title("🎧 Conversor de Audio a Texto")
st.write("Convierte fácilmente tus grabaciones en texto. Solo sube el archivo y deja que la magia suceda 💫")

# 🌼 Imagen decorativa
image = Image.open("Interfaces Mult2.png")
st.image(image, caption="Interfaces multimodales", use_column_width=True)

# 🎵 Subir audio
audio_file = st.file_uploader("📂 Sube tu archivo de audio (mp3, wav, m4a, etc.)", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    st.audio(audio_file, format="audio/mp3")
    st.write("✨ Procesando tu audio... espera un momento ⏳")

    # Guardar archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(audio_file.read())
        temp_path = temp_file.name

    # Cargar modelo Whisper
    model = whisper.load_model("tiny")
    result = model.transcribe(temp_path, language="es")

    # Mostrar transcripción
    st.subheader("📝 Transcripción:")
    st.success(result["text"])

    # Botón de descarga
    st.download_button(
        label="⬇️ Descargar transcripción",
        data=result["text"],
        file_name="transcripcion.txt",
        mime="text/plain"
    )

# 🌸 Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    st.write("Selecciona el modelo de Whisper:")
    model_choice = st.radio(
        "Modelo",
        ("tiny", "base", "small", "medium", "large"),
        index=0
    )
    st.write("🌟 Mientras más grande el modelo, mejor precisión (pero más lento).")

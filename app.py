import streamlit as st
import os
import tempfile
import whisper
from PIL import Image

# ---------- Configuración ----------
st.set_page_config(page_title="🎧 Audio a Texto", page_icon="🎙️", layout="centered")

st.title("🎧 Conversor de Audio a Texto")
st.write("Sube tu archivo de audio y convierte tu voz en texto fácilmente 💫")

# Imagen decorativa (opcional)
try:
    image = Image.open("Zayn2.jpg")
    st.image(image, caption="Transcripción con IA", use_container_width=True)
except Exception:
    st.info("Pon una imagen llamada 'Zayn2.jpg' en la carpeta para mostrarla 😊")

# ---------- Subida de archivo ----------
uploaded_audio = st.file_uploader(
    "📂 Sube un archivo de audio (.mp3, .wav, .m4a, .flac, .ogg)",
    type=["mp3", "wav", "m4a", "flac", "ogg"]
)

# ---------- Selección de idioma ----------
stt_lang = st.selectbox(
    "🌎 Idioma del audio (puedes dejarlo en Auto)",
    ["Auto", "Español", "English", "Italiano"],
    index=0
)
lang_map = {"Auto": None, "Español": "es", "English": "en", "Italiano": "it"}

# ---------- Selección del modelo ----------
model_name = st.selectbox(
    "🧠 Modelo Whisper",
    ["tiny", "base", "small"],
    index=0,
    help="‘tiny’ es rápido, ‘base’ y ‘small’ son más precisos pero más lentos."
)

# ---------- Transcripción ----------
if st.button("🚀 Transcribir Audio"):
    if not uploaded_audio:
        st.warning("Por favor, sube un archivo de audio primero.")
    else:
        with st.spinner("Transcribiendo tu audio... 🎶"):
            try:
                # Guardar archivo temporal
                suffix = os.path.splitext(uploaded_audio.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_audio.read())
                    tmp_path = tmp.name

                # Cargar modelo Whisper
                model = whisper.load_model(model_name)
                result = model.transcribe(
                    tmp_path,
                    language=lang_map[stt_lang],
                    task="transcribe"
                )

                transcript = (result or {}).get("text", "").strip()

                if transcript:
                    st.success("✨ ¡Transcripción completada!")
                    st.text_area("📝 Texto obtenido:", value=transcript, height=200)
                    st.download_button(
                        "⬇️ Descargar transcripción",
                        data=transcript.encode("utf-8"),
                        file_name="transcripcion.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("No se obtuvo texto. Prueba con otro modelo o revisa el audio.")
            except Exception as e:
                st.error(f"Ocurrió un error: {e}")
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Configuración")
    st.caption("Whisper corre localmente y requiere tener instalado FFmpeg.")
    st.caption("Modelos más grandes = mayor precisión, pero más lentos.")

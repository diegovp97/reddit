import praw
import csv
import streamlit as st
from transformers import pipeline, AutoTokenizer
from dotenv import load_dotenv
import os

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Obtener las credenciales de Reddit desde las variables de entorno
reddit = praw.Reddit(
    client_id=os.getenv('CLIENT_ID'),
    client_secret=os.getenv('CLIENT_SECRET'),
    user_agent=os.getenv('USER_AGENT'),
    username=os.getenv('USERNAME'),
    password=os.getenv('PASSWORD')
)

# Configurar el tokenizer y el modelo de análisis de emociones
tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# Función para analizar emociones y generar recomendaciones
def analizar_y_dar_consejos(texto):
    if not isinstance(texto, str) or len(texto.strip()) == 0:
        return "neutral", "El contenido está vacío o no es un texto válido."
    
    tokens = tokenizer(
        texto, 
        max_length=512, 
        padding='max_length', 
        truncation=True, 
        return_tensors="pt"
    )

    emociones = emotion_classifier(texto[:512])  # Limitar a 512 caracteres
    emociones_ordenadas = sorted(emociones[0], key=lambda x: x['score'], reverse=True)
    emocion_principal = emociones_ordenadas[0]['label']

    consejos = {
        'anger': "Considera realizar ejercicios de relajación como la respiración profunda o dar un paseo para calmar tu mente.",
        'sadness': "Podría ser útil hablar con un amigo cercano o profesional de salud mental para expresar lo que sientes.",
        'fear': "Identifica qué te preocupa específicamente y da pequeños pasos para enfrentarlo. Considera hablar con alguien de confianza.",
        'joy': "Es maravilloso sentir alegría. Comparte tu felicidad con otros o dedica tiempo a lo que te apasiona.",
        'disgust': "Tómate un momento para reflexionar sobre lo que te provoca esta sensación y si es algo que puedas cambiar o evitar.",
        'trust': "Valora las conexiones de confianza que tienes y sigue cultivando relaciones positivas en tu vida.",
        'anticipation': "Planifica y prepárate para lo que viene, pero también recuerda vivir el presente sin dejar que la ansiedad te consuma.",
        'surprise': "Las sorpresas pueden ser emocionantes o estresantes. Si es algo positivo, disfrútalo; si es negativo, busca adaptarte con calma.",
        'hope': "Sigue adelante con esa actitud positiva, pero también es bueno prepararse para los retos que puedan surgir.",
        'submission': "Si sientes que te estás sometiendo demasiado en una situación, reflexiona si es lo mejor para ti a largo plazo.",
        'remorse': "Todos cometemos errores, lo importante es aprender de ellos y seguir adelante con una mentalidad de crecimiento.",
        'contempt': "Reflexiona sobre lo que te provoca este desprecio y si es algo que se puede cambiar con un cambio de perspectiva.",
        'aggression': "Si sientes impulsos agresivos, trata de canalizar esa energía en actividades físicas o creativas para liberar la tensión.",
        'love': "El amor es un sentimiento hermoso. Aprovecha para fortalecer las relaciones que valoras y comparte tu cariño."
    }

    consejo = consejos.get(emocion_principal, "Recuerda siempre cuidar tu bienestar emocional y buscar ayuda si es necesario.")
    return emocion_principal, consejo

# Configuración de Streamlit
st.title("Análisis de emociones de Reddit")
st.write("Este es un análisis de emociones para publicaciones del subreddit **Depresion**.")

# Obtener las 5 publicaciones más recientes del subreddit
subreddit = reddit.subreddit("Depresion")
publicaciones = subreddit.new(limit=5)

# Mostrar publicaciones en Streamlit
for submission in publicaciones:
    st.subheader(submission.title)
    st.write(f"**Autor**: {submission.author}")
    st.write(f"**Contenido**: {submission.selftext}")
    st.write(f"[Ver publicación en Reddit]({submission.url})")
    
    # Analizar emociones y generar consejo
    emocion_detectada, consejo = analizar_y_dar_consejos(submission.selftext)
    
    # Mostrar emociones y consejo en Streamlit
    st.write(f"**Emoción detectada**: {emocion_detectada}")
    st.write(f"**Consejo**: {consejo}")
    st.write("---")

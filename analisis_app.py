# --------------------------------------------------------------------------
# COMMENT ANALYSIS PLATFORM V13.0 - The Final Client-Ready Edition
# All inputs are processed; all errors are classified as Neutral.
# --------------------------------------------------------------------------
import streamlit as st
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import google.generativeai as genai
import time
import emoji
from collections import Counter
import concurrent.futures
import re
import json

# --- Page Configuration ---
st.set_page_config(page_title="Comment Analysis", layout="wide")

# --- STYLE INJECTION ---
st.markdown("""
<style>
    .stApp { background-color: #F0F2F6; }
    .stButton>button { background-color: #8A2BE2; color: #FFFFFF; border-radius: 8px; border: none; padding: 10px 20px; font-weight: bold; }
    .stButton>button:hover { background-color: #7B1FA2; }
    .st-emotion-cache-1r4qj8v, .st-emotion-cache-0 { background-color: #FFFFFF; border-radius: 10px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)


# --- CORE FUNCTIONS ---
def parse_ai_batch_response(response_text, original_batch):
    """Parses the structured JSON-like response from the AI for a batch."""
    try:
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        else:
            json_str = response_text
        analyses = json.loads(json_str)
        for i, analysis in enumerate(analyses):
            if i < len(original_batch):
                analysis['Original Comment'] = original_batch[i]
        return analyses
    except (json.JSONDecodeError, IndexError):
        # CHANGE: If parsing fails, classify the whole batch as Neutral.
        return [{'Original Comment': c, 'Sentiment': 'Neutral', 'Explanation': 'Unanalyzable content.'} for c in original_batch]

@st.cache_data(ttl="24h")
def analyze_comment_batch_cached(_api_key, comment_batch):
    """Worker function to analyze a BATCH of comments."""
    genai.configure(api_key=_api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    comments_str = "\n".join([f'{i+1}. "{comment}"' for i, comment in enumerate(comment_batch)])
    prompt = f"""
    Act as a sentiment analysis API. Analyze EACH of the following customer comments.
    Your response MUST be a single, valid JSON array `[]` containing one JSON object `{{}}` for each comment.
    Maintain the original order.
    Each JSON object must have these exact keys: "Sentiment", "Explanation".
    - Sentiment: Must be one of "Positive", "Negative", or "Neutral".
    - Explanation: A concise justification for the sentiment.
    - If a comment is nonsensical (e.g., a single letter) or you cannot determine a sentiment, classify it as "Neutral".

    COMMENT LIST:
    {comments_str}
    """
    try:
        response = model.generate_content(prompt)
        return parse_ai_batch_response(response.text, comment_batch)
    except Exception:
        # CHANGE: If the API call itself fails, classify the whole batch as Neutral.
        return [{'Original Comment': c, 'Sentiment': 'Neutral', 'Explanation': 'Unanalyzable content (API error).'} for c in comment_batch]

def run_batch_analysis(api_key, comments):
    """Manages concurrent execution of BATCH comment analysis."""
    results = []
    batch_size = 50
    comment_batches = [comments[i:i + batch_size] for i in range(0, len(comments), batch_size)]
    with st.spinner(f"Analyzing {len(comments)} comments in {len(comment_batches)} batches..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_batch = {executor.submit(analyze_comment_batch_cached, api_key, batch): batch for batch in comment_batches}
            for future in concurrent.futures.as_completed(future_to_batch):
                results.extend(future.result())
    return pd.DataFrame(results)

def load_comments_from_source(uploaded_file, gsheets_link, text_input):
    """Intelligently loads and validates comments from the provided source."""
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, header=None)
            return df.iloc[:, 0].dropna().astype(str).tolist()
        except Exception as e: st.error(f"Error reading Excel file: {e}"); return []
    if gsheets_link:
        pattern = re.compile(r'^https:\/\/docs\.google\.com\/spreadsheets\/d\/[a-zA-Z0-9_-]+(\/edit.*)?$')
        if not pattern.match(gsheets_link):
            st.error("Invalid URL. Please paste a valid Google Sheets sharing link."); return []
        try:
            url_csv = gsheets_link.replace('/edit?usp=sharing', '/export?format=csv').split('/edit')[0] + '/export?format=csv'
            df = pd.read_csv(url_csv, header=None, engine='python', on_bad_lines='skip')
            return df.iloc[:, 0].dropna().astype(str).tolist()
        except Exception as e:
            st.error(f"Error reading Google Sheets. Ensure the link is public ('Anyone with the link'). Error: {e}"); return []
    if text_input:
        return [line.strip() for line in text_input.split('\n') if line.strip()]
    return []
# Esta es una nueva "receta" para tu dibujante
# Le dice cómo sacar los colores principales de una imagen
def extract_colors_from_image(image_file, num_colors=5):
    """
    Saca los colores principales de una foto.
    Devuelve una lista de los códigos de esos colores.
    """
    img = Image.open(image_file).convert("RGB")
    img = img.resize((100, 100)) # Hacemos la imagen pequeña para que sea más rápido
    pixels = np.array(img).reshape(-1, 3) # Convertimos la imagen en una lista de puntos de color

    # Usamos una herramienta inteligente para agrupar colores parecidos
    kmeans = MiniBatchKMeans(n_clusters=num_colors, n_init=10, random_state=42)
    kmeans.fit(pixels)
    dominant_colors_rgb = kmeans.cluster_centers_.astype(int)

    # Convertimos los colores a un formato que tu dibujante entienda (código hexadecimal)
    hex_colors = ['#%02x%02x%02x' % (r, g, b) for r, g, b in dominant_colors_rgb]
    return hex_colors

def generate_visuals(df, sentiment_colors, wordcloud_color_func=None):
    """Generates all visual elements for the dashboard."""
    visuals = {};
    if df.empty: return visuals
    
    # Clean up sentiment labels, just in case
    df['Sentiment'] = df['Sentiment'].str.strip().str.replace(r'[\[\]]', '', regex=True)
    
    # --- Sentiment Chart ---
    # CHANGE: Removed 'Error' from the map and order.
    sentiment_counts = df['Sentiment'].value_counts()
    # Usa los colores que tú elijas para cada sentimiento
    color_map_sentiment = {
        'Positive': sentiment_colors.get('Positive', '#2ca02c'), # Si no hay color, usa el verde por defecto
        'Negative': sentiment_colors.get('Negative', '#d62728'), # Si no hay color, usa el rojo por defecto
        'Neutral': sentiment_colors.get('Neutral', '#ff7f0e')   # Si no hay color, usa el naranja por defecto
    }
    plot_order = [s for s in ['Positive', 'Negative', 'Neutral'] if s in sentiment_counts.index]
    
    fig_sent, ax_sent = plt.subplots(figsize=(8, 5)) # Un tamaño mejor para el gráfico
    if not sentiment_counts.loc[plot_order].empty:
        # Asegúrate de que solo se usen los colores de los sentimientos que existen
        colors_to_plot = [color_map_sentiment.get(s, '#cccccc') for s in plot_order]
        sentiment_counts.loc[plot_order].plot(kind='bar', ax=ax_sent, color=colors_to_plot)
        ax_sent.set_ylabel('Number of Comments')
        ax_sent.set_xticklabels(ax_sent.get_xticklabels(), rotation=0)
        ax_sent.set_title('Sentiment Distribution', fontsize=16) # Ponemos un título al gráfico
    plot_order = [s for s in ['Positive', 'Negative', 'Neutral'] if s in sentiment_counts.index]
    
    fig_sent, ax_sent = plt.subplots();
    if not sentiment_counts.loc[plot_order].empty:
        sentiment_counts.loc[plot_order].plot(kind='bar', ax=ax_sent, color=[color_map.get(s) for s in plot_order])
        ax_sent.set_ylabel('Number of Comments'); ax_sent.set_xticklabels(ax_sent.get_xticklabels(), rotation=0)
        total = len(df)
        for p in ax_sent.patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            ax_sent.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    visuals['sentiment_chart'] = fig_sent
    
    # --- Word Cloud ---
    stopwords_es = ['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'mas', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'ha', 'me', 'si', 'porque', 'esta', 'cuando', 'muy', 'sin', 'sobre', 'también', 'fue', 'hasta', 'hay', 'mi', 'eso', 'todo', 'está', 'son', 'qué', 'pero', 'eso', 'te', 'estar', 'así', 'hacer', 'tiene', 'tienes', 'ser', 'eres', 'soy', 'es']
    text_for_cloud = ' '.join(df['Original Comment'].dropna())
    text_no_emojis = ''.join(c for c in text_for_cloud if c not in emoji.EMOJI_DATA)
    if text_no_emojis.strip():
        wc = WordCloud(width=800, height=400, background_color='white',
                       stopwords=set(stopwords_es), collocations=False,
                       color_func=wordcloud_color_func # ¡Aquí le decimos que use la función de color que elegiste!
                      ).generate(text_for_cloud)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5)) # Un tamaño mejor para la nube de palabras
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        ax_wc.set_title('Word Cloud', fontsize=16) # Ponemos un título a la nube de palabras
        fig_wc, ax_wc = plt.subplots(); ax_wc.imshow(wc, interpolation='bilinear'); ax_wc.axis('off')
        visuals['word_cloud'] = fig_wc
        
    # --- Emoji Ranking ---
    all_emojis = [c for c in ''.join(df['Original Comment'].dropna()) if c in emoji.EMOJI_DATA]
    if all_emojis:
        visuals['emoji_ranking'] = Counter(all_emojis).most_common(5)
        
    return visuals


# ==========================================================================
# APP LAYOUT
# ==========================================================================
if "analysis_df" not in st.session_state: st.session_state.analysis_df = None
if "text_input_val" not in st.session_state: st.session_state.text_input_val = ""
if "chat_history" not in st.session_state: st.session_state.chat_history = []

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("FATAL ERROR: Your Google AI API Key is not configured in Streamlit Secrets."); st.stop()

# --- Sidebar ---
with st.sidebar:
    st.title("Controls")
    # --- Aquí empezamos con los controles para personalizar los colores ---
    st.markdown("---") # Pone una línea separadora bonita
    st.subheader("Personalizar Colores") # Un título para esta sección
    
    # --- Opciones para los colores del Gráfico de Sentimiento ---
    st.markdown("### Colores del Gráfico de Sentimiento")
    sentiment_palette_option = st.selectbox(
        "Elige cómo quieres los colores del gráfico:",
        ["Por Defecto", "Colores Personalizados", "Colores de una Imagen"], # Opciones para el usuario
        key="sentiment_palette_option" # Un nombre único para este control
    )

    sentiment_colors_to_use = {} # Aquí guardaremos los colores que el usuario elija
    if sentiment_palette_option == "Por Defecto":
        sentiment_colors_to_use = {
            'Positive': '#2ca02c', # Verde por defecto
            'Negative': '#d62728', # Rojo por defecto
            'Neutral': '#ff7f0e'   # Naranja por defecto
        }
    elif sentiment_palette_option == "Colores Personalizados":
        st.write("O define tus propios colores:")
        sentiment_colors_to_use['Positive'] = st.color_picker('Color para POSITIVO', '#2ca02c', key="pos_color")
        sentiment_colors_to_use['Negative'] = st.color_picker('Color para NEGATIVO', '#d62728', key="neg_color")
        sentiment_colors_to_use['Neutral'] = st.color_picker('Color para NEUTRAL', '#ff7f0e', key="neu_color")
    
    # --- Opciones para los colores de la Nube de Palabras ---
    st.markdown("### Colores de la Nube de Palabras")
    wordcloud_color_option = st.selectbox(
        "Elige cómo quieres los colores de la nube de palabras:",
        ["Por Defecto (Aleatorio)", "Azules", "Verdes", "Viridis", "Plasma", "Colores de una Imagen"],
        key="wordcloud_palette_option"
    )

    image_for_colors = None # Aquí guardaremos la imagen si el usuario sube una
    extracted_colors = [] # Aquí guardaremos los colores que saquemos de la imagen

    # Si el usuario elige sacar colores de una imagen para cualquier gráfico
    if sentiment_palette_option == "Colores de una Imagen" or wordcloud_color_option == "Colores de una Imagen":
        image_for_colors = st.file_uploader("Upload an image to extract colors:", type=["png", "jpg", "jpeg"], key="color_image_uploader")
    
    # Aquí es donde guardaremos los colores extraídos de la imagen en la sesión de Streamlit
    if "extracted_image_colors" not in st.session_state:
        st.session_state.extracted_image_colors = []

    if image_for_colors:
        try:
            # Solo procesa la imagen si es nueva o no la hemos procesado antes en esta sesión
            if image_for_colors != st.session_state.get("last_uploaded_image_hash"):
                st.session_state.extracted_image_colors = extract_colors_from_image(image_for_colors, num_colors=10)
                st.session_state.last_uploaded_image_hash = image_for_colors # Guarda un "hash" para saber que ya la procesamos
                st.info(f"Se sacaron {len(st.session_state.extracted_image_colors)} colores de la imagen.")
            
            # Asignar los colores para el gráfico de sentimiento si se eligió "Colores de una Imagen"
            if sentiment_palette_option == "Colores de una Imagen":
                if len(st.session_state.extracted_image_colors) >= 3:
                    st.session_state.sentiment_colors_from_image = {
                        'Positive': st.session_state.extracted_image_colors[0],
                        'Neutral': st.session_state.extracted_image_colors[1],
                        'Negative': st.session_state.extracted_image_colors[2]
                    }
                else:
                    st.warning("No hay suficientes colores en la imagen para el gráfico de sentimiento. Usando colores por defecto.")
                    st.session_state.sentiment_colors_from_image = { # Colores de respaldo
                        'Positive': '#2ca02c', 'Negative': '#d62728', 'Neutral': '#ff7f0e'
                    }
                
        except Exception as e:
            st.error(f"Error al procesar la imagen para sacar colores: {e}")
            st.session_state.extracted_image_colors = []
            if "sentiment_colors_from_image" in st.session_state: del st.session_state.sentiment_colors_from_image # Limpiar
    else:
        st.warning("Por favor, sube una imagen para usar la opción 'Colores de una Imagen'.")
        st.session_state.extracted_image_colors = []
        if "sentiment_colors_from_image" in st.session_state: del st.session_state.sentiment_colors_from_image # Limpiar

    wordcloud_color_func_to_use = None # Esta será la función para la nube de palabras

    if wordcloud_color_option == "Azules":
        wordcloud_color_func_to_use = plt.cm.Blues # Una paleta de azules de Matplotlib
    elif wordcloud_color_option == "Verdes":
        wordcloud_color_func_to_use = plt.cm.Greens # Una paleta de verdes
    elif wordcloud_color_option == "Viridis":
        wordcloud_color_func_to_use = plt.cm.viridis
    elif wordcloud_color_option == "Plasma":
        wordcloud_color_func_to_use = plt.cm.plasma
    elif wordcloud_color_option == "Colores de una Imagen" and image_for_colors and extracted_colors:
        # Si elegimos colores de la imagen y hay colores
        from wordcloud import get_single_color_func
        if extracted_colors:
            # Creamos una función que elija un color aleatorio de los que sacamos de la imagen
            wordcloud_color_func_to_use = lambda word, font_size, position, orientation, random_state, **kwargs: \
                                          extracted_colors[random_state.randint(0, len(extracted_colors) -1)]
        else:
             wordcloud_color_func_to_use = None # Si no hay colores, volvemos al por defecto
    else: # Si el usuario elige "Por Defecto (Aleatorio)" o si algo falló
        wordcloud_color_func_to_use = None # La nube de palabras usará sus colores aleatorios por defecto

    # --- Fin de los controles de personalización de color ---
    st.markdown("---") # Otra línea separadora
    if st.session_state.analysis_df is not None:
        if st.button("Start New Analysis"):
            st.session_state.analysis_df = None; st.session_state.text_input_val = ""; st.session_state.chat_history = []; st.rerun()
    else:
        st.info("Analyze data to enable controls.")

# --- Main Content ---
if st.session_state.analysis_df is None:
    # --- Data Input View ---
    st.header("Comment Analysis Platform")
    st.markdown("Provide your data through one of the methods below to start the analysis.")
    input_container = st.container(border=True)
    with input_container:
        tab1, tab2, tab3 = st.tabs(["Paste Comments", "Upload Excel", "Google Sheets"])
        with tab1:
            st.session_state.text_input_val = st.text_area("Paste comments here:", height=250, value=st.session_state.text_input_val)
            if st.button("Clear"): st.session_state.text_input_val = ""; st.rerun()
        with tab2:
            uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'], label_visibility="collapsed")
        with tab3:
            gsheets_link = st.text_input("Paste Google Sheets link", label_visibility="collapsed")
    st.info("Note: For Excel and Google Sheets, the app will automatically analyze the first column.", icon="💡")

    if st.button("✨ Analyze Now!"):
        comments = load_comments_from_source(uploaded_file, gsheets_link, st.session_state.text_input_val)
        if comments:
            st.session_state.analysis_df = run_batch_analysis(api_key, comments)
            st.rerun()
        else:
            st.warning("Please provide comments to analyze.")
else:

        # --- Resultados del Dashboard ---
        df = st.session_state.analysis_df
        st.header("Analysis Dashboard")
        
        with st.expander("💬 Open IA Chat to ask about these results"):
            # ... (Tu lógica del chat existente, no la modifiques aquí) ...
            pass

        # --- DETERMINACIÓN FINAL DE COLORES PARA LOS GRÁFICOS ---
        # Estos son los valores que usará generate_visuals
        sentiment_colors_to_use = {
            'Positive': '#2ca02c', # Por defecto
            'Negative': '#d62728', # Por defecto
            'Neutral': '#ff7f0e'   # Por defecto
        }
        wordcloud_color_func_to_use = None # Por defecto, aleatorio

        # Obtener las opciones que el usuario eligió en la barra lateral
        # Usamos .get() con un valor por defecto para evitar errores si aún no se han seleccionado
        sentiment_palette_option = st.session_state.get("sentiment_palette_option", "Por Defecto")
        wordcloud_color_option = st.session_state.get("wordcloud_palette_option", "Por Defecto (Aleatorio)")

        # Lógica para el Gráfico de Sentimiento
        if sentiment_palette_option == "Colores Personalizados":
            sentiment_colors_to_use['Positive'] = st.session_state.get('pos_color', '#2ca02c')
            sentiment_colors_to_use['Negative'] = st.session_state.get('neg_color', '#d62728')
            sentiment_colors_to_use['Neutral'] = st.session_state.get('neu_color', '#ff7f0e')
        elif sentiment_palette_option == "Colores de una Imagen":
            # Usar los colores que se guardaron en la sesión desde el sidebar
            if "sentiment_colors_from_image" in st.session_state:
                sentiment_colors_to_use = st.session_state.sentiment_colors_from_image
            else: # Fallback si por alguna razón no se guardaron
                st.warning("Colores de imagen no disponibles para el gráfico de sentimiento. Usando valores por defecto.")


        # Lógica para la Nube de Palabras
        if wordcloud_color_option == "Azules":
            wordcloud_color_func_to_use = plt.cm.Blues
        elif wordcloud_color_option == "Verdes":
            wordcloud_color_func_to_use = plt.cm.Greens
        elif wordcloud_color_option == "Viridis":
            wordcloud_color_func_to_use = plt.cm.viridis
        elif wordcloud_color_option == "Plasma":
            wordcloud_color_func_to_use = plt.cm.plasma
        elif wordcloud_color_option == "Colores de una Imagen":
            # Usar los colores extraídos y guardados en la sesión
            extracted_colors_for_wc = st.session_state.get("extracted_image_colors", [])
            if extracted_colors_for_wc:
                # Asegúrate que esta importación de `get_single_color_func` esté al principio del archivo con las otras importaciones de `wordcloud`
                wordcloud_color_func_to_use = lambda word, font_size, position, orientation, random_state, **kwargs: \
                                              extracted_colors_for_wc[random_state.randint(0, len(extracted_colors_for_wc) -1)]
            else:
                st.warning("No hay colores de imagen para la nube de palabras. Usando valores por defecto.")
                wordcloud_color_func_to_use = None
        # Si es "Por Defecto (Aleatorio)", wordcloud_color_func_to_use ya es None, lo cual es correcto.
        
        # --- FIN DE DETERMINACIÓN FINAL DE COLORES ---

        # Ahora sí, genera los visuales con las variables ya definidas y actualizadas
        visuals = generate_visuals(df, sentiment_colors_to_use, wordcloud_color_func_to_use)
        
        # ... (El resto de tu código para mostrar los gráficos, no lo modifiques) ...
# ==========================================================================
# COMMENT ANALYSIS DASHBOARD - Final Premium UI Version
# Top Emojis with wide separation + colorful and airy Word Cloud
# ==========================================================================
import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import google.generativeai as genai
import emoji
from collections import Counter
import concurrent.futures
import json
from matplotlib.colors import ListedColormap

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Comment Analysis Dashboard", layout="wide")

# --- STYLES ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* === GLOBAL === */
.stApp {
    background-color: #FAFAFA;
    font-family: 'Inter', sans-serif;
}

/* === CARD CONTAINERS === */
[data-testid="stHorizontalBlock"] > div {
    background-color: #FFFFFF;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

/* === HEADERS === */
h1, h2, h3, h4 {
    color: #111111;
    font-weight: 600;
}

/* === BUTTONS === */
.stButton>button {
    background: linear-gradient(90deg, #8A2BE2 0%, #6C63FF 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg, #7B1FA2 0%, #5A55FF 100%);
}

/* === PANEL CARDS === */
.st-emotion-cache-1r4qj8v, .st-emotion-cache-0 {
    background-color: #FFFFFF;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* === EMOJI BADGES (con separación amplia tipo ejemplo) === */
.emoji-card {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    background-color: #ffffff;
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    font-size: 2rem;
    gap: 180px; /* Separación amplia entre emoji y etiqueta */
}

.emoji-count {
    color: white;
    font-weight: 600;
    padding: 6px 16px;
    border-radius: 10px;
    font-size: 1rem;
}

.pink {background-color: #E11D74;}
.orange {background-color: #FF6B00;}
.yellow {background-color: #FFC107;}
.purple {background-color: #9C27B0;}
</style>
""", unsafe_allow_html=True)

# --- CORE FUNCTIONS ---
def parse_ai_batch_response(response_text, original_batch):
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
    except Exception:
        return [{'Original Comment': c, 'Sentiment': 'Neutral', 'Explanation': 'Unanalyzable content.'} for c in original_batch]


@st.cache_data(ttl="24h")
def analyze_comment_batch_cached(_api_key, comment_batch):
    genai.configure(api_key=_api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    comments_str = "\n".join([f'{i+1}. "{comment}"' for i, comment in enumerate(comment_batch)])
    prompt = f"""
    Act as a sentiment analysis API. Analyze EACH of the following customer comments.
    Respond with a valid JSON array of objects:
    [{{"Sentiment": "...", "Explanation": "..."}}]
    Sentiment must be one of Positive, Negative, or Neutral.

    COMMENTS:
    {comments_str}
    """
    try:
        response = model.generate_content(prompt)
        return parse_ai_batch_response(response.text, comment_batch)
    except Exception:
        return [{'Original Comment': c, 'Sentiment': 'Neutral', 'Explanation': 'API error.'} for c in comment_batch]


def run_batch_analysis(api_key, comments):
    results = []
    batch_size = 50
    comment_batches = [comments[i:i + batch_size] for i in range(0, len(comments), batch_size)]
    with st.spinner(f"Analyzing {len(comments)} comments in {len(comment_batches)} batches..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(analyze_comment_batch_cached, api_key, batch) for batch in comment_batches]
            for f in concurrent.futures.as_completed(futures):
                results.extend(f.result())
    return pd.DataFrame(results)


def load_comments(uploaded_file, gsheets_link, text_input):
    if uploaded_file:
        df = pd.read_excel(uploaded_file, header=None)
        return df.iloc[:, 0].dropna().astype(str).tolist()
    if gsheets_link:
        url_csv = gsheets_link.replace('/edit?usp=sharing', '/export?format=csv').split('/edit')[0] + '/export?format=csv'
        df = pd.read_csv(url_csv, header=None, engine='python', on_bad_lines='skip')
        return df.iloc[:, 0].dropna().astype(str).tolist()
    if text_input:
        return [line.strip() for line in text_input.split('\n') if line.strip()]
    return []


def generate_visuals(df):
    visuals = {}
    if df.empty: return visuals

    # --- Sentiment Chart ---
    df['Sentiment'] = df['Sentiment'].str.strip()
    sentiment_counts = df['Sentiment'].value_counts()
    color_map = {'Positive': '#2ecc71', 'Neutral': '#f1c40f', 'Negative': '#e74c3c'}

    fig_sent, ax = plt.subplots()
    fig_sent.patch.set_alpha(0)
    ax.set_facecolor('none')
    sentiment_counts.plot(kind='bar', ax=ax, color=[color_map.get(x, '#ccc') for x in sentiment_counts.index])
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelrotation=0, labelsize=11)
    ax.tick_params(axis='y', left=False, labelleft=False)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    total = sentiment_counts.sum()
    for p in ax.patches:
        percent = f"{100*p.get_height()/total:.0f}%"
        ax.annotate(percent, (p.get_x() + p.get_width()/2, p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='black')
    visuals['sentiment_chart'] = fig_sent

    # --- Word Cloud (colorido, aireado, limpio) ---
    text = ' '.join(df['Original Comment'].dropna())
    stopwords_es = set(list(STOPWORDS) + [
        "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las",
        "por", "un", "para", "con", "no", "una", "su", "al", "lo", "como",
        "más", "sus", "le", "ya", "o", "este", "ha", "me", "si", "mi", "yo",
        "porque", "esta", "muy", "sin", "sobre", "también", "fue", "esa",
        "son", "está", "ni", "solo", "puede", "uno", "delos"
    ])

    colorful_map = ListedColormap([
        "#FF5722", "#4CAF50", "#2196F3", "#FFC107", "#9C27B0", "#E91E63", "#00BCD4"
    ])

    wc = WordCloud(
        width=900, height=450,
        background_color='white',
        colormap=colorful_map,
        prefer_horizontal=0.9,
        collocations=False,
        stopwords=stopwords_es,
        max_words=60,
        max_font_size=110,
        min_font_size=15,
        margin=8,  # más aire entre palabras
        relative_scaling=0.3
    ).generate(text)

    fig_wc, ax_wc = plt.subplots()
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis('off')
    visuals['word_cloud'] = fig_wc

    # --- Top Emojis ---
    all_emojis = [c for c in ''.join(df['Original Comment'].dropna()) if c in emoji.EMOJI_DATA]
    if all_emojis:
        visuals['emoji_ranking'] = Counter(all_emojis).most_common(5)
    return visuals


# --- APP LAYOUT ---
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None

st.title("Sentiment Analysis Dashboard")

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("⚠️ API key missing in Streamlit Secrets.")
    st.stop()

if st.session_state.analysis_df is None:
    st.header("Load Comments")
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    gsheets_link = st.text_input("Google Sheets Link")
    text_input = st.text_area("Or paste comments manually:", height=200)
    if st.button("✨ Analyze Now!"):
        comments = load_comments(uploaded_file, gsheets_link, text_input)
        if comments:
            st.session_state.analysis_df = run_batch_analysis(api_key, comments)
            st.rerun()
        else:
            st.warning("Please provide comments to analyze.")
else:
    df = st.session_state.analysis_df
    visuals = generate_visuals(df)

    col1, col2 = st.columns([2, 1])
    with col1:
        if 'sentiment_chart' in visuals:
            with st.container():
                st.pyplot(visuals['sentiment_chart'])
    with col2:
        if 'emoji_ranking' in visuals:
            with st.container():
                st.subheader("Top Emojis")
                color_classes = ["pink", "orange", "yellow", "purple", "pink"]
                for idx, (emoji_char, count) in enumerate(visuals['emoji_ranking']):
                    color_class = color_classes[idx % len(color_classes)]
                    st.markdown(f"""
                    <div class="emoji-card">
                        <div>{emoji_char}<span class="emoji-count {color_class}">x{count}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

    if 'word_cloud' in visuals:
        with st.container():
            st.subheader("Word Cloud")
            st.pyplot(visuals['word_cloud'])

    with st.container():
        st.subheader("Detailed Data")
        st.dataframe(df)








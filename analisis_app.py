# ==========================================================================
# COMMENT ANALYSIS DASHBOARD - Final Premium UX/UI + IA Chat Restored
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

# ----------------------------- PAGE CONFIG --------------------------------
st.set_page_config(page_title="Comment Analysis Dashboard", layout="wide")

# ------------------------------ STYLES -------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* GLOBAL */
.stApp { background-color: #FAFAFA; font-family: 'Inter', sans-serif; }

/* CARDS */
[data-testid="stHorizontalBlock"] > div {
  background-color:#FFFFFF;border-radius:18px;padding:24px;
  box-shadow:0 4px 10px rgba(0,0,0,0.05);margin-bottom:20px;
}

/* HEADERS */
h1,h2,h3,h4{color:#111111;font-weight:600}

/* BUTTONS */
.stButton>button{
  background:linear-gradient(90deg,#8A2BE2 0%,#6C63FF 100%);
  color:#fff;border:none;border-radius:10px;padding:10px 24px;
  font-weight:600;transition:.2s}
.stButton>button:hover{transform:scale(1.02);
  background:linear-gradient(90deg,#7B1FA2 0%,#5A55FF 100%)}

/* GENERIC CONTAINERS */
.st-emotion-cache-1r4qj8v,.st-emotion-cache-0{
  background-color:#FFFFFF;border-radius:18px;padding:24px;
  box-shadow:0 4px 12px rgba(0,0,0,0.05)}

/* TOP EMOJIS (como imagen “bien”) */
.emoji-card{
  display:flex;align-items:center;justify-content:space-between;
  background:#fff;border-radius:14px;padding:18px 26px;margin-bottom:16px;
  box-shadow:0 2px 8px rgba(0,0,0,0.06);transition:.2s}
.emoji-card:hover{transform:translateY(-2px);
  box-shadow:0 4px 12px rgba(0,0,0,0.10)}
.emoji-icon{font-size:2.3rem;line-height:1;flex-shrink:0}
.emoji-count{
  color:#fff;font-weight:600;font-size:1.05rem;padding:6px 18px;border-radius:10px;
  min-width:58px;text-align:center;margin-left:20px}
.pink{background:#E11D74}.orange{background:#FF6B00}
.yellow{background:#FFC107}.purple{background:#9C27B0}.red{background:#E91E63}

/* CHAT IA */
.chat-card{background:#fff;border:1px solid #eee;border-radius:14px;padding:14px 16px;margin:8px 0}
.role-user{background:#F5F7FF}.role-assistant{background:#F9F9F9}
</style>
""", unsafe_allow_html=True)

# --------------------------- CORE FUNCTIONS --------------------------------
def parse_ai_batch_response(response_text, original_batch):
    try:
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        else:
            json_str = response_text
        analyses = json.loads(json_str)
        for i, a in enumerate(analyses):
            if i < len(original_batch):
                a['Original Comment'] = original_batch[i]
        return analyses
    except Exception:
        return [{'Original Comment': c, 'Sentiment': 'Neutral',
                 'Explanation': 'Unanalyzable content.'} for c in original_batch]

@st.cache_data(ttl="24h")
def analyze_comment_batch_cached(_api_key, comment_batch):
    genai.configure(api_key=_api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    comments_str = "\n".join([f'{i+1}. "{c}"' for i, c in enumerate(comment_batch)])
    prompt = f"""
    Act as a sentiment analysis API. Analyze EACH of the following customer comments.
    Respond with a valid JSON array of objects:
    [{{"Sentiment":"...", "Explanation":"..."}}]
    Sentiment must be Positive, Negative, or Neutral.
    COMMENTS:
    {comments_str}
    """
    try:
        resp = model.generate_content(prompt)
        return parse_ai_batch_response(resp.text, comment_batch)
    except Exception:
        return [{'Original Comment': c, 'Sentiment': 'Neutral',
                 'Explanation': 'API error.'} for c in comment_batch]

def run_batch_analysis(api_key, comments):
    results, batch_size = [], 50
    batches = [comments[i:i+batch_size] for i in range(0, len(comments), batch_size)]
    with st.spinner(f"Analyzing {len(comments)} comments..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            futs = [ex.submit(analyze_comment_batch_cached, api_key, b) for b in batches]
            for f in concurrent.futures.as_completed(futs):
                results.extend(f.result())
    return pd.DataFrame(results)

def load_comments(uploaded_file, gsheets_link, text_input):
    if uploaded_file:
        df = pd.read_excel(uploaded_file, header=None)
        return df.iloc[:,0].dropna().astype(str).tolist()
    if gsheets_link:
        url_csv = gsheets_link.replace('/edit?usp=sharing','/export?format=csv').split('/edit')[0]+'/export?format=csv'
        df = pd.read_csv(url_csv, header=None, engine='python', on_bad_lines='skip')
        return df.iloc[:,0].dropna().astype(str).tolist()
    if text_input:
        return [ln.strip() for ln in text_input.split('\n') if ln.strip()]
    return []

def generate_visuals(df):
    visuals = {}
    if df.empty: return visuals

    # --- Sentiment bars (vertical, etiquetas horizontales, sin números laterales)
    df['Sentiment'] = df['Sentiment'].str.strip()
    counts = df['Sentiment'].value_counts()
    color_map = {'Positive':'#2ecc71','Neutral':'#f1c40f','Negative':'#e74c3c'}

    fig, ax = plt.subplots()
    fig.patch.set_alpha(0); ax.set_facecolor('none')
    counts.plot(kind='bar', ax=ax, color=[color_map.get(x,'#ccc') for x in counts.index])
    ax.set_ylabel(''); ax.set_xlabel('')
    ax.tick_params(axis='x', labelrotation=0, labelsize=11)
    ax.tick_params(axis='y', left=False, labelleft=False)
    for s in ['top','right','left','bottom']: ax.spines[s].set_visible(False)
    total = counts.sum()
    for p in ax.patches:
        pct = f"{100*p.get_height()/total:.0f}%"
        ax.annotate(pct, (p.get_x()+p.get_width()/2, p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black')
    visuals['sentiment_chart'] = fig

    # --- Word Cloud (claro, aireado)
    text = ' '.join(df['Original Comment'].dropna())
    stopwords_es = set(list(STOPWORDS) + [
        "de","la","que","el","en","y","a","los","del","se","las","por","un","para",
        "con","no","una","su","al","lo","como","más","sus","le","ya","o","este",
        "ha","me","si","mi","yo","porque","esta","muy","sin","sobre","también",
        "fue","esa","son","está","ni","solo","puede","uno","delos"
    ])
    colorful_map = ListedColormap(["#FF5722","#4CAF50","#2196F3","#FFC107",
                                   "#9C27B0","#E91E63","#00BCD4"])
    wc = WordCloud(
        width=900, height=450, background_color='white', colormap=colorful_map,
        prefer_horizontal=0.9, collocations=False, stopwords=stopwords_es,
        max_words=60, max_font_size=110, min_font_size=15, margin=10, relative_scaling=0.3
    ).generate(text)
    fig_wc, ax_wc = plt.subplots(); ax_wc.imshow(wc, interpolation='bilinear'); ax_wc.axis('off')
    visuals['word_cloud'] = fig_wc

    # --- Top Emojis
    all_emojis = [c for c in ''.join(df['Original Comment'].dropna()) if c in emoji.EMOJI_DATA]
    if all_emojis: visuals['emoji_ranking'] = Counter(all_emojis).most_common(5)
    return visuals

# ----------------------------- STATE ---------------------------------------
if "analysis_df" not in st.session_state: st.session_state.analysis_df = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []  # <-- RESTORED

# ------------------------------ HEADER -------------------------------------
st.title("Sentiment Analysis Dashboard")

# API key
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("⚠️ Google API key missing in Streamlit Secrets.")
    st.stop()

# ------------------------------- INPUT -------------------------------------
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

    # --------- IA CHAT (RESTORED) ----------
    with st.expander("💬 IA Chat sobre estos resultados", expanded=False):
        # Muestra historial
        for msg in st.session_state.chat_history:
            role_cls = "role-user" if msg["role"]=="user" else "role-assistant"
            with st.container():
                st.markdown(f'<div class="chat-card {role_cls}"><b>{msg["role"].title()}:</b><br>{msg["content"]}</div>',
                            unsafe_allow_html=True)

        user_q = st.chat_input("Haz una pregunta sobre el análisis...")
        if user_q:
            st.session_state.chat_history.append({"role":"user","content":user_q})
            # Construir contexto compacto para el LLM
            # 1) Resumen de conteos de sentimiento
            counts = df['Sentiment'].value_counts().to_dict()
            # 2) Muestra de filas (máx 30)
            sample_rows = df.head(30).to_string(index=False)
            context = f"Sentiment counts: {counts}\nSample rows (up to 30):\n{sample_rows}"
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                prompt = (
                    "You are an expert business analyst. Answer concisely using only the given data.\n"
                    f"--- DATA ---\n{context}\n--- END DATA ---\n"
                    f"QUESTION: {user_q}"
                )
                resp = model.generate_content(prompt)
                answer = resp.text.strip()
            except Exception as e:
                answer = f"No pude completar la respuesta ({e})."
            st.session_state.chat_history.append({"role":"assistant","content":answer})
            st.rerun()
    # --------- END CHAT ----------

    # -------------------------- MAIN GRID --------------------------
    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        if 'sentiment_chart' in visuals:
            with st.container():
                st.pyplot(visuals['sentiment_chart'])
    with col2:
        if 'emoji_ranking' in visuals:
            with st.container():
                st.subheader("Top Emojis")
                color_classes = ["pink","orange","yellow","purple","red"]
                for idx, (emoji_char, count) in enumerate(visuals['emoji_ranking']):
                    color_class = color_classes[idx % len(color_classes)]
                    st.markdown(f"""
                    <div class="emoji-card">
                        <div class="emoji-icon">{emoji_char}</div>
                        <div class="emoji-count {color_class}">x{count}</div>
                    </div>
                    """, unsafe_allow_html=True)

    if 'word_cloud' in visuals:
        with st.container():
            st.subheader("Word Cloud")
            st.pyplot(visuals['word_cloud'])

    with st.container():
        st.subheader("Detailed Data")
        st.dataframe(df)










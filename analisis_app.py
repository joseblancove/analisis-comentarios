# --------------------------------------------------------------------------
# COMMENT ANALYSIS PLATFORM V13.0 - The Final Client-Ready Edition
# All inputs are processed; all errors are classified as Neutral.
# --------------------------------------------------------------------------
import streamlit as st
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
    model = genai.GenerativeModel('gemini-2.5-flash')
    
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

def generate_visuals(df):
    """Generates all visual elements for the dashboard."""
    visuals = {};
    if df.empty: return visuals
    
    # Clean up sentiment labels, just in case
    df['Sentiment'] = df['Sentiment'].str.strip().str.replace(r'[\[\]]', '', regex=True)
    
    # --- Sentiment Chart ---
    # CHANGE: Removed 'Error' from the map and order.
    sentiment_counts = df['Sentiment'].value_counts()
    color_map = {'Positive': '#2ca02c', 'Negative': '#d62728', 'Neutral': '#ff7f0e'}
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
        wc = WordCloud(width=800, height=400, background_color='white', stopwords=set(stopwords_es), collocations=False).generate(text_for_cloud)
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
    # --- Results Dashboard View ---
    df = st.session_state.analysis_df
    st.header("Analysis Dashboard")
    
    with st.expander("💬 Open IA Chat to ask about these results"):
        # ... Chat logic remains the same
        # ...
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]): st.markdown(message["content"])
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.spinner("Thinking..."):
                model = genai.GenerativeModel('gemini-2.5-flash')
                context_for_ia = f"Dataframe:\n{st.session_state.analysis_df.to_string()}"
                full_prompt = f"You are an expert business analyst. Based on the following data analysis, answer the user's question concisely.\n--- DATA ---\n{context_for_ia}\n--- END DATA ---\nQUESTION: {prompt}"
                response = model.generate_content(full_prompt)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
            st.rerun()
            
    visuals = generate_visuals(df)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if 'sentiment_chart' in visuals:
            with st.container(border=True): st.pyplot(visuals['sentiment_chart'])
    with col2:
        if 'emoji_ranking' in visuals:
            with st.container(border=True):
                st.subheader("Top Emojis")
                for emoji_char, count in visuals['emoji_ranking']:
                    st.markdown(f"### {emoji_char} &nbsp;&nbsp;`x{count}`")

    if 'word_cloud' in visuals:
        with st.container(border=True):
            st.subheader("Word Cloud")
            st.pyplot(visuals['word_cloud'])

    with st.container(border=True):
        st.subheader("Detailed Data")
        cols = df.columns.tolist()
        if 'Original Comment' in cols:
            cols.insert(0, cols.pop(cols.index('Original Comment')))
            st.dataframe(df[cols])
        else:
            st.dataframe(df)
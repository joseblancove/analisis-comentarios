# --------------------------------------------------------------------------
# COMMENT ANALYSIS PLATFORM V7.4 - Intelligent Cache Edition
# Final professional architecture with smart caching and robust UI.
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

# --- Page Configuration ---
st.set_page_config(page_title="Comment Analysis", layout="wide")

# --- STYLE INJECTION ---
st.markdown("""
<style>
    .stApp { background-color: #F0F2F6; }
    [data-testid="stSidebar"] { background-color: #1E293B; }
    [data-testid="stSidebar"] * { color: #FFFFFF; }
    .stButton>button { background-color: #8A2BE2; color: #FFFFFF; border-radius: 8px; border: none; padding: 10px 20px; font-weight: bold; }
    .stButton>button:hover { background-color: #7B1FA2; }
    .st-emotion-cache-1r4qj8v, .st-emotion-cache-0 { background-color: #FFFFFF; border-radius: 10px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)


# --- CORE FUNCTIONS ---
# FIX: Caching is now applied to the individual worker function for robustness.
@st.cache_data(ttl="24h")
def analyze_single_comment_cached(_api_key, comment):
    """
    Worker function to analyze one comment. This is the cached part.
    The api_key is passed to ensure the cache is valid for a specific user.
    """
    # Configure GenAI within the cached function
    genai.configure(api_key=_api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    prompt = f"""
    Act as a world-class sentiment analysis expert. Your analysis must be sharp and decisive.
    Analyze the following customer comment. Acknowledge and interpret all emojis.
    Only classify as 'Neutral' if the comment is purely informational with zero emotional content.
    If there is any hint of emotion, positive or negative, classify it accordingly.
    Provide the output in this exact format:
    Sentiment: [Positive, Negative, or Neutral]
    Explanation: [A concise explanation of the sentiment, referencing specific words or emojis that justify your classification.]
    COMMENT: "{comment}"
    """
    try:
        response = model.generate_content(prompt)
        analysis = {}
        lines = response.text.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                analysis[key.strip()] = value.strip()
        analysis['Original Comment'] = comment
        return analysis
    except Exception:
        return {'Original Comment': comment, 'Sentiment': 'Error', 'Explanation': 'Failed to analyze.'}

def run_concurrent_analysis(api_key, comments, progress_bar_placeholder):
    """
    Manages the concurrent execution of comment analysis. THIS FUNCTION IS NOT CACHED.
    It calls the cached worker function.
    """
    results = []
    total_comments = len(comments)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_comment = {executor.submit(analyze_single_comment_cached, api_key, comment): comment for comment in comments}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_comment)):
            results.append(future.result())
            progress_bar_placeholder.progress((i + 1) / total_comments, text=f"Analyzing comment {i+1}/{total_comments}")
            
    progress_bar_placeholder.empty()
    return pd.DataFrame(results)

# ... Other functions (load_comments, generate_visuals) remain the same ...
def load_comments_from_source(uploaded_file, gsheets_link, text_input):
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, header=None)
            return df.iloc[:, 0].dropna().astype(str).tolist()
        except Exception as e:
            st.error(f"Error reading Excel file: {e}"); return []
    if gsheets_link:
        try:
            url_csv = gsheets_link.replace('/edit?usp=sharing', '/export?format=csv')
            df = pd.read_csv(url_csv, header=None, engine='python', on_bad_lines='skip')
            return df.iloc[:, 0].dropna().astype(str).tolist()
        except Exception as e:
            st.error(f"Error reading Google Sheets: {e}"); return []
    if text_input:
        return [line.strip() for line in text_input.split('\n') if line.strip()]
    return []

def generate_visuals(df):
    visuals = {};
    if df.empty: return visuals
    sentiment_counts = df['Sentiment'].value_counts()
    color_map = {'Positive': '#2ca02c', 'Negative': '#d62728', 'Neutral': '#ff7f0e', 'Error': '#7f7f7f'}
    plot_order = [s for s in ['Positive', 'Negative', 'Neutral', 'Error'] if s in sentiment_counts.index]
    fig_sent, ax_sent = plt.subplots(); sentiment_counts.loc[plot_order].plot(kind='bar', ax=ax_sent, color=[color_map.get(s) for s in plot_order])
    ax_sent.set_ylabel('Number of Comments'); ax_sent.set_xticklabels(ax_sent.get_xticklabels(), rotation=0)
    total = len(df)
    for p in ax_sent.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        ax_sent.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    visuals['sentiment_chart'] = fig_sent
    stopwords_es = ['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'mas', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'ha', 'me', 'si', 'porque', 'esta', 'cuando', 'muy', 'sin', 'sobre', 'también', 'fue', 'hasta', 'hay', 'mi', 'eso', 'todo', 'está', 'son', 'qué', 'pero', 'eso', 'te', 'estar', 'así', 'hacer', 'tiene', 'tienes', 'ser', 'eres', 'soy', 'es']
    text_for_cloud = ' '.join(df['Original Comment'].dropna())
    text_no_emojis = ''.join(c for c in text_for_cloud if c not in emoji.EMOJI_DATA)
    if text_no_emojis.strip():
        wc = WordCloud(width=800, height=400, background_color='white', stopwords=set(stopwords_es), collocations=False).generate(text_no_emojis)
        fig_wc, ax_wc = plt.subplots(); ax_wc.imshow(wc, interpolation='bilinear'); ax_wc.axis('off')
        visuals['word_cloud'] = fig_wc
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
if "open_chat" not in st.session_state: st.session_state.open_chat = False

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("FATAL ERROR: Your Google AI API Key is not configured in Streamlit Secrets."); st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("Controls")
    if st.session_state.analysis_df is not None:
        if st.button("💬 Open IA Chat"):
            st.session_state.open_chat = True
        if st.button("Start New Analysis"):
            st.session_state.analysis_df = None
            st.session_state.text_input_val = ""
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("Analyze data to enable controls.")

# --- CHAT MODAL ---
if st.session_state.open_chat:
    with st.dialog("IA Chat"):
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("Ask about the results..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.spinner("Thinking..."):
                model = genai.GenerativeModel('gemini-1.5-pro')
                context_for_ia = f"Dataframe:\n{st.session_state.analysis_df.to_string()}"
                full_prompt = f"You are an expert business analyst. Based on the following data analysis, answer the user's question concisely.\n--- DATA ---\n{context_for_ia}\n--- END DATA ---\nQUESTION: {prompt}"
                response = model.generate_content(full_prompt)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
            st.rerun()

# --- MAIN CONTENT ---
if st.session_state.analysis_df is None:
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
            progress_bar_placeholder = st.empty()
            st.session_state.analysis_df = run_concurrent_analysis(api_key, comments, progress_bar_placeholder)
            st.rerun()
        else:
            st.warning("Please provide comments to analyze.")
else:
    # --- RESULTS DASHBOARD VIEW ---
    df = st.session_state.analysis_df
    st.header("Analysis Dashboard")
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

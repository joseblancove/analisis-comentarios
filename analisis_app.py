# --------------------------------------------------------------------------
# COMMENT ANALYSIS PLATFORM V6.0 - The Definitive Edition
# Designed by a World-Class Developer, inspired by "famosos ADS" UI.
# --------------------------------------------------------------------------
import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import google.generativeai as genai
import time
import emoji
from collections import Counter

# --- Page Configuration ---
st.set_page_config(page_title="Comment Analysis", layout="wide")

# --- STYLE INJECTION (Professional Light Theme with Dark Sidebar) ---
st.markdown("""
<style>
    /* Main Content Area */
    .stApp {
        background-color: #F0F2F6; /* Light gray background */
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E293B; /* Dark sidebar */
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFFFFF;
    }
    /* Main Action Button */
    .stButton>button {
        background-color: #8A2BE2; /* Brand Violet */
        color: #FFFFFF;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #7B1FA2;
    }
    /* Containers/Cards in Main Area */
    .st-emotion-cache-1r4qj8v, .st-emotion-cache-0 {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# --- CORE FUNCTIONS (Cached for Performance) ---
@st.cache_data(ttl="1h")
def get_analysis_from_ai(_api_key, comments):
    """Analyzes a list of comments using the high-accuracy AI model."""
    genai.configure(api_key=_api_key)
    # UPGRADE: Using the more powerful gemini-1.5-pro model
    model = genai.GenerativeModel('gemini-1.5-pro')
    results = []

    for comment in comments:
        # UPGRADE: Refined prompt for higher accuracy and fewer neutrals.
        prompt = f"""
        Act as a world-class sentiment analysis expert. Your analysis must be sharp and decisive.
        Analyze the following customer comment. Acknowledge and interpret all emojis.
        Only classify as 'Neutral' if the comment is purely informational with zero emotional content (e.g., "The package arrived").
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
            results.append(analysis)
            time.sleep(0.5) # Rate limiting
        except Exception:
            # If a single comment fails, record it and move on.
            results.append({'Original Comment': comment, 'Sentiment': 'Error', 'Explanation': 'Failed to analyze.'})
            
    return pd.DataFrame(results)

def generate_visuals(df):
    """Generates a dictionary of all visual elements (figures)."""
    visuals = {}

    # --- Sentiment Chart ---
    sentiment_counts = df['Sentiment'].value_counts()
    color_map = {'Positive': '#2ca02c', 'Negative': '#d62728', 'Neutral': '#ff7f0e', 'Error': '#7f7f7f'}
    plot_order = [s for s in ['Positive', 'Negative', 'Neutral', 'Error'] if s in sentiment_counts.index]
    
    fig_sent, ax_sent = plt.subplots()
    sentiment_counts.loc[plot_order].plot(kind='bar', ax=ax_sent, color=[color_map.get(s) for s in plot_order])
    ax_sent.set_ylabel('Number of Comments')
    ax_sent.set_xticklabels(ax_sent.get_xticklabels(), rotation=0)
    total = len(df)
    for p in ax_sent.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        ax_sent.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    visuals['sentiment_chart'] = fig_sent

    # --- Word Cloud ---
    # FIX: Comprehensive Spanish stopwords list
    stopwords_es = ['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'ha', 'me', 'si', 'porque', 'esta', 'cuando', 'muy', 'sin', 'sobre', 'también', 'fue', 'hasta', 'hay', 'mi', 'eso', 'todo', 'está', 'son', 'qué', 'pero', 'eso']
    text_for_cloud = ' '.join(df['Original Comment'].dropna())
    text_no_emojis = ''.join(c for c in text_for_cloud if c not in emoji.EMOJI_DATA)
    if text_no_emojis.strip():
        wc = WordCloud(width=800, height=400, background_color='white', stopwords=set(stopwords_es), collocations=False).generate(text_no_emojis)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        visuals['word_cloud'] = fig_wc

    # --- Emoji Ranking ---
    all_emojis = extract_emojis(''.join(df['Original Comment'].dropna()))
    if all_emojis:
        visuals['emoji_ranking'] = Counter(all_emojis).most_common(5)

    return visuals

def extract_emojis(text):
    return [c for c in text if c in emoji.EMOJI_DATA]

# ==========================================================================
# APP LAYOUT
# ==========================================================================

# --- Initialize session state ---
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None
if "text_input_val" not in st.session_state:
    st.session_state.text_input_val = ""

# --- API Key Check ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("FATAL ERROR: Your Google AI API Key is not configured in Streamlit Secrets.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("IA Chat")
    st.write("Ask questions about the analyzed data.")
    
    if st.session_state.analysis_df is not None:
        if prompt := st.chat_input("What is the main complaint?"):
            # Placeholder for future advanced chat functionality
            st.info("Chat functionality is in development.")
    else:
        st.info("Analyze some data first to enable the chat.")
    
    if st.session_state.analysis_df is not None:
        if st.button("Start New Analysis"):
            st.session_state.analysis_df = None
            st.session_state.text_input_val = ""
            st.rerun()

# --- MAIN CONTENT ---
if st.session_state.analysis_df is None:
    # --- DATA INPUT VIEW ---
    st.header("Comment Analysis Platform")
    st.markdown("Provide your data through one of the methods below to start the analysis.")
    
    input_container = st.container(border=True)
    with input_container:
        tab1, tab2, tab3 = st.tabs(["Paste Comments", "Upload Excel", "Google Sheets"])
        with tab1:
            st.session_state.text_input_val = st.text_area("Paste comments here:", height=250, value=st.session_state.text_input_val)
            if st.button("Clear"):
                st.session_state.text_input_val = ""
                st.rerun()
        with tab2:
            uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])
        with tab3:
            gsheets_link = st.text_input("Paste Google Sheets link")
            
    col_name = st.text_input("Name of the comment column:", value="Comments")

    if st.button("✨ Analyze Now!"):
        comments = []
        if st.session_state.text_input_val:
            comments = [line.strip() for line in st.session_state.text_input_val.split('\n') if line.strip()]
        # Add your Excel and GSheets loading logic here if needed
        
        if comments:
            with st.spinner("The AI is working its magic... this may take a moment."):
                st.session_state.analysis_df = get_analysis_from_ai(api_key, comments)
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
            with st.container(border=True):
                st.pyplot(visuals['sentiment_chart'])
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
        # FIX: Reorder columns to show original comment first
        cols = df.columns.tolist()
        if 'Original Comment' in cols:
            cols.insert(0, cols.pop(cols.index('Original Comment')))
            st.dataframe(df[cols])
        else:
            st.dataframe(df)

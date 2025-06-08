# --------------------------------------------------------------------------
# COMMENT ANALYSIS APP V4.2 - By Asistente de Programación
# Separated Word Cloud and Emoji Ranking for robustness.
# --------------------------------------------------------------------------
import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import google.generativeai as genai
import time
import re
import emoji # <-- New library
from collections import Counter

# --- AI analysis function (no changes) ---
def analyze_comment_with_gemini(api_key, comment):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Analyze the following customer comment as an expert business analyst.
        Extract the following information in this exact format:
        Sentiment: [Positive, Negative, or Neutral]
        Topics: [A list of 2-3 main keywords or topics, separated by commas]
        Sarcasm: [Yes or No]
        Explanation: [A concise 1-2 sentence explanation of what the customer means, interpreting emojis and context]

        COMMENT: "{comment}"
        """
        response = model.generate_content(prompt)
        analysis = {}
        lines = response.text.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                analysis[key.strip()] = value.strip()
        return analysis
    except Exception as e:
        return {"Error": f"API Error: {e}"}

# --- NEW: Function to extract emojis ---
def extract_emojis(text):
    return [c for c in text if c in emoji.EMOJI_DATA]

# --- Word Cloud function (updated to be word-only) ---
def generate_word_cloud(df):
    # Remove emojis before creating the word cloud
    text_no_emojis = df['Original Comment'].str.cat(sep=' ')
    text_no_emojis = ''.join(c for c in text_no_emojis if c not in emoji.EMOJI_DATA)
    
    if not text_no_emojis.strip():
        return None
    
    stopwords_en = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        stopwords=stopwords_en,
        collocations=False,
        colormap='viridis'
    ).generate(text_no_emojis)
    
    return wordcloud

# --------------------------------------------------------------------------
# INTERFACE V4.2
# --------------------------------------------------------------------------
st.set_page_config(page_title="Comment Analysis", layout="wide")
st.title("💡 Comment Analysis")

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    api_key = None

st.sidebar.header("1. Load Data")
# ... (Sidebar code remains the same) ...
uploaded_file = st.sidebar.file_uploader("📂 Upload an Excel File", type=['xlsx'])
gsheets_link = st.sidebar.text_input("🔗 Paste a Google Sheets Link")
text_input = st.sidebar.text_area("✍️ Or paste comments here")
column_name = st.sidebar.text_input("Name of the column with comments:", value="Comments")

if st.sidebar.button("🚀 Analyze Comments"):
    if not api_key:
        st.error("ERROR: API Key not found. Please make sure you have a .streamlit/secrets.toml file.")
    else:
        # ... (Data loading logic remains the same) ...
        comments_list = []
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                if column_name in df.columns:
                    comments_list = df[column_name].dropna().astype(str).tolist()
                else: st.error(f"Column '{column_name}' not found.")
            except Exception as e: st.error(f"Error reading Excel file: {e}")
        elif gsheets_link:
            try:
                csv_url = gsheets_link.replace('/edit?usp=sharing', '/export?format=csv')
                df = pd.read_csv(csv_url, engine='python', on_bad_lines='skip')
                if column_name in df.columns:
                    comments_list = df[column_name].dropna().astype(str).tolist()
                else: st.error(f"Column '{column_name}' not found.")
            except Exception as e: st.error(f"Error reading Google Sheets: {e}")
        elif text_input:
            comments_list = [line.strip() for line in text_input.split('\n') if line.strip()]

        if not comments_list:
            st.warning("No comments found to analyze.")
        else:
            analysis_results = []
            progress_bar = st.progress(0, text="Analyzing comments...")
            for i, comment in enumerate(comments_list):
                result = analyze_comment_with_gemini(api_key, comment)
                result['Original Comment'] = comment
                analysis_results.append(result)
                time.sleep(0.5) 
                progress_bar.progress((i + 1) / len(comments_list), text=f"Analyzing comment {i+1}/{len(comments_list)}")
            
            df_results = pd.DataFrame(analysis_results)
            st.session_state['df_results'] = df_results
            st.success("Deep analysis complete!")

# --- Display Results ---
if 'df_results' in st.session_state:
    df_results = st.session_state['df_results']
    
    if st.button("🧹 Clear Results & Start Over"):
        del st.session_state['df_results']
        st.rerun()

    st.header("📈 Analysis Dashboard")

    if 'Sentiment' in df_results.columns:
        df_results['Sentiment'] = df_results['Sentiment'].str.strip().str.replace(r'[\[\]]', '', regex=True)
        
        col1, col2 = st.columns([2, 1.5]) # Adjust column ratio
        with col1:
            st.subheader("Sentiment Distribution")
            # ... (Sentiment chart code remains the same) ...
            sentiment_counts = df_results['Sentiment'].value_counts()
            color_map = {'Positive': '#2ca02c', 'Negative': '#d62728', 'Neutral': '#ff7f0e'}
            plot_order = [s for s in ['Positive', 'Negative', 'Neutral'] if s in sentiment_counts.index]
            sentiment_counts = sentiment_counts.loc[plot_order]
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind='bar', ax=ax, color=[color_map.get(s, '#7f7f7f') for s in sentiment_counts.index])
            ax.set_ylabel('Number of Comments')
            ax.set_xticklabels(sentiment_counts.index, rotation=45)
            total_comments = len(df_results)
            for p in ax.patches:
                percentage = f'{100 * p.get_height() / total_comments:.1f}%'
                x = p.get_x() + p.get_width() / 2
                y = p.get_height()
                ax.annotate(percentage, (x, y), ha='center', va='bottom')
            st.pyplot(fig)

        with col2:
            # --- NEW: Emoji Ranking ---
            st.subheader("Top Emojis Used")
            all_text = " ".join(comment for comment in df_results['Original Comment'].dropna())
            emojis_found = extract_emojis(all_text)
            if emojis_found:
                emoji_counts = Counter(emojis_found)
                top_emojis = emoji_counts.most_common(5)
                for emoji_char, count in top_emojis:
                    st.markdown(f"### {emoji_char} <span style='font-size: 1rem; color: grey;'>&nbsp;x {count}</span>", unsafe_allow_html=True)
            else:
                st.info("No emojis found in the comments.")

    # --- NEW Word Cloud section (now separate) ---
    st.subheader("Word Cloud")
    wordcloud = generate_word_cloud(df_results)
    if wordcloud:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    st.header("📝 Detailed Analysis Results")
    # ... (Detailed results table and chat remain the same) ...
    if 'Original Comment' in df_results.columns:
        all_cols = df_results.columns.tolist()
        all_cols.insert(0, all_cols.pop(all_cols.index('Original Comment')))
        display_cols = ['Original Comment', 'Sentiment', 'Sarcasm', 'Topics', 'Explanation']
        final_cols_to_display = [col for col in display_cols if col in all_cols]
        st.dataframe(df_results[final_cols_to_display])
    else:
        st.dataframe(df_results)

    st.header("💬 Chat About the Analysis")
    if not api_key:
        st.info("API Key not found. Chat is disabled.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hi! I've analyzed the data in depth. What do you need to know?"}]
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("E.g., What is the main complaint based on the explanations?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    context_for_ia = f"Here is the full analysis in a dataframe:\n{df_results.to_string()}"
                    genai.configure(api_key=api_key)
                    chat_model = genai.GenerativeModel('gemini-1.5-flash')
                    full_prompt = f"""
                    You are an expert business analyst. Based on the following data analysis, answer the user's question.
                    --- ANALYSIS DATA ---
                    {contexto_for_ia}
                    --- END DATA ---
                    USER QUESTION: {prompt}
                    """
                    response = chat_model.generate_content(full_prompt)
                    st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        


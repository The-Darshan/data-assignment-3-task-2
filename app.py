import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import io

st.set_page_config(page_title="Aurora Skies Airways FAQ Chatbot",
                   page_icon="✈️",
                   layout="wide")

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stTextInput > div > div > input {
        background-color: white;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #2b5797;
        color: white;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #ffffff;
        color: #000000;
        margin-right: 20%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .source-info {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid #e0e0e0;
    }
    </style>
""",
            unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'faq_data' not in st.session_state:
    st.session_state.faq_data = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'tfidf_matrix' not in st.session_state:
    st.session_state.tfidf_matrix = None

def load_faq_data():
    """Load FAQ data from CSV string"""
    df = pd.read_csv(io.StringIO(airline_faq.csv))
    return df


def initialize_rag_system(faq_df):
    """Initialize the RAG system with TF-IDF vectorization"""
    # Combine question and answer for better retrieval
    documents = (faq_df['Question'] + ' ' + faq_df['Answer']).tolist()

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(documents)

    return vectorizer, tfidf_matrix


def retrieve_relevant_faqs(query, vectorizer, tfidf_matrix, faq_df, top_k=3):
    # Transform query
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    relevant_faqs = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Minimum similarity threshold
            relevant_faqs.append({
                'question': faq_df.iloc[idx]['Question'],
                'answer': faq_df.iloc[idx]['Answer'],
                'score': similarities[idx]
            })

    return relevant_faqs


def generate_answer_with_gemini(query, relevant_faqs, api_key):
    """Generate answer using Gemini API"""
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Build context from relevant FAQs
        context = "\n\n".join([
            f"FAQ {i+1}:\nQ: {faq['question']}\nA: {faq['answer']}"
            for i, faq in enumerate(relevant_faqs)
        ])

        # Create prompt
        prompt = f"""You are a helpful customer service assistant for Aurora Skies Airways. 
Based on the following FAQ information, answer the user's question accurately and concisely.

FAQ Context:
{context}

User Question: {query}

Instructions:
- Provide a clear, helpful answer based on the FAQ information above
- If the information isn't in the FAQs, politely say so and suggest contacting customer service
- Be professional and friendly
- Keep your answer concise but complete

Answer:"""

        # Generate response
        response = model.generate_content(prompt)
        return response.text, relevant_faqs

    except Exception as e:
        return f"Error generating response: {str(e)}", relevant_faqs


with st.sidebar:
    st.title("⚙️ Configuration")

    # API Key input
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Get your API key from https://aistudio.google.com/app/apikey")

    st.markdown("---")

    st.subheader("📤 Upload Custom FAQs")
    uploaded_file = st.file_uploader(
        "Upload FAQ CSV",
        type=['csv'],
        help="CSV file with 'Question' and 'Answer' columns")

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            if 'Question' in uploaded_df.columns and 'Answer' in uploaded_df.columns:
                if st.button("Load Custom FAQs"):
                    st.session_state.faq_data = uploaded_df
                    st.session_state.vectorizer, st.session_state.tfidf_matrix = initialize_rag_system(
                        uploaded_df)
                    st.session_state.messages = []
                    st.success(
                        f"✅ Loaded {len(uploaded_df)} FAQs successfully!")
                    st.rerun()
            else:
                st.error("⚠️ CSV must have 'Question' and 'Answer' columns")
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")

    if st.button("Reset to Default FAQs"):
        st.session_state.faq_data = load_faq_data()
        st.session_state.vectorizer, st.session_state.tfidf_matrix = initialize_rag_system(
            st.session_state.faq_data)
        st.session_state.messages = []
        st.success("✅ Reset to default FAQs")
        st.rerun()

    st.markdown("---")

    st.markdown("""
    ### About
    This chatbot uses:
    - **RAG** (Retrieval-Augmented Generation)
    - **TF-IDF** for document retrieval
    - **Gemini API** for response generation
    
    ### How it works
    1. Your question is processed
    2. Relevant FAQs are retrieved
    3. Gemini generates a contextual answer
    """)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.title("✈️ Aurora Skies Airways FAQ Chatbot")
st.markdown("Ask me anything about refunds, cancellations, or flight changes!")

if st.session_state.faq_data is None:
    with st.spinner("Loading FAQ database..."):
        st.session_state.faq_data = load_faq_data()
        st.session_state.vectorizer, st.session_state.tfidf_matrix = initialize_rag_system(
            st.session_state.faq_data)

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]

    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong><br>{content}
        </div>
        """,
                    unsafe_allow_html=True)
    else:
        sources_html = ""
        if "sources" in message and message["sources"]:
            sources_html = "<div class='source-info'><strong>📚 Relevant FAQs:</strong><br>"
            for i, source in enumerate(message["sources"], 1):
                sources_html += f"{i}. {source['question'][:100]}...<br>"
            sources_html += "</div>"

        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>Assistant:</strong><br>{content}
            {sources_html}
        </div>
        """,
                    unsafe_allow_html=True)

user_input = st.chat_input("Type your question here...")

if user_input:
    if not api_key:
        st.error(
            "⚠️ Please enter your Gemini API key in the sidebar to continue.")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong><br>{user_input}
        </div>
        """,
                    unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            # Retrieve relevant FAQs
            relevant_faqs = retrieve_relevant_faqs(
                user_input, st.session_state.vectorizer,
                st.session_state.tfidf_matrix, st.session_state.faq_data)

            answer, sources = generate_answer_with_gemini(
                user_input, relevant_faqs, api_key)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        # Rerun to display new messages
        st.rerun()

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 1rem;'>
    <p>Aurora Skies Airways FAQ Chatbot | Powered by Gemini AI</p>
</div>
""",
            unsafe_allow_html=True)

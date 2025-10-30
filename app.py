import chainlit as cl
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
 
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
 
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
gemini_assistant = genai.GenerativeModel("gemini-2.5-flash")
faq_vector_index = None
faq_questions_list = []
faq_answers_list = []
 
def setup_faq_index(data_file="airline_faq.csv"):
    """Load FAQ data and create FAISS index for semantic search."""
    global faq_vector_index, faq_questions_list, faq_answers_list
 
    faq_df = pd.read_csv(data_file, usecols=["Question", "Answer"])
    faq_questions_list = faq_df["Question"].dropna().tolist()
    faq_answers_list = faq_df["Answer"].dropna().tolist()
 
    question_embeddings = embedding_model.encode(faq_questions_list, show_progress_bar=False)
    embedding_dim = question_embeddings.shape[1]
 
    faq_vector_index = faiss.IndexFlatL2(embedding_dim)
    faq_vector_index.add(np.array(question_embeddings).astype("float32"))
 
def find_relevant_faqs(user_input, top_k=3):
    """Search for top-k semantically similar FAQs."""
    query_embedding = embedding_model.encode([user_input])
    distances, indices = faq_vector_index.search(np.array(query_embedding).astype("float32"), top_k)
 
    results = []
    for idx in indices[0]:
        results.append({
            "question": faq_questions_list[idx],
            "answer": faq_answers_list[idx]
        })
    return results
 
def generate_assistant_reply(user_input, matched_faqs):
    """Generate a response using Gemini model based on matched FAQs."""
    faq_context_block = "\n\n".join([
        f"FAQ {i+1}:\nQ: {faq['question']}\nA: {faq['answer']}"
        for i, faq in enumerate(matched_faqs)
    ])
 
    prompt = f"""
You are Aurora Skies Airways' virtual assistant.
Respond strictly using the FAQs below. If the answer isn't found, say:
"I don't have specific information about that in our FAQ database. Please contact Aurora Skies Airways customer service for assistance."
 
FAQs:
{faq_context_block}
 
Customer Query: {user_input}
 
Response:"""
 
    response = gemini_assistant.generate_content(prompt)
    return response.text
 
# Build index on startup
setup_faq_index()
 
@cl.on_chat_start
async def welcome_user():
    await cl.Message(content="You can ask Your Queries from our AI ChatBot Developed by SDE at CG Infinity").send()
    await cl.Message(
        content="""👋 Hi there! I'm your travel assistant from **Aurora Skies Airways**.
 
I can help you with common inquiries
 
How can I assist you today?""",
       
    ).send()
 
@cl.on_message
async def process_user_query(message: cl.Message):
    loading = cl.Message(content="🔍 Evaluating...")
    await loading.send()
 
    matched_faqs = find_relevant_faqs(message.content, top_k=3)
    loading.content = "🧠 Thinking..."
    await loading.update()
 
    assistant_reply = generate_assistant_reply(message.content, matched_faqs)
    await loading.remove()
 
    await cl.Message(content=assistant_reply, author="Aurora Assistant ✨").send()
 
    reference_note = "\n\n**📚 FAQ Sources:**\n"
    for i, faq in enumerate(matched_faqs[:2], 1):
        reference_note += f"\n{i}. *{faq['question']}*"
    await cl.Message(content=reference_note).send()
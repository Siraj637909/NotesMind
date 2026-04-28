import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import numpy as np
import os

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(page_title="College Notes RAG", page_icon="📚")
st.title("📚 College Notes Q&A")
st.caption("Upload your notes → Ask any question → Get answers instantly")

# ─── Sidebar: API Key ───────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Enter your Gemini API key")
    st.markdown("2. Upload your PDF notes")
    st.markdown("3. Ask any question!")

# ─── Stop if no API key ─────────────────────────────────────────
if not api_key:
    st.warning("👈 Enter your Gemini API key in the sidebar to start.")
    st.stop()

# ─── Configure Gemini ───────────────────────────────────────────
genai.configure(api_key=api_key)

# ─── Helper: Extract text from PDF ──────────────────────────────
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ─── Helper: Split text into chunks ─────────────────────────────
def split_into_chunks(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# ─── Helper: Get embedding from Gemini ──────────────────────────
def get_embedding(text):
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return np.array(result['embedding'])

# ─── Helper: Find top relevant chunks ───────────────────────────
def find_relevant_chunks(question_embedding, chunk_embeddings, chunks, top_k=3):
    similarities = []
    for i, emb in enumerate(chunk_embeddings):
        # Cosine similarity
        score = np.dot(question_embedding, emb) / (
            np.linalg.norm(question_embedding) * np.linalg.norm(emb) + 1e-10
        )
        similarities.append((score, i))
    
    similarities.sort(reverse=True)
    top_chunks = [chunks[i] for _, i in similarities[:top_k]]
    return top_chunks

# ─── Helper: Ask Gemini with context ────────────────────────────
def ask_gemini(question, context_chunks):
    context = "\n\n".join(context_chunks)
    
    prompt = f"""You are a helpful study assistant. Answer the student's question 
using ONLY the information from the notes provided below.
If the answer is not in the notes, say "I couldn't find this in your notes."

NOTES:
{context}

STUDENT QUESTION: {question}

ANSWER:"""
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# ─── Session State ───────────────────────────────────────────────
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── File Upload ─────────────────────────────────────────────────
uploaded_file = st.file_uploader("📄 Upload your college notes (PDF)", type=["pdf"])

if uploaded_file:
    # Only process if new file uploaded
    if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
        with st.spinner("Reading and indexing your notes... (this takes ~30 seconds)"):
            # Extract text
            raw_text = extract_text_from_pdf(uploaded_file)
            
            if not raw_text.strip():
                st.error("Could not extract text from this PDF. Try a different file.")
                st.stop()
            
            # Chunk it
            chunks = split_into_chunks(raw_text)
            
            # Embed all chunks
            embeddings = []
            progress = st.progress(0)
            for i, chunk in enumerate(chunks):
                emb = get_embedding(chunk)
                embeddings.append(emb)
                progress.progress((i + 1) / len(chunks))
            
            st.session_state.chunks = chunks
            st.session_state.chunk_embeddings = embeddings
            st.session_state.last_file = uploaded_file.name
            st.session_state.chat_history = []
        
        st.success(f"✅ Notes indexed! Found {len(chunks)} sections. Ask me anything!")

# ─── Chat Interface ───────────────────────────────────────────────
if st.session_state.chunks:
    st.markdown("---")
    
    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Question input
    question = st.chat_input("Ask a question about your notes...")
    
    if question:
        # Show user message
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Searching your notes..."):
                question_emb = get_embedding(question)
                relevant = find_relevant_chunks(
                    question_emb,
                    st.session_state.chunk_embeddings,
                    st.session_state.chunks
                )
                answer = ask_gemini(question, relevant)
            st.write(answer)
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

elif not uploaded_file:
    st.info("👆 Upload a PDF to get started.")

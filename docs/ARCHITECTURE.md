# 🏗️ NotesMind — Architecture Deep Dive

## RAG Pipeline

```
┌─────────────────────────────────────────────────────┐
│                    USER BROWSER                      │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │ PDF.js   │───▶│ Chunker  │───▶│ Keyword      │  │
│  │ (reader) │    │ 400 words│    │ Scorer (TF)  │  │
│  └──────────┘    └──────────┘    └──────┬───────┘  │
│                                         │           │
│                                    Top 5 chunks     │
│                                         │           │
│                                         ▼           │
│                                  ┌─────────────┐   │
│                                  │ Gemini API  │   │
│                                  │ REST call   │   │
│                                  └──────┬──────┘   │
│                                         │           │
│                                         ▼           │
│                                    Chat Answer      │
└─────────────────────────────────────────────────────┘
```

## Component Details

### PDF.js
- Library by Mozilla
- Reads PDF entirely in browser
- No server upload — privacy preserved
- Extracts text page by page

### Chunker
- Splits full text by words
- chunk_size = 400 words
- overlap = 60 words
- Overlap prevents losing context at boundaries

### Keyword Scorer (TF)
- For each chunk, count occurrences of question words
- Filter words shorter than 2 characters
- Sort chunks by score descending
- Return top 5 chunks

### Gemini API Call
- Model: gemini-2.0-flash
- Endpoint: generativelanguage.googleapis.com
- Prompt includes: system instruction + 5 chunks + question
- Temperature: default (0.7) — can tune lower for factual use

## Data Flow

```
User uploads PDF
       │
       ▼
extractPDFText(file)        → raw text string
       │
       ▼
chunkText(text, 400, 60)    → array of chunk strings
       │
       ▼
[stored in browser memory as JS array]
       │
User asks question
       │
       ▼
topChunks(question, chunks) → top 5 most relevant chunks
       │
       ▼
askGemini(question, chunks) → answer string
       │
       ▼
displayed in chat UI
```

## Why No Vector DB?

For small documents (1 PDF = a few hundred chunks), keyword scoring is:
- Fast enough (milliseconds)
- No API calls needed
- Works offline
- Zero setup

For larger projects (many documents), replace keyword scoring with:
- sentence-transformers embeddings
- ChromaDB or FAISS vector store
- Cosine similarity search

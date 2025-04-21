🔹 How It Works?
User uploads a PDF → Extracts text using PyMuPDF (fitz).
Splits text into chunks using RecursiveCharacterTextSplitter.
Stores chunks in ChromaDB for retrieval.
User asks a question → Finds most relevant chunks.
Summarization LLM generates a concise response.
📝 Alternative Summarization Approach
While using RAG to retrieve relevant sections and summarize is efficient, we could consider:

Hierarchical Summarization:

First pass: Extract key sentences from each chunk (e.g., using BART or T5).
Second pass: Summarize across extracted sentences.
🔥 Pros: More contextually aware, structured summaries.
⚡ Cons: More compute-intensive.
Fine-Tuning a Summarization Model:

Train DeepSeek-1.5 or FLAN-T5 on scientific papers.
🔥 Pros: Better summaries tailored to papers.
⚡ Cons: Requires a dataset and training resources.
Hybrid Approach (RAG + Summarization LLM):

Retrieve best chunks and use a specialized summarization LLM (BART, T5, DeepSeek-1.5).
🔥 Pros: Ensures precision while reducing hallucination.
⚡ Cons: Needs proper prompt engineering.
📊 Performance Evaluation: Agentic RAG vs DeepSeek-1.5
We can evaluate both models using these metrics:

ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Measures overlap of words/phrases between generated summary and reference summary.
✅ Higher ROUGE → Better recall-based summarization.
BLEU (Bilingual Evaluation Understudy)

Measures how well generated summaries match ground truth summaries.
✅ Better for structured summarization.
BERTScore

Uses BERT embeddings to compare semantic similarity of generated vs. reference summary.
✅ Captures meaning better than word overlap metrics.
Summarization Length & Readability

DeepSeek-1.5 → May generate longer, more verbose summaries.
Agentic RAG → Focuses only on retrieved content.
📌 Conclusion
Agentic RAG → Great for retrieving & summarizing relevant sections.
DeepSeek-1.5 → May work better for full-document summarization.
Hybrid Approach → 🔥 Best of both worlds: Use RAG for retrieval + DeepSeek-1.5 for summarization.

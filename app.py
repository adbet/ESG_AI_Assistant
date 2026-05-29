import streamlit as st
from rag.ingestion import load_pdf, chunk_text
from rag.embeddings import get_embeddings
from rag.vectorstore import VectorStore
from rag.retriever import basic_retrieve, hyde_retrieve, mqr_retrieve
from rag.llm import generate_answer

st.set_page_config(page_title="ESG Assistant", page_icon="🌱", layout="wide")

if "store" not in st.session_state:
    st.session_state.store = VectorStore()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

with st.sidebar:
    st.title("🌱 ESG Assistant")
    st.caption("AI-powered ESG document analysis · FTD Master")
    st.divider()

    st.subheader("📄 Documents")
    uploaded_files = st.file_uploader("Upload ESG PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        if new_files:
            for file in new_files:
                with st.spinner(f"Indexing {file.name}…"):
                    try:
                        pages = load_pdf(file.read(), file.name)
                        chunks = chunk_text(pages)
                        if chunks:
                            embeddings = get_embeddings([c["text"] for c in chunks])
                            st.session_state.store.add(chunks, embeddings)
                            st.session_state.processed_files.append(file.name)
                            st.success(f"✅ {file.name} — {len(chunks)} chunks")
                    except Exception as e:
                        st.error(f"Error indexing {file.name}: {e}")

    if st.session_state.processed_files:
        st.divider()
        for fname in st.session_state.processed_files:
            st.write(f"✅ {fname}")

    st.divider()
    st.subheader("⚙️ Retrieval Strategy")
    method = st.selectbox("Method", ["Basic", "HyDE", "MQR"])
    top_k = st.slider("Top-K chunks", 3, 10, 5)

    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

st.header("💬 Ask about ESG Documents")

if not st.session_state.processed_files:
    st.info("Upload one or more ESG PDF reports in the sidebar to get started.")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📎 Sources", expanded=False):
                for s in msg["sources"]:
                    st.caption(f"**{s['source']}** — page {s['page']}  |  score: {s['score']:.3f}")
                    st.text(s["text"][:300] + ("…" if len(s["text"]) > 300 else ""))

if prompt := st.chat_input("Ask a question about the ESG documents…"):
    if len(st.session_state.store) == 0:
        st.warning("Please upload at least one ESG PDF first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Retrieving ({method}) and generating…"):
                cid = st.session_state.store
                if method == "HyDE":
                    chunks = hyde_retrieve(prompt, cid, k=top_k)
                elif method == "MQR":
                    chunks = mqr_retrieve(prompt, cid, k=top_k)
                else:
                    chunks = basic_retrieve(prompt, cid, k=top_k)
                answer = generate_answer(prompt, chunks)

            st.markdown(answer)
            with st.expander("📎 Sources", expanded=False):
                for chunk in chunks:
                    st.caption(f"**{chunk['source']}** — page {chunk['page']}  |  score: {chunk['score']:.3f}")
                    st.text(chunk["text"][:300] + ("…" if len(chunk["text"]) > 300 else ""))

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": chunks,
        })

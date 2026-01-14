import streamlit as st
import os
import tempfile
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# for Tab area
st.set_page_config(page_title="Mini RAG (Offline Llama 3.1)", page_icon="🧠")

# Sidebar stuffs
st.sidebar.title("Specifications")
st.sidebar.success("✅ Running locally on RTX 3050 6GB VRAM")
st.sidebar.markdown("**Model:** Llama 3.1 (8B)")

# Page title / area
st.title("📄 Document Q&A AI (Mini RAG)")
st.write("Upload documents. I will answer from them using local **Llama 3.1**.")

# File upload
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs (2-3 files)", 
    type="pdf", 
    accept_multiple_files=True
)

# to keep the memory of the st
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# for document processing area
if st.sidebar.button("Click To Process Documents"):
    if not uploaded_files:
        st.sidebar.error("⚠️ Please upload at least one PDF.")
    else:
        with st.spinner("Processing Documents locally..."):
            try:
                # 1. Load Documents
                documents = []
                for uploaded_file in uploaded_files:
                    # Create a temp area to store the uploaded file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_path = temp_file.name
                    
                    # PDF loading
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    documents.extend(docs)
                    
                    # Cleans the temp 
                    os.remove(temp_path)

                # 2. Splitting Text into Chunks from pdfs
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                
                # 3. Embeddings area
                # We use a small, fast model for the search part to keep it quick
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # 4. Vector Storing
                vector_store = FAISS.from_documents(chunks, embeddings)
                st.session_state.vector_store = vector_store
                
                st.success(f"✅ Ready! Processed {len(chunks)} chunks locally.")
                
            except Exception as e:
                st.error(f"Error processing documents: {e}")

# Prompt area
question = st.text_input("Ask a question about your documents:")

if question:
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload documents first.")
    else:
        # 5. Define LLM (Local Llama 3.1 8B - 4.9GB) 
        llm = ChatOllama(model="llama3.1", temperature=0.3)

        with st.spinner("Thinking locally (Llama 3.1 is reading)..."):
            try:
                # 6. Retrieval process
                # search among chunks(k)
                docs = st.session_state.vector_store.similarity_search(question, k=4)
                # combine those chunks into 1
                context_text = "\n\n".join([doc.page_content for doc in docs])
                
                # 7. Force to stick to the document knowledge than general knowledge 
                final_prompt = f"""
                You are a helpful assistant. Answer the question based ONLY on the following context.
                If the answer is not in the context, strictly reply with "I don't know based on the given documents."
                
                Context:
                {context_text}

                Question: {question}
                """
                # context + user query will be matched here👆 & sent to AI

                # 8. ANSWER
                response = llm.invoke(final_prompt) # looks like computer code so need to convert it to text
                
                st.write("### Answer:")
                st.write(response.content) # normal text conversion & display
                
            except Exception as e:
                st.error(f"Error: {e}") # to avoid app crash
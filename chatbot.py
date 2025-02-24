from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
# from langchain.llms import HuggingFaceHub
from langchain_community.llms import HuggingFaceHub
from google.generativeai import GenerativeModel
import google.generativeai as genai


# Streamlit UI
st.title("Multi-PDF Chat System")
st.sidebar.header("Upload PDFs")

uploaded_files = st.sidebar.file_uploader("Upload multiple PDFs", accept_multiple_files=True, type=["pdf"])

# Load environment variables
load_dotenv()

from huggingface_hub import InferenceClient


if uploaded_files:
    all_text = ""
    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            all_text += page.extract_text() + "\n"

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(all_text)

    # Embed text
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Initialize Chat Model
    # huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    # model_repo_id = "meta-llama/Llama-2-7b-chat-hf"
    # client = InferenceClient(model=model_repo_id, token=huggingface_api_token)

    # def query_hf(prompt, history=[]):
    #     response = client.chat_completion(messages=[{"role": "user", "content": prompt}])
    #     return response["choices"][0]["message"]["content"]

    # Initialize Gemini Chat Model
    google_api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=google_api_key)
    model = GenerativeModel("gemini-pro")

    def query_gemini(prompt, history=[]):
        response = model.generate_content(prompt)
        return response.text

    retriever = vector_store.as_retriever()

    def custom_qa_chain(question, history):
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Context:\n{context}\n\nUser: {question}\nAssistant:"
        return query_gemini(prompt, history)
    
    st.session_state.qa_chain = custom_qa_chain
    st.session_state.history = []
    st.success("PDFs processed successfully!")
    st.ballons()

# Chat interface
if "qa_chain" in st.session_state:
    user_input = st.text_input("Ask something about the PDFs")
    submit_button = st.button("Submit")

    if submit_button and user_input:
        with st.spinner("Generating response..."):
            result = st.session_state.qa_chain(user_input, st.session_state.history)
            st.session_state.history.append((user_input, result))
            st.write(result)




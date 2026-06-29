import os
import uuid
import streamlit as st

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate




load_dotenv()


st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Chat with Mistral AI")


@st.cache_resource
def load_embeddings():
    return MistralAIEmbeddings()


@st.cache_resource
def load_llm():
    return ChatMistralAI(
        model_name="mistral-small-2506"
    )


embeddings = load_embeddings()
llm = load_llm()


if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None


st.sidebar.title("📂 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()


if uploaded_file is not None:

    if st.session_state.current_pdf != uploaded_file.name:

        st.session_state.current_pdf = uploaded_file.name
        st.session_state.messages = []

        os.makedirs("temp", exist_ok=True)

        pdf_path = os.path.join("temp", uploaded_file.name)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing PDF... Please wait..."):

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=400
            )

            chunks = splitter.split_documents(documents)

            db_path = os.path.join(
                "chroma_db",
                str(uuid.uuid4())
            )

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=db_path
            )

            st.session_state.retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10,
                    "lambda_mult": 0.8
                }
            )

        st.sidebar.success("✅ PDF Loaded Successfully!")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI Teacher.

Answer ONLY using the provided PDF context.

If the answer is not present in the PDF, reply exactly:

I couldn't find that information in the PDF.

Keep the answer concise and accurate.
"""
        ),
        (
            "human",
            """
PDF Context:

{docs}

Question:

{question}
"""
        )
    ]
)


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a question from the PDF...")

if query:

    if st.session_state.retriever is None:

        st.warning("Please upload a PDF first.")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Thinking..."):

        docs = st.session_state.retriever.invoke(query)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        final_prompt = prompt.invoke(
            {
                "docs": context,
                "question": query
            }
        )

        response = llm.invoke(final_prompt)

        answer = response.content

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):
        st.markdown(answer)
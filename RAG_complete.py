from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()


Loader = PyPDFLoader("Documents loader/III CSC_IOMP_A7_Facial Age Estimation from Images using Image processing Techniques and Neural Networks_Mrs.Rupinder Saini - First and Final.pdf")
docs = Loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400
)

chunk = splitter.split_documents(docs)

embedding_model = MistralAIEmbeddings()


vectorstore = Chroma.from_documents(
    documents = chunk,
    persist_directory="chroma_db",
    embedding=embedding_model    
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k" : 8,
        "fetch_k": 5,
        "lambda_mult": 0.85
    }
)

llm = ChatMistralAI(
    model_name = "mistral-small-2506"
)


prompt = ChatPromptTemplate.from_messages([

    ("system",
    """
        You are an helpful AI Teacher.

        Answer ONLY from the provided PDF.

        If the answer is not present, say:
        "I couldn't find that information in the PDF."
        """
    ),
    ("human",
      """
        PDF:

        {docs}

        Question:

        {question}
        """
    )

])



while True:
    query = input("Ask a question:")
    if query == "0":
        break

    docs = retriever.invoke(query)
    
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "docs" : context,
        "question" : query
    })
 
    response = llm.invoke(final_prompt)

    print(f"\nAI:\n{response.content}")

import streamlit as st
import pandas as pd
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import CohereEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_cohere import ChatCohere
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()
import os
CSV_PATH = "./data/BetterUp-Classified.csv"
PDF_PATH = "./data/Beliefs-About-Linear-Social-Progress.pdf"
PDF_PATH_2 = "./data/deeds-pamphile-ruttan-2022-the-(bounded)-role-of-stated-lived-value-congruence-and-authenticity-in-employee-evaluations.pdf"
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
llm = ChatCohere(cohere_api_key=COHERE_API_KEY, model="command-r", user_agent="streamlit-app")
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""

You are an AI assistant that answers questions based on provided context. Use the context to answer the question as accurately as possible.
If the context/question relates to conversations, focus on connecting the context together to provide a coherent answer.
If the context/question relates to research papers, focus on extracting relevant information from the papers to answer the question, and ALWAYS IGNORE THE CITATIONS, WORKS CITED,
and FOCUS on the contents of the main papers.

Context:
{context}

Question:
{question}

Answer:
""")



st.title("RAG Chat with CSV + PDF")

source_selection = st.radio(
    "Choose knowledge source for your question:",
    ("None", "Conversations", "Research Paper", "Both"),
    index=0
)

if source_selection == "Conversations":
    df = pd.read_csv(CSV_PATH)
    df.fillna("", inplace=True)
    df["formatted"] = df["Conversation Part 1"] + " " + df["Conversation Part 2"] + " " + df["Conversation Part 3"]
    df = df.drop_duplicates(subset=["formatted"])
    df = df[:50]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=150)
    all_chunks = []
    for idx, row in df.iterrows():
        conversation_id = row.get("convo_id", idx)
        text = row["formatted"]
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(Document(page_content=chunk, metadata={"source": "csv", "id": f"{conversation_id}_{i}"}))

    convo_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    convo_vectordb = Chroma.from_documents(documents=all_chunks,embedding=convo_embeddings,persist_directory="./docs/chroma_conversations")
    convo_retriever = convo_vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8})

if source_selection == "Research Paper":
    loader = PyPDFLoader(PDF_PATH)
    pdf_docs = loader.load()
    loader_2 = PyPDFLoader(PDF_PATH_2)
    pdf2_docs = loader_2.load()

    pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    pdf_chunks = pdf_splitter.split_documents(pdf_docs)
    pdf2_chunks = pdf_splitter.split_documents(pdf2_docs)
    all_pdf_chunks = pdf_chunks + pdf2_chunks

    pdf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    pdf_vectordb = Chroma.from_documents(documents=all_pdf_chunks, embedding=pdf_embeddings, collection_name="research_papers_updated")
    pdf_retriever = pdf_vectordb.as_retriever(search_type = "mmr", search_kwargs={"k": 8})

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Thinking..."):

        if source_selection == "Conversations":
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=convo_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
            result = qa_chain.invoke(query)

        elif source_selection == "Research Paper":
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=pdf_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
            result = qa_chain.invoke(query)

        st.markdown("### Answer")
        st.write(result["result"])

        st.markdown("### Sources")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Chunk {i+1}:**")
            st.markdown(f"**Source:** {doc.metadata.get('id', 'Unknown')}")
            st.markdown(doc.page_content)

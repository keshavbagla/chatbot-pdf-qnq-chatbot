import streamlit as st
import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


load_dotenv()

st.title("📄 Conversational RAG with PDF")
st.write("Upload a PDF and chat with it")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

api_key = st.text_input("Enter your GROQ API key", type="password")

if api_key:

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct"
    )

    session_id = st.text_input("Session ID", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader(
        "Upload PDF",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:

        documents = []

        for uploaded_file in uploaded_files:

            temp_pdf = f"temp_{uploaded_file.name}"

            with open(temp_pdf, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()

            documents.extend(docs)

        st.write("Documents Loaded:", len(documents))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = text_splitter.split_documents(documents)

        st.write("Text Chunks:", len(splits))

        if "vectorstore" not in st.session_state and len(splits) > 0:

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings
            )

            st.session_state.vectorstore = vectorstore

        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        system_prompt = (
            "You are a helpful assistant for answering questions.\n"
            "Use the provided context to answer the question.\n"
            "If the answer is not in the context, say you don't know.\n"
            "Use maximum three sentences.\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        def retrieve_docs(inputs):

            docs = retriever.invoke(inputs["input"])

            context = "\n\n".join(doc.page_content for doc in docs)

            return {
                "context": context,
                "input": inputs["input"],
                "chat_history": inputs["chat_history"],
            }

        retrieval_runnable = RunnableLambda(retrieve_docs)

        rag_pipeline = (
            RunnablePassthrough()
            | retrieval_runnable
            | qa_prompt
            | llm
        )

        def get_session_history(session: str) -> BaseChatMessageHistory:

            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()

            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_pipeline,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        
        user_input = st.text_input("Ask a question about the PDF")

        if user_input:

            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            st.success(response.content)

            st.subheader("Chat History")

            for msg in get_session_history(session_id).messages:
                st.write(f"{msg.type}: {msg.content}")

else:
    st.warning("Please enter your GROQ API key")

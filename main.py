import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

# Check for API Key
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OpenAI API Key is missing! Please set it in your environment variables.")
    st.stop()

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Collect URLs
urls = [st.sidebar.text_input(f"URL {i + 1}") for i in range(3)]
urls = [url for url in urls if url.strip()]  # Remove empty inputs

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

docs = []  # Define before using

if process_url_clicked:
    if not urls:
        st.error("⚠️ Please enter at least one valid URL before processing.")
    else:
        try:
            main_placeholder.text("Loading articles... ⏳")
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            # Split data
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Splitting text... ✂️")
            docs = text_splitter.split_documents(data)

            if not docs:
                st.error("⚠️ No valid text extracted from URLs. Try different sources.")
            else:
                # Create embeddings
                embeddings = OpenAIEmbeddings()
                vectorstore_openai = FAISS.from_documents(docs, embeddings)
                main_placeholder.text("Building FAISS index... 🏗️")
                time.sleep(2)

                # Save FAISS index
                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore_openai, f)
                st.success("✅ Articles processed successfully!")

        except Exception as e:
            st.error(f"❌ Error processing URLs: {str(e)}")

query = st.text_input("Ask a question about the articles:")
if query:
    if not os.path.exists(file_path):
        st.error("⚠️ No data available. Please process URLs first.")
    else:
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                for source in sources.split("\n"):
                    st.write(source)

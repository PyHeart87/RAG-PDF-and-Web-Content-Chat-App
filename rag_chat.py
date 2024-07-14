import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Bedrock
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def process_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def scrape_website(url, max_depth=2):
    def scrape_page(url, depth):
        if depth > max_depth:
            return ""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            
            # Recursively scrape linked pages
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                if urlparse(full_url).netloc == urlparse(url).netloc:
                    text += scrape_page(full_url, depth + 1)
            
            return text
        except Exception as e:
            st.error(f"Error scraping {url}: {str(e)}")
            return ""

    return scrape_page(url, 0)

def create_vector_db(text, identifier):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings()
    
    db_name = f"faiss_{identifier}"
    
    if os.path.exists(f"{db_name}.faiss"):
        vectorstore = FAISS.load_local(db_name, embeddings)
        vectorstore.add_texts(chunks)
        st.success(f"Updated existing database: {db_name}")
    else:
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        st.success(f"Created new database: {db_name}")
    
    vectorstore.save_local(db_name)
    return vectorstore

def get_llm(llm_choice):
    if llm_choice == "Ollama":
        return Ollama(model="llama2:7b")
    elif llm_choice == "OpenAI":
        return ChatOpenAI(api_key=st.secrets["openai_api_key"])
    elif llm_choice == "Claude (Bedrock)":
        return Bedrock(
            model_id="anthropic.claude-v2",
            credentials_profile_name="default",
            region_name="eu-central-1"
        )

def get_conversation_chain(vectorstore, llm):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs and Websites", page_icon=":books:")
    st.header("Chat with Multiple PDFs and Websites :books:")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        st.subheader("Web Content")
        url = st.text_input("Enter a URL to scrape:")
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # Process PDFs
                if pdf_docs:
                    raw_text = process_pdf(pdf_docs)
                    pdf_vectorstore = create_vector_db(raw_text, "pdf_content")
                    st.session_state.vectorstore = pdf_vectorstore
                
                # Process URL
                if url:
                    web_content = scrape_website(url)
                    web_vectorstore = create_vector_db(web_content, "web_content")
                    if st.session_state.vectorstore:
                        st.session_state.vectorstore.merge_from(web_vectorstore)
                    else:
                        st.session_state.vectorstore = web_vectorstore
                
                st.success("Processing complete!")
        
        st.subheader("Choose your LLM")
        llm_choice = st.selectbox("Select LLM", ["Ollama", "OpenAI", "Claude (Bedrock)"])
        
    user_question = st.text_input("Ask a question about your documents or web content:")
    if user_question:
        if st.session_state.vectorstore is None:
            st.warning("Please upload PDFs or provide a URL to scrape first.")
        else:
            llm = get_llm(llm_choice)
            if st.session_state.conversation is None:
                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore, llm)
            
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(f"Human: {message.content}")
                else:
                    st.write(f"AI: {message.content}")

if __name__ == '__main__':
    main()
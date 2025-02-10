import streamlit as st
import imaplib
import email
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

def get_vector_store(text_content):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text_content)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

def get_response(user_query, chat_history, vector_store):
    template = """
    You are an AI assistant analyzing email content. Answer the following question based on the email context and conversation history:

    Context: {context}
    Chat history: {chat_history}
    User question: {user_question}

    Provide a detailed response based on the email content.
    """

    prompt = ChatPromptTemplate.from_template(template)
    
    # Use a local Ollama model
    llm = ChatOllama(model="mistral")  # Change model if needed (e.g., "gemma", "llama3")

    docs = vector_store.similarity_search(user_query)
    context = "\n".join(doc.page_content for doc in docs)
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "user_question": user_query,
    })

def get_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'lxml')
    return soup.get_text()

def process_email_content(email_bytes):
    try:
        msg = BytesParser(policy=policy.default).parsebytes(email_bytes)
        text_content = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    text_content += part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8')
                elif part.get_content_type() == 'text/html':
                    html_content = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8')
                    text_content += get_text_from_html(html_content)
        else:
            if msg.get_content_type() == 'text/plain':
                text_content = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8')
            elif msg.get_content_type() == 'text/html':
                text_content = get_text_from_html(msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8'))
        
        return text_content
    except Exception as e:
        st.error(f"Error processing email: {str(e)}")
        return ""

def search_emails(imap_client, search_keyword):
    try:
        if search_keyword:
            search_criteria = f'BODY "{search_keyword}"'
        else:
            search_criteria = 'ALL'
        
        st.info(f"Searching emails with keyword: {search_keyword}")
        typ, data = imap_client.search(None, search_criteria)
        
        if typ != 'OK':
            st.error("Failed to search emails")
            return []
            
        return data[0].split()
    except Exception as e:
        st.error(f"Error searching emails: {str(e)}")
        return []

def main():
    load_dotenv()
    
    st.set_page_config(page_title="Email RAG Assistant", page_icon="📧")
    st.title("📧 Email RAG Assistant")

    # Initialize session state
    if 'email_content' not in st.session_state:
        st.session_state.email_content = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I can help you analyze your email content. Please search for emails first using the sidebar."),
        ]
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    # Sidebar
    with st.sidebar:
        st.title("Email Search Parameters")
        if st.button("🗑️ Reset Email Content"):
            st.session_state.email_content = [] 
            st.session_state.vector_store = None
            st.success("Email content has been cleared!")
        
        # Email credentials input
        email_user = st.text_input("Gmail Address", os.getenv('GMAIL_USERNAME', ''))
        email_password = st.text_input("App Password", os.getenv('GMAIL_PASSWORD', ''), type="password")
        
        # Search parameter
        keyword = st.text_input("Search Keyword", "")

        if st.button("Process Emails"):
            if not email_user or not email_password:
                st.error("Please provide email credentials")
                return

            try:
                with st.spinner("Connecting to Gmail..."):
                    M = imaplib.IMAP4_SSL('imap.gmail.com')
                    M.login(email_user, email_password)
                    M.select('inbox')
                    
                    email_ids = search_emails(M, keyword)
                    
                    if email_ids:
                        st.session_state.email_content = []
                        progress_bar = st.progress(0)
                        all_content = ""
                        
                        for i, num in enumerate(email_ids):
                            progress = (i + 1) / len(email_ids)
                            progress_bar.progress(progress)
                            
                            typ, email_data = M.fetch(num, '(RFC822)')
                            if typ == 'OK':
                                content = process_email_content(email_data[0][1])
                                if content:
                                    st.session_state.email_content.append(content)
                                    all_content += content + "\n\n"
                        
                        # Create vector store from all email content
                        st.session_state.vector_store = get_vector_store(all_content)
                        st.success(f"Successfully processed {len(st.session_state.email_content)} emails!")
                    else:
                        st.warning("No emails found matching the criteria.")
                    
                    M.logout()
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Display chat interface
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    # Chat input
    if st.session_state.vector_store is not None:
        user_query = st.chat_input("Ask about your emails...")
        if user_query:
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            with st.chat_message("Human"):
                st.write(user_query)

            with st.chat_message("AI"):
                with st.spinner("Analyzing emails..."):
                    response = get_response(user_query, st.session_state.chat_history, st.session_state.vector_store)
                    st.write(response)
                    st.session_state.chat_history.append(AIMessage(content=response))
    else:
        st.info("Please search and process emails first before starting the chat.")

if __name__ == "__main__":
    main()
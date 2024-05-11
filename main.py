import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Set the Google API Key here
GOOGLE_API_KEY = "AIzaSyDCLGksvrltb7vP54stNbSfF7XTmuIo6GM"

# Configure Gemini-Pro API using the correct API key
genai.configure(api_key=GOOGLE_API_KEY)

# Start a new chat session if it doesn't already exist
model = genai.GenerativeModel('gemini-pro')
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

# Define the embeddings variable
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

def user_input(user_question, pdf_paths):
    """Process user input and generate response."""
    combined_text = ""
    for pdf_path in pdf_paths:
        with open(pdf_path, "rb") as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                combined_text += page.extract_text()

    text_chunks = get_text_chunks(combined_text)
    get_vector_store(text_chunks)

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response['output_text']

def get_text_chunks(text):
    """Split the text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Generate vector store from text chunks."""
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Load conversational chain for question answering."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Define paths of the PDF files
PDF_PATHS = [
    "C:/Users/HP/Desktop/banque_AR.pdf",
    "C:/Users/HP/Desktop/Banque_FR.pdf"
]

def main():
    """Main function to setup Streamlit app."""
    st.set_page_config("Chat PDF")

    st.header("WELCOME to our ChatBot ")

    # Add a radio button to choose the type of question
    question_type = st.radio("Choose the type of question", ("From PDF Files", "Outside PDF Files"))

    if question_type == "From PDF Files":
        user_question = st.text_input("Ask a Question ")

        if user_question:
            assistant_response = user_input(user_question, PDF_PATHS)
            st.markdown(
                f"""
                <div class="message-container">
                    <div class="message user-message">
                        <span class="message-text" style="font-size: 17px; font-weight: bold;">👤 {user_question}</span>
                    </div>
                </div>
                <div class="message-container">
                    <div class="message assistant-message">
                        <span class="message-text" style="font-size: 17px; font-weight: bold;">🤖 {assistant_response}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    elif question_type == "Outside PDF Files":
        user_question = st.text_input("Ask a Question")

        if user_question:
            # Process the user's question directly without PDF files
            # You may need to modify this part based on your requirements
            response = st.session_state.chat.send_message(user_question)
            st.markdown(
                f"""
                <div class="message-container">
                    <div class="message user-message">
                        <span class="message-text" style="font-size: 17px; font-weight: bold;">👤 {user_question}</span>
                    </div>
                </div>
                <div class="message-container">
                    <div class="message assistant-message">
                        <span class="message-text" style="font-size: 17px; font-weight: bold;">🤖 {response.text}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# Triggering the main function
if __name__ == "__main__":
    main()

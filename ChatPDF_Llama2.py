# Import necessary modules for processing documents, embeddings, Q&A, etc. from 'langchain' library.
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file.
from langchain.document_loaders import PyPDFLoader  # For loading and reading PDF documents.
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting large texts into smaller chunks.
from langchain.vectorstores import Chroma  # Vector storage system for embeddings.
from langchain.llms import CTransformers  # For loading transformer models.
from InstructorEmbedding import INSTRUCTOR  # Not clear without context, possibly a custom embedding.
from langchain.embeddings import HuggingFaceInstructEmbeddings  # Embeddings from HuggingFace models with instructions.
from langchain.embeddings import HuggingFaceEmbeddings  # General embeddings from HuggingFace models.
from langchain.embeddings import LlamaCppEmbeddings  # Embeddings using the Llama model.
from langchain.chains import RetrievalQA  # Q&A retrieval system.
from langchain.embeddings import OpenAIEmbeddings  # Embeddings from OpenAI models.
from langchain.vectorstores import FAISS  # Another vector storage system for embeddings.

# Import Streamlit for creating a web application and other necessary modules for file handling.
import streamlit as st  # Main library for creating the web application.
import tempfile  # For creating temporary directories and files.
import os  # For handling file and directory paths.

# Import a handler for streaming outputs.
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # For live updates in the Streamlit app.

# Set the title of the Streamlit web application.
st.title("ChatPDF")
# Create a visual separator in the app.
st.write("---")

# Add a file uploader widget for users to upload their PDF files.
uploaded_file = st.file_uploader("Upload your PDF file!", type=['pdf'])
# Another visual separator after the file uploader.
st.write("---")

# Function to convert the uploaded PDF into a readable document format.
def pdf_to_document(uploaded_file):
    # Create a temporary directory for storing the uploaded PDF.
    temp_dir = tempfile.TemporaryDirectory()
    # Get the path where the uploaded PDF will be stored temporarily.
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    
    # Save the uploaded PDF to the temporary path.
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Load the PDF and split it into individual pages.
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# Check if a user has uploaded a file.
if uploaded_file is not None:
    # Convert the uploaded PDF into a document format.
    pages = pdf_to_document(uploaded_file)

    # Initialize a tool to split the document into smaller textual chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,  # Define the size of each chunk.
        chunk_overlap  = 20,  # Define how much chunks can overlap.
        length_function = len  # Function to determine the length of texts.
    )
    # Split the document into chunks.
    texts = text_splitter.split_documents(pages)

    ## Below are examples of different embedding techniques, but they are commented out.

    # Load the desired embeddings model.
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # Load the textual chunks into the Chroma vector store.
    db = Chroma.from_documents(texts, embeddings)

    # Custom handler to stream outputs live to the Streamlit application.
    from langchain.callbacks.base import BaseCallbackHandler
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container  # Streamlit container to display text.
            self.text=initial_text
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text+=token  # Add new tokens to the text.
            self.container.markdown(self.text)  # Display the text.

    # Header for the Q&A section of the web app.
    st.header("Ask the PDF a question!")
    # Input box for users to type their questions.
    question = st.text_input('Type your question')

    # Check if the user has pressed the 'Ask' button.
    if st.button('Ask'):
        # Display a spinner while processing the question.
        with st.spinner('Processing...'):
            # Space to display the answer.
            chat_box = st.empty()
            # Initialize the handler to stream outputs.
            stream_hander = StreamHandler(chat_box)

            # Initialize the Q&A model and chain.
            llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q2_K.bin", model_type="llama", callbacks=[stream_hander])
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            # Get the answer to the user's question.
            qa_chain({"query": question})
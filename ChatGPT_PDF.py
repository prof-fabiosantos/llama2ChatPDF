# Import required modules from 'langchain' for document processing, embeddings, Q&A, etc.
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Importing Streamlit for creating the web app, and other necessary modules for file handling.
import streamlit as st
import tempfile
import os

# Import a handler for streaming outputs.
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Set the title of the Streamlit web application.
st.title("ChatPDF")
# Create a horizontal line for better visual separation in the app.
st.write("---")

# Provide an input box for users to enter their OpenAI API key.
openai_key = st.text_input('Enter OPEN_AI_API_KEY', type="password")

# Provide a file upload widget to let users upload their PDF files.
uploaded_file = st.file_uploader("Upload your PDF file!", type=['pdf'])
# Another visual separation after the file uploader.
st.write("---")

# Define a function that converts the uploaded PDF into a document format.
def pdf_to_document(uploaded_file):
    # Create a temporary directory to store the uploaded PDF file temporarily.
    temp_dir = tempfile.TemporaryDirectory()
    # Join the directory path with the uploaded file name to get the complete path.
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    
    # Write the content of the uploaded file into the temporary file path.
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Use PyPDFLoader to read and split the PDF into individual pages.
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# Check if a file has been uploaded by the user.
if uploaded_file is not None:
    # Convert the uploaded PDF into a document format.
    pages = pdf_to_document(uploaded_file)

    # Initialize a text splitter to break the document into smaller chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        # Define parameters for the splitter: chunk size, overlap, etc.
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len
    )
    # Split the document pages into chunks.
    texts = text_splitter.split_documents(pages)

    # Initialize the OpenAIEmbeddings model for creating embeddings from texts using the provided API key.
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

    # Load the textual chunks into Chroma after creating embeddings.
    db = Chroma.from_documents(texts, embeddings_model)

    # Define a custom handler to stream outputs to the Streamlit app.
    from langchain.callbacks.base import BaseCallbackHandler
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text=initial_text
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text+=token
            self.container.markdown(self.text)

    # Display a header for the question section of the web app.
    st.header("Ask the PDF a question!")
    # Provide an input box for users to type in their questions.
    question = st.text_input('Type your question')

    # Check if the user has clicked on the 'Ask' button.
    if st.button('Ask'):
        # Show a spinner animation while processing the user's question.
        with st.spinner('Processing...'):
            # Create a space to display the answer.
            chat_box = st.empty()
            # Initialize a handler to stream outputs.
            stream_hander = StreamHandler(chat_box)
            # Initialize the ChatOpenAI model for Q&A tasks with streaming enabled.
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key, streaming=True, callbacks=[stream_hander])
            # Create a RetrievalQA chain that uses the ChatOpenAI model and Chroma retriever to answer the question.
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            # Fetch the answer to the user's question.
            qa_chain({"query": question})
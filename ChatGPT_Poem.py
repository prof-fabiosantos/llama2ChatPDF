from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain.chat_models import ChatOpenAI

chat_model = ChatOpenAI()

st.title('AI Poem Generator')

content = st.text_input('Write a content of poem')

if st.button('Request a poem'):
    with st.spinner('writting a poem ...'):
        result = chat_model.predict("write poem about "+content)
        st.write(result)
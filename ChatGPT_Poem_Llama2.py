import streamlit as st
from langchain.llms import CTransformers

llm = CTransformers(
    model="llama-2-7b-chat.ggmlv3.q2_K.bin",
    model_type="llama"
)

st.title('AI Poem Generator')

content = st.text_input('Write a content of poem')

if st.button('Request a poem'):
    with st.spinner('writting a poem ...'):
        result = llm.predict("write poem about "+content)
        st.write(result)
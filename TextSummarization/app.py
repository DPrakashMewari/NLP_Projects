import streamlit as st
import transformers 
from transformers import pipeline




summarizer = pipeline("summarization",model="Falconsai/text_summarization")


# Streamlit App
st.title("Text Summarization with Falcon Sai")
text_input = st.text_area("Enter your text here")
if st.button("Summarize"):
    summary_text = summarizer(text_input, max_length=1000, min_length=30, do_sample=True)
    st.write("Summary:", summary_text[0]['summary_text'])

import newspaper
from transformers import pipeline
import gradio as gr
from deep_translator import GoogleTranslator


# Function to extract and summarize news article
def summarize_news(url, target_language):
    # Initialize HuggingFace pipeline for summarization
    summarizer = pipeline("summarization",model="Falconsai/text_summarization")

    article = newspaper.Article(url)
    article.download()
    article.parse()


    
    print(article.text)
    # Summarize the article content
    summary = summarizer(article.text, max_length=1000, min_length=30, do_sample=False)[0]['summary_text']
    print(summary)


    # convert into hindi    
    summary = GoogleTranslator(source='auto', target='hi').translate(summary)



    return summary

# Define Gradio interface
gr.Interface(
    fn=summarize_news,
    inputs=["text"],
    outputs=["text"],
    title="News Summarizer with Translation",
    description="Enter the URL of the news article and select target language for summary."
).launch()
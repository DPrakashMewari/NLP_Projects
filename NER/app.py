import streamlit as st
import spacy
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from transformers import pipeline

# Initialize models
nlp_spacy = spacy.load('en_core_web_sm')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
nlp_transformers = pipeline("ner", model="Babelscape/wikineural-multilingual-ner", grouped_entities=True)

# Define functions for NER
def spacy_ner(text):
    doc = nlp_spacy(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def nltk_ner(text):
    words = word_tokenize(text)
    tags = pos_tag(words)
    named_entities = ne_chunk(tags)
    return [(chunk[0][0], chunk.label()) for chunk in named_entities if hasattr(chunk, 'label')]

def transformers_ner(text):
    ner_results = nlp_transformers(text)
    return [(entity['word'], entity['entity_group']) for entity in ner_results]

# Streamlit app
st.title("Named Entity Recognition (NER) App")

text = st.text_area("Enter text:", "Apple is looking at buying U.K. startup for $1 billion")

ner_method = st.selectbox("Select NER method:", ("SpaCy", "NLTK", "Transformers"))

if ner_method == "SpaCy":
    entities = spacy_ner(text)
elif ner_method == "NLTK":
    entities = nltk_ner(text)
elif ner_method == "Transformers":
    entities = transformers_ner(text)

# Display results
st.write("### Named Entities:")
for entity in entities:
    st.write(f"{entity[0]} ({entity[1]})")

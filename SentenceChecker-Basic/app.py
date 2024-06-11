import streamlit as st
from spellchecker import SpellChecker

spell = SpellChecker()

# Function to correct spelling in a sentence
def correct_sentence(sentence, exceptions=None):
    exceptions = exceptions or []
    
    exceptions += [word for word in sentence.split() if word.lower() not in spell]

    corrected_words = [spell.correction(word.lower()) if word.lower() not in exceptions else word for word in sentence.split()]
    
    corrected_sentence = ' '.join(corrected_words)
    return corrected_sentence, exceptions

st.title("Spell Checker App")

sentence = st.text_area("Enter a sentence:")

if st.button("Check Spelling"):
    corrected_sentence, exceptions = correct_sentence(sentence)
    
    st.write("Corrected Sentence:", corrected_sentence)
    st.write("Exceptions List:", exceptions)

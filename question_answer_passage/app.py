import streamlit as st
from transformers import pipeline



class QAModel:
    def __init__(self):
        self.qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
    
    def answer_question(self, passage, question):
        answer = self.qa_pipeline(question=question, context=passage)
        return answer['answer'], answer['score']


qa_model = QAModel()
passage = None
question = None

def main():
    st.title("Question Answering with RoBERTa")
    passage = st.text_area("Enter the passage:")
    question = st.text_input("Enter the question:")

    if not passage.strip():
        st.error("Please enter a passage.")
    elif not question.strip():
        st.error("Please enter a question.")
    
    else:
        answer,score = qa_model.answer_question(passage,question)
        st.subheader("Question:")
        st.write(question)
        
        st.subheader("Answer:")
        st.write(answer)
        
        st.subheader("Confidence score:")
        st.write(score)





if __name__ == "__main__":
    main()

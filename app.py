import pdfplumber
import streamlit as st
from transformers import pipeline

# Function to extract text from a PDF file using pdfplumber
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Step 3: Load a question-answering model
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Step 4: Define a function to answer questions based on the PDF content
def answer_question(question, context):
    """
    Answers a question using the provided context from the PDF.
    
    Args:
        question (str): The question to be answered.
        context (str): The context from the PDF to base the answer on.
    
    Returns:
        str: The answer to the question.
    """
    result = qa_model(question=question, context=context)
    return result['answer']

# Step 5: Streamlit app setup
def main():
    st.title("PDF Question-Answering System")
    
    # Upload PDF file
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if pdf_file is not None:
        # Extract text from the uploaded PDF
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(pdf_file)
        
        st.success("Text extracted from the PDF successfully!")
        
        # Display extracted text (optional)
        with st.expander("Show extracted text"):
            st.write(pdf_text)
        
        # Ask questions about the content
        question = st.text_input("Ask a question about the PDF content")
        
        if st.button("Get Answer"):
            if question:
                with st.spinner("Answering your question..."):
                    answer = answer_question(question, pdf_text)
                st.write(f"Answer: {answer}")
            else:
                st.warning("Please enter a question.")
    
if __name__ == "__main__":
    main()

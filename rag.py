import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("OPENAI_API_KEY")

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question with as much detail as possible using the provided context. 

    - If the required information is **fully available** in the context, provide an **in-depth** answer.  
    - If something **related** to the query exists in the context, use **logical reasoning** and **your knowledge** to provide a well-thought-out response.  
    - If the answer is **not available in the context**, respond with: **"Answer is not available in the context."**  
    - If the task involves **arithmetic** (such as addition, subtraction, etc.), **perform the calculation directly** and provide the exact result.  
    - Do **not** generate incorrect or misleading answers.  
    - Do **not** change your response if the same question is asked multiple times.  
    - If the query is **unclear or ambiguous**, ask for **clarification** instead of making assumptions.  
    - If the response contains multiple key points, **structure it using bullet points or paragraphs** for better clarity.  
    - Support **multi-turn conversations**, remembering previous interactions if relevant.  
    - Maintain an **appropriate tone** based on the context—**formal, conversational, concise, or elaborate**, as needed.  

    Context:\n {context}\n
    Question:\n {question}\n

    Answer:
    """



    model = ChatOpenAI(model="gpt-4o",
                       temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF")

    user_question = st.text_input("Ask any Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
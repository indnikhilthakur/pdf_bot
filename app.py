from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    # load environment variable from .env
    load_dotenv()
    # retrieve openai_api_key
    print(os.getenv("OPENAI_API_KEY"))

    # created streamlit web page
    st.set_page_config(page_title="PDF BOT")
    st.header("Ask your pdf: ")

    # upload pdf file
    pdf = st.file_uploader("Upload your pdf", type="pdf")

    # extract text from pdf
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # st.write(text)

        #text split in chunks
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text)

        # st.write(chunks)

        # create embedings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # User Input part

        # user input
        user_question = st.text_input("Ask your question to PDF: ")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            # st.write(docs)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            # tocheck spendings of openai api
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = user_question)
                print(cb)

            st.write(response)
        



if __name__ == '__main__':
    main()

import streamlit as st
import openai
import os
import tempfile
from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
import shutil

# Set OpenAI API key
OPENAI_API_KEY = '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'
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1)

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
st.header("Document Question Answering ChatBot")

class Document:
    def __init__(self, text, metadata={}):
        self.page_content = text
        self.metadata = metadata

persist_directory = './docs/chroma/'
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
os.makedirs(persist_directory, exist_ok=True)

def load_pdf(tmp_file_path):
    pdf_reader = PdfReader(tmp_file_path)
    document_text = ""
    for page in pdf_reader.pages:
        document_text += page.extract_text() if page.extract_text() else ""
    return document_text

def split_pdf(document_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    document_chunks = text_splitter.split_text(document_text)

    # Prepare documents using the Document class
    documents = [Document(chunk) for chunk in document_chunks]
    return documents

def vectorstore_and_chain(documents):
    vectordb = Chroma.from_documents(documents, OpenAIEmbeddings())
    template = """ Use the following pieces of context and the instructions to answer the question at the end
    Instructions:
    1.Distill the Context: Extract the core information from the context to fully grasp the question's intent.
    2.Scour the Annual Report: Search for the answer within the specific annual report.
    3.Craft a Simple and Clear Explanation: Ensure your response is understandable to a layperson, avoiding jargon and complex terms.
    4.Incorporate Visuals (If Applicable): Enhance understanding with relevant images, charts, or graphs.
    5.Cite Page Numbers: If the answer is found, provide the specific page number(s) for reference.
    6.Address Information Gaps: If the answer is not found:
         Acknowledge the absence of supporting data and
        explain the terms involved in the question and
        Offer any relevant insights or information, even without a direct answer.
    7.Iterate for Confidence: Review and refine your response until you're confident in its clarity and accuracy.
    {context}
    Question: {question}
    Answer:"""


    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

def answers_predefined(predefined_questions,qa_chain):
    predefined_answers = []
    #st.subheader("Predefined Question Answers:")
    for question in predefined_questions:
       #st.write("Q:",question)
        result = qa_chain.invoke({"query": question},max_tokens=500)
        answer = result["result"] if result["result"] else "No Answer Found"
        #st.write("A:",answer)
        predefined_answers.append(answer)

# Create a DataFrame with predefined questions and answers
    st.subheader("Predefined Question Answers:")
    predefined_qna_data = pd.DataFrame({"Question": predefined_questions, "Answer": predefined_answers})
    display(predefined_qna_data)
# Display predefined answers


def answers_query(questions,qa_chain):
    answers = []
    for question in questions:
       if question:
        result = qa_chain.invoke({"query": question},max_tokens=500)
        answer = result["result"] #if result["result"] else "No Answer Found"
        answers.append(answer)
    else:
        answers.append("No question provided")
    while len(questions) < len(answers):
       questions.append("")
    qna_data = pd.DataFrame({"Question": questions, "Answer": answers})
    display(qna_data)
    csv_file_generation(qna_data)
# Input area for multiple questions

def display(qna_data):
     # Ensure that questions and answers lists have the same length
    #st.dataframe(qna_data)
    # Display answers for each question separately
    for index, row in qna_data.iterrows():
        if(row["Question"]!= ""):
            st.subheader(f"Answer to Question {index+1}:")
            st.write(row["Answer"])
        
def csv_file_generation(qna_data_df):
    # Output the results in CSV format
    csv = qna_data_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download QA as CSV", csv, "QA.csv", "text/csv", key='download-csv')


predefined_questions = [
    ' Any mergers',
 'Any Acquisitions',
 'What are the key revenue streams',
 'What are the key service areas',
 'YoY growth',
 'Targets for the coming year',
 'Does the company have any ESG initiative',
 'What steps have the company taken on ESG in current year and plan to take in next year',
 'What changes are the company planning to bring in to manage for next year',
 'What investments have the company planned for next year',
 'What sector is the company',
 'What is the global presence',
 'Business operations where is it spread',
 'Global footprint',
 'Which continent/countries are the biggest contributors of the revenue stream',
 'Is there dealings with any sanctioned countries /war torn countries',
 'Is there any impact on the business due to natural catastrophe',
 'Is there any impact due to climate change',
 'Is there any seasonal impact on the business',
 'Dependency on seasonal changes',
 'Above /below target executive',
 'Executive Summary - can we summarize the information'
    ]


def main():
    pdf_file = st.file_uploader("Upload a PDF document", type="pdf")
    if pdf_file is not None:
        
         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name

         docs=load_pdf(tmp_file_path)
         chunks=split_pdf(docs)
         chain=vectorstore_and_chain(chunks)
         answers_predefined(predefined_questions,chain)
         questions_input = st.text_area("Enter multiple questions (one question per line):")
         if questions_input and not all(line.isspace() for line in questions_input.splitlines()):
          questions_ip = questions_input.split('\n')
          questions = [question.strip() for question in questions_ip if question.strip()]
          answers_query(questions,chain)

main()
          





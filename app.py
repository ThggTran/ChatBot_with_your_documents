import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
import torch
from torch import cuda, bfloat16
import transformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from io import BytesIO
from docx import Document
from pdfminer.high_level import extract_text
from langchain.memory import ConversationBufferMemory
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from operator import itemgetter
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

import sys
sys.path.append('/content/drive/MyDrive/ColabNotebooks/Rag')

import html_templates

from html_templates import css, user_template , assistant_template

def read_file(file):
    file_type = file.name.split(".")[-1]
    if file_type == "txt":
      text = file.read().decode("utf-8")
    elif file_type == "docx":
      doc = Document(file)
      text = ""
      for paragraph in doc.paragraphs:  
        text += paragraph.text + "\n" 
    elif file_type == "pdf":
      bytes_io = BytesIO(file.read())
      text = extract_text(bytes_io)
    else:
      raise ValueError("Invalid file format!")
    return text



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,
        chunk_overlap=0,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.create_documents(text_splitter.split_text(text))
    return chunks

def load_embed_model():

    #Model embed
    embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

    #Cấu hình thiết bị
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )
    return embed_model


def load_llm_model():
  import os
  os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or "your-api-key"

  llm = ChatOpenAI(
      openai_api_key=os.environ["OPENAI_API_KEY"],
      model='gpt-3.5-turbo'
  )
  return llm

def generate(query,llm,vectorstore):
    prompt_template = """Please use context to answer question and always say if you are not satisfied with the answer, provide more information.

    {context}

    Question: {question}
    Answer: """

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    
    chain_type_kwargs = {"prompt": PROMPT}


    rag = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=vectorstore.as_retriever()
                                    ,chain_type_kwargs=chain_type_kwargs
    )
    return rag(query)

def create_promt_template():
  template = """Given the following conversation and a follow-up message, \
  rephrase the follow-up message to a stand-alone question or instruction that \
  represents the user's intent, add all context needed if necessary to generate a complete and \
  unambiguous question or instruction, only based on the history, don't make up messages. \
  Maintain the same language as the follow up input message.
  If you cannot answer the question with the context, please respond with 'I don't know'.
  Chat History:
  {chat_history}

  Follow Up Input: {question}
  Standalone question or instruction:"""

  # Create the prompt template
  return PromptTemplate.from_template(template)


def create_database_chroma(chunks,embed_model):

    # Create a new DB from the documents.
    # db = Chroma.from_documents(
    #     chunks, embed_model,persist_directory='/content/drive/MyDrive/ColabNotebooks/Rag/data'
    # )
    # db.persist()
    db = Chroma.from_documents(
        chunks, 
        embed_model,
        persist_directory="./chroma_db3"
    )
    db.persist()
    return db


def create_memory():
  message_history = ChatMessageHistory()

  return ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",
    chat_memory=message_history,
    return_messages=True)

def create_qa_chain(retriever):
  from langchain.prompts import ChatPromptTemplate
  # Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know'
  template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know:

  ### CONTEXT
  {context}

  ### QUESTION
  Question: {question}
  """

  prompt = ChatPromptTemplate.from_template(template)

  primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
  created_qa_chain = (
    {"context": itemgetter("question") | retriever,
     "question": itemgetter("question")
    }
    | RunnablePassthrough.assign(
        context=itemgetter("context")
      )
    | {
         "response": prompt | primary_qa_llm,
         "context": itemgetter("context"),
      }
  )

  return created_qa_chain


def history_aware_retriever(llm,retriever):
  contextualize_q_system_prompt = """Given a chat history and the latest user question \
  which might reference context in the chat history, formulate a standalone question \
  which can be understood without the chat history. Do NOT answer the question, \
  just reformulate it if needed and otherwise return it as is."""
  contextualize_q_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", contextualize_q_system_prompt),
          MessagesPlaceholder("chat_history"),
          ("human", "{input}"),
      ]
  )
  history_aware_retriever = create_history_aware_retriever(
      llm, retriever, contextualize_q_prompt
  )
  return history_aware_retriever

def question_answer_chain(llm):
  qa_system_prompt = """You are an assistant for question-answering tasks. \
  Use the following pieces of retrieved context to answer the question. \
  If you don't know the answer, just say that you don't know. \
  Use three sentences maximum and keep the answer concise.\

  {context}"""
  qa_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", qa_system_prompt),
          MessagesPlaceholder("chat_history"),
          ("human", "{input}"),
      ]
  )


  question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
  return question_answer_chain


def rag_with_mem(llm,retriever):
  contextualize_q_system_prompt = """Given a chat history and the latest user question \
  which might reference context in the chat history, formulate a standalone question \
  which can be understood without the chat history. Do NOT answer the question, \
  just reformulate it if needed and otherwise return it as is."""
  contextualize_q_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", contextualize_q_system_prompt),
          MessagesPlaceholder("chat_history"),
          ("human", "{input}"),
      ]
  )
  history_aware_retriever = create_history_aware_retriever(
      llm, retriever, contextualize_q_prompt
  )

  qa_system_prompt = """You are an assistant for question-answering tasks. \
  Use the following pieces of retrieved context to answer the question. \
  If you don't know the answer, just say that you don't know. \
  Use three sentences maximum and keep the answer concise.\

  {context}"""
  qa_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", qa_system_prompt),
          MessagesPlaceholder("chat_history"),
          ("human", "{input}"),
      ]
  )


  question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

  rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
  return rag_chain


def main():
  st.set_page_config(page_title="Chat with your file",
                      page_icon=":books:")

  st.write(css, unsafe_allow_html=True)
  

  raw_text="temp"

  st.header("Chat with your Documents:")
  with st.sidebar:
    st.subheader("Your documents")
    uploaded_files = st.file_uploader(
          "Upload your file (PDF,DOCX, TXT) here", accept_multiple_files=True)
    if st.button("Process"):
      with st.spinner("Processing"):
            if uploaded_files is not None:
              for file in uploaded_files:
                raw_text += read_file(file) + "\n\n"

  embedding_model = load_embed_model()
  llm = load_llm_model()
  text_chunks = get_text_chunks(raw_text)
  vectorstore = create_database_chroma(text_chunks,embedding_model)
  retriever = vectorstore.as_retriever(search_kwargs={"k" : 3})


  # with st.sidebar:
  #   st.subheader("See Document")
  #   st.write(vectorstore._collection.count())
  #   if st.button("Clean Cache"):
  #     with st.spinner("Processing"):
  #       vectorstore.delete_collection()

  # rag_chain = rag_with_mem(llm,retriver)

  if "messages" not in st.session_state:
    st.session_state.messages = []

  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])


  if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

  if "history_aware_retriever" not in st.session_state:
    st.session_state.history_aware_retriever = history_aware_retriever(llm,retriever)
  if "question_answer_chain" not in st.session_state:
    st.session_state.question_answer_chain = question_answer_chain(llm)
  if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = create_retrieval_chain(
                            st.session_state.history_aware_retriever,
                            st.session_state.question_answer_chain )
  
  prompt = st.chat_input("Ask a question about your documents:")
  if prompt:
    with st.chat_message("user"):
      # st.markdown(user_template.replace("{{msg}}",prompt),unsafe_allow_html=True)
      st.markdown(prompt)
      
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = st.session_state.rag_chain.invoke({"input": prompt, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.extend([HumanMessage(content=prompt), response["answer"]])
      # print(response)
      
    with st.chat_message("assistant"):
      # st.markdown(assistant_template.replace("{{msg}}",response["answer"]),unsafe_allow_html=True)
      st.markdown(response["answer"])
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
  
if __name__ == '__main__':
    main()

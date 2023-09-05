import os
import streamlit as st
import streamlit.components.v1 as stc

# File Processing Pkgs
import pandas as pd
import docx2txt
from PIL import Image


documents = []

# Fxn
@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img 




def load_documents(file):

    import os
    name, extension = os.path.splitext(file) 


    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)

    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
		
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None
	
    data = loader.load()
	
    return append_docs(data)
 
def append_docs(load):

	global documents
	documents.extend(load) 

def chunk_data(data, chunk_size=256,chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) 
    chunks = text_splitter.split_documents(data) 
    return chunks

def create_embeddings(chunks):
    from langchain.vectorstores import Chroma
    from langchain.embeddings.openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answer(vector_store, q, k=8):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.2)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

def main():
	st.title(" HR process optimization ")
	with st.sidebar:
		api_key = st.text_input('OpenAI API Key:', type='password', value=os.environ.get("OPENAI_API_KEY"))
		if api_key:
			os.environ['OPENAI_API_KEY'] = api_key
            
	uploaded_file = st.file_uploader('Upload resumes', type=['pdf', 'docx', 'txt'],accept_multiple_files=True)
	chunk_size = 512
	k = 6
	add_data = st.button('Analyse Documents', on_click=clear_history)
	if uploaded_file and add_data:
		with st.spinner('Reading, chunking and embedding file ...'):
			for file in uploaded_file:
				bytes_data = file.read()
				file_name = os.path.join('./', file.name)
				with open(file_name, 'wb') as f:
					f.write(bytes_data)
					load_documents(file_name)

			chunks = chunk_data(documents, chunk_size=chunk_size)

			vector_store = create_embeddings(chunks)
			st.session_state.vs = vector_store
			st.success('File uploaded, chunked and embedded successfully.')
			st.session_state.chunked = True 

	if 'chunked' in st.session_state:
		q = st.text_input('Ask a question about the selected resumes:')
		if  q:
			if 'vs' in st.session_state:
				vector_store = st.session_state.vs
				#st.write(f'k: {k}')
				answer = ask_and_get_answer(st.session_state.vs, q, k)
				st.text_area('Answer: ', value=answer)
				st.divider()
				if 'history' not in st.session_state:
					st.session_state.history = ''
				
				value = f'Q: {q} \nA: {answer}'
				st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
				h = st.session_state.history
				st.text_area(label='Chat History', value=h, key='history', height=400)


if __name__ == '__main__':
	main()
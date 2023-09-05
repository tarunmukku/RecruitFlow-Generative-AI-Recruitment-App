import json
import time
from tkinter import Image
from automated_email import send_email
from db_util import addItem
from jd_extraction import get_jd_values,get_ranking,get_required_skills,analyse_resume_extract,analyse_score_resume,generate_shortlist_email,generate_interview_questions
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import chroma
import os
from dotenv import load_dotenv , find_dotenv

from pages.Chatbot_Resumes import ask_and_get_answer
load_dotenv('.keys',override=True)
from pysondb import db
from Utils import extract_text_values, get_parse_jd_details, get_time_based_greeting



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


def load_document(file):
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
    return data

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



def ask_recommendations_jd(data):
    from langchain import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI
    from langchain.chains.summarize import load_summarize_chain
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    
    prompt = '''Imagine you are an expert in Analyzing job descriptions, 
    Provide suggestions in ARRAY format to improve the following job description.only output suggestions which are not incorporated in the given job description :`{text}`
    '''
    map_prompt_template = PromptTemplate(
        input_variables=['text'],
        template=prompt,
    )
    chain = LLMChain(llm=llm, prompt=map_prompt_template)
    
    output = chain.run({"text":data})
    return output
     

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens,total_tokens / 1000 * 0.0004

if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.set_page_config(
        page_title="RecruitFlow",
        page_icon="🧊",
        #layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.image('main.png')
    with st.sidebar:
        
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        api_key = st.text_input('OpenAI API Key:', type='password', value=os.environ.get("OPENAI_API_KEY"))
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        
    main = st.empty()
    js_uploadcontainer = main.container()   
    greeting = get_time_based_greeting()
    js_uploadcontainer.subheader(f"{greeting}")
    js_uploadcontainer.subheader("Welcome to RecruitFlow.")
    uploaded_file = js_uploadcontainer.file_uploader('Upload a Job Description:', type=['pdf', 'docx', 'txt'],accept_multiple_files=False)
    chunk_size = 512
    k = 6
    uploaded_resumes = js_uploadcontainer.file_uploader('Upload Resumes:', type=['pdf', 'docx', 'txt'],accept_multiple_files=True)
   
    add_data = js_uploadcontainer.button('Analyse', on_click=clear_history)

    recommendations = st.container()
    proceed = recommendations.empty()
    if uploaded_file and add_data and uploaded_resumes: # if the user browsed a file
        with st.spinner('Processing Job Description ...'):             
            bytes_data = uploaded_file.read()
            file_name = os.path.join('./', uploaded_file.name)
            with open(file_name, 'wb') as f:
                    f.write(bytes_data)
            data = load_document(file_name)
            st.session_state.data = str(data[0].page_content)
            chunks = []
            st.session_state.data_resume = list()
            for file in uploaded_resumes:
                uploaded_resumes_bytes = file.read()
                file_name_resume = os.path.join('./', file.name)
                with open(file_name_resume, 'wb') as f:
                        f.write(uploaded_resumes_bytes)
                data_resumes = load_document(file_name_resume)
                chunks_resume = chunk_data(data_resumes, chunk_size=chunk_size)
                chunks.extend(chunks_resume)
               # 
                st.session_state.data_resume+= [str(data_resumes[0].page_content)]

            tokens, embedding_cost = calculate_embedding_cost(chunks_resume)
            #st.write(f'Embedding cost: ${embedding_cost:.4f}')
            
                # creating the embeddings and returning the Chroma vector store
            vector_store = create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
            st.session_state.vs = vector_store
        

            
          
        with st.spinner('Generating recommendatoins....'):
           #add_data = st.empty()
            json_array = ask_recommendations_jd(data=st.session_state.data)
            text_values = extract_text_values(json_array)
            
            if text_values:
                recommendations.subheader("Here are some of recommendations to improve job description")
                for idx, value in enumerate(text_values, start=1):
                   recommendations.success(f"{idx}. {value}",icon="🚨",)
                st.session_state.recommendations = True   
           
    if 'recommendations' in st.session_state:
        proceed_btn = st.empty()    
        proceed_ctnr = proceed_btn
        st.markdown(
        """
        <style>
        .fixed-bottom {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            background-color: #f0f0f0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
        proceed = proceed_ctnr.button('Proceed')
        if  proceed:
                main = st.empty()    
                time.sleep(.2)
                st.session_state.analyse = True  

    if 'analyse' in st.session_state and proceed:
            recommendations = st.empty()
            time.sleep(.2)  
            proceed_btn= st.empty()  
            time.sleep(.4)
            st.title("Job Description summary")
            main_jd_summary = st.empty()
            main_jd_container = main_jd_summary.container() 
            proceed = False
            with  st.spinner('Analyzing Job Description....'):                   
                jd_values = get_jd_values(chunk_text=st.session_state.data)


                table_rows =get_parse_jd_details(jd_values)
                
                table_columns = main_jd_container.columns(2)  # Split the screen into 3 columns
                for i, (key, value) in enumerate(table_rows):
                    with table_columns[i % 2]:  # Distribute the items across the columns
                      st.write(f"**{key}:**", value)

                DESIRED_EXP_SKILLS = get_required_skills(chunk_text=st.session_state.data)
               # st.success(DESIRED_EXP_SKILLS)
            
                json_data = json.loads(DESIRED_EXP_SKILLS)

                for key, value in json_data.items():
                    expander = st.expander(label=key)
                    with expander:
                        if isinstance(value, list):
                            for val in value:
                                st.write(val)
                        else:
                            st.write(value)
            st.divider()
            st.title("Candidates Analysis")

            

            for resume in st.session_state.data_resume:
                
                resume_insights = analyse_resume_extract(json_data['Title'],resume)
                score_resume = analyse_score_resume(resume_exp=resume_insights, jd=st.session_state.data)


                candidates_data = json.loads(score_resume)


                expander_label = f"{candidates_data['candidate_name']} - Score: {candidates_data['score']}"
                st.markdown("<h3 style='font-size: 20px;'>"+expander_label+"</h3>", unsafe_allow_html=True)
                with st.expander(label="Explanation"):
                
                    st.write(candidates_data['justification'])

                new_item = {
                        "Candidate_Name": candidates_data['candidate_name'],
                        "Job_id":jd_values['job_id'][0]['job_id'],
                        "status":"Open",
                        
                        "Questions": "",
                        "responses":"" }
                ques = generate_interview_questions(jd=DESIRED_EXP_SKILLS)
                new_item["Questions"] = ques
                st.success(new_item)
                key = addItem(new_item)  
                linkkey=str(key)    
                email_body = generate_shortlist_email(candidate_name=candidates_data['candidate_name'],job_id=jd_values['job_id'][0]['job_id'],
                                role=jd_values['job_title'][0]['job_title'],link="http://localhost:8501/questions_link?id="+linkkey)
                #print(candidates_data) 
                print(get_ranking(vector_store=st.session_state.vs ,JOB_DESCRIPTION=st.session_state.data))      
            # st.success(email_body)
                #print("email address is"+str(candidates_data['email_address']))         
                send_email(body=str(email_body),receiver=str(candidates_data['email_address']))


          

           
            
                   
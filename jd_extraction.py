from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number





tools = Object(
    id="tools",
    description="""
        A tool, application, or other company that is listed in a job description.
        Analytics, eCommerce and GTM are not tools
    """,
    attributes=[
        Text(
            id="tool",
            description="The name of a tool or company"
        )
    ],
    examples=[
        (
            "Experience in working with Netsuite, or Looker a plus.",
            [
                {"tool": "Netsuite"},
                {"tool": "Looker"},
            ],
        ),
        (
           "Experience with Microsoft Excel",
            [
               {"tool": "Microsoft Excel"}
            ] 
        ),
        (
           "You must know AWS to do well in the job",
            [
               {"tool": "AWS"}
            ] 
        ),
        (
           "Troubleshooting customer issues and debugging from logs (Splunk, Syslogs, etc.) ",
            [
               {"tool": "Splunk"},
            ] 
        )
    ],
    many=True,
)




tools = Object(
    id="exp_range",
    description="""
        The work expereince range which candidate should be in as mentioned in a job description
    """,
    attributes=[
        Number(
            id="min_exp",
            description="The low end of a work experience range"
        ),
        Number(
            id="max_exp",
            description="The high end of work experience range"
        )
    ],
    examples=[
        (
            "5 to 9 years of experience in Cloud & DevOps with Full-time Bachelor’s /Master’s degree (Science or Engineering preferred)",
            [
                {"min_exp": 5, "max_exp": 9},
            ]
        ),
               (
            "You have 8+ years of professional work experience building large-scale, large-volume services & distributed apps., taking them through production and post-production life cycles",
            [
                {"min_exp": 8},
            ]
        ),
                (
            "Currently pursuing a degree in Computer Science or a related field.)",
            [
                {"min_exp": 0},
            ]
        ),

    ],
    many=True,
)

jobid_schema = Object(
 
    id="job_id",
    
 
    description="Uniquie job identification number present in job description",
    
    # Fields you'd like to capture from a piece of text about your object.
    attributes=[
        Text(
            id="job_id",
            description="unique job identifier",
        )
    ],
    

    examples=[
        ("Job ID 00000386725", [{"job_id": "00000386725"}])
    ],
    many=True
)


job_education_schema = Object(
 
    id="job_education",
    
 
    description="required education qualifications for the candidate as mentioned in job description",
    
    # Fields you'd like to capture from a piece of text about your object.
    attributes=[
        Text(
            id="job_education",
            description="required education qualifications for the candidate",
        )
    ],
    

    examples=[
        ("5 to 9 years of experience in Cloud & DevOps with Full-time Bachelor’s /Master’s degree (Science or Engineering preferred) ", 
        [{"job_education": "Full-time Bachelor’s /Master’s degree (Science or Engineering preferred)"}]),
       

    ],
    many=True

)

experience_range = Object(
    id="experience_range",
    description="""
        the desired years of  experience required as mentioned in the job description
    """,
    attributes=[
        Text(
            id="experience_range",
            description="Desired years of expereience "
        )
    ] ,  many=True,
)


salary_range = Object(
    id="salary",
    description="""
        The range of salary offered for a job mentioned in a job description
    """,
    attributes=[
        Text(
           id="salary",
            description="The low end and high end  of salary offered for a job"
        )
    ],
    many=True,
)


tools = Object(
    id="tools",
    description="""
        A tool, application, or other company that is listed in a job description.
        Analytics, eCommerce and GTM are not tools
    """,
    attributes=[
        Text(
            id="tool",
            description="The name of a tool or company"
        )
    ],
    examples=[
        (
            "Experience in working with Netsuite, or Looker a plus.",
            [
                {"tool": "Netsuite"},
                {"tool": "Looker"},
            ],
        ),
        (
           "Experience with Microsoft Excel",
            [
               {"tool": "Microsoft Excel"}
            ] 
        ),
        (
           "You must know AWS to do well in the job",
            [
               {"tool": "AWS"}
            ] 
        ),
        (
           "Troubleshooting customer issues and debugging from logs (Splunk, Syslogs, etc.) ",
            [
               {"tool": "Splunk"},
            ] 
        )
    ],
    many=True,
)


job_schema = Object(
 
    id="job_title",
    
 
    description="a designation being offered for a job and mentioned in job description",
    
    # Fields you'd like to capture from a piece of text about your object.
    attributes=[
        Text(
            id="job_title",
            description="a designation being offered for a job",
                examples=[
        ("Your role as Software Development Engineer (SDE-II) ", "Software Development Engineer (SDE-II)"),
        ("Title: data scientist ","data scientist")],
       
        ),
 salary_range,
jobid_schema,
experience_range,
 job_education_schema       
    ],
       


    many=True

)

def get_jd_values(chunk_text):
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0)
  
    chain = create_extraction_chain(llm, job_schema, encoder_or_encoder_class="json")
    
    return chain.run(
        chunk_text
    )["data"]

def get_jd_tool_values(chunk_text):
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0)
    #chain = create_extraction_chain(llm, job_schema, encoder_or_encoder_class="json")
    chain = create_extraction_chain(llm, tools, input_formatter="triple_quotes")
    return chain.run(
        chunk_text
    )["data"]

def get_required_skills(chunk_text):
    from langchain import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0)
    template = """  	Imagine that you are reviewing job adversitements.
        Based on {JOB_DESCRIPTION},extract job relevant education, experience requirements and skills in JSON format.
        """
    prompt_template = PromptTemplate(input_variables=['JOB_DESCRIPTION'],template=template)
    llmchain = LLMChain(llm=llm,prompt=prompt_template)
    return llmchain.run({
        "JOB_DESCRIPTION":chunk_text
    }) 

def analyse_resume_extract(role,resume):
    from langchain import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0)
    template = """  	Imagine that you are interviewer who is reviewing resume for {ROLE} role. You are looking
understand the relevant job experience Of a candidate based on {RESUME}. Extract job relevant
information from the resume """
    prompt_template = PromptTemplate(input_variables=['ROLE','RESUME'],template=template)
    llmchain = LLMChain(llm=llm,prompt=prompt_template)
    return llmchain.run({
        "ROLE":role,
        "RESUME":resume
    }) 

def analyse_score_resume(resume_exp,jd):
    from langchain import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0)
    template = """  	Given a job description:{JOB_DESCRIPTION}, you are reviewing candidate with following experience,education and skills: {RELEVANT_EXPERIENCE}.
 rank the candidate between 1-10 indicating fit of skills and experience to desired job description.output candidate's name and add it to  candidate_name,extract his email addresses and add it to  email_address,  rank and add it to rank and justification for rank add it to justification in JSON format"""
    prompt_template = PromptTemplate(input_variables=['RELEVANT_EXPERIENCE','JOB_DESCRIPTION'],template=template)
    llmchain = LLMChain(llm=llm,prompt=prompt_template)
    return llmchain.run({
        "RELEVANT_EXPERIENCE":resume_exp,
        "JOB_DESCRIPTION":jd
    }) 

def generate_shortlist_email(candidate_name,role,job_id,link):
    from langchain import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7)
    template = """  Imagine that you are recruiter with Name Zack from ABC Company and you have shortlisted a candidate  with name : {CANDIDATE_NAME}.
for the job for role : {ROLE} and with Job identifier :{JOB_ID} ,  generate email subject and body  to communicate that candidate is shortlisted and ask him to complete the assement provided under the link :{LINK}"""
    prompt_template = PromptTemplate(input_variables=['CANDIDATE_NAME','ROLE','JOB_ID','LINK'],template=template)
    llmchain = LLMChain(llm=llm,prompt=prompt_template)
    return llmchain.run({
        "CANDIDATE_NAME":candidate_name,
        "ROLE":role,
        "JOB_ID":job_id,
        "LINK":link
    }) 

def generate_questions(jd):
    from langchain.output_parsers import StructuredOutputParser, ResponseSchema
    from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    output_parser = StructuredOutputParser.from_response_schemas(
        response_schemas=[
            ResponseSchema(
                description="""Imagine that you are interviewer who are looking for {JOB_DESCRIPTION}.generate ten
technical questions of varying difficulty that you would ask to the candidates.
    output array of of 10 places in the following format: [
        {{ "Question": string // name of the place's}}
    ]
    """,
            ),
        ]
    )

    format_instructions = output_parser.get_format_instructions().replace(
        '"places": string', '"places": array of objects'
    )


def get_ranking(vector_store,JOB_DESCRIPTION,  k=3):

    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI


    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.3)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    q = """ Imagine that you are interviewer who are looking hire for the job:"""+JOB_DESCRIPTION+""".rank each candidate in the given context based on the sutability for the job.
    Extract Name and add it to candidate_name and email address to email_address and rank to rank key and justification for rank to explanation key.respond in JSON Array format with candidates as key 
    """


    answer = chain.run(q)
    return answer



def generate_interview_questions(jd):
    from langchain import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7)
    template = """ Imagine that you are interviewer who are looking for {JOB_DESCRIPTION}.generate ten
technical questions of varying difficulty that you would ask to the candidates in the following format: [
        {{ "Question": string // name of the place's}}
    ]
    """
    prompt_template = PromptTemplate(input_variables=['JOB_DESCRIPTION'],template=template)
    llmchain = LLMChain(llm=llm,prompt=prompt_template)
    return llmchain.run({
        "JOB_DESCRIPTION":jd
    })     
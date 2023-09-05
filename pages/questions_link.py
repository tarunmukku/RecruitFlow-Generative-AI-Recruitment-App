import streamlit as st
import json
from pysondb import getDb

todo_db = getDb('questions.json')

res = st.experimental_get_query_params()
st.write(res['id'][0])
json_data = todo_db.getById(pk=res['id'][0])



# Extracting questions from JSON data
questions = json.loads(json_data['Questions'])

# Streamlit app
st.title('Questionnaire Submission')
st.write(f"Candidate: {json_data['Candidate_Name']}")
st.write(f"Job ID: {json_data['Job_id']}")
st.write("Please answer the following questions:")

if json_data['status'] == "Open":
    responses = {}
else:
   responses = json_data['responses']     

for q in questions:
    question = q['Question']
    if json_data['status'] == "Open":
        response = st.text_area(question)
        responses[question] = response
    else:
        st.text_area(question,value=responses[question],disabled=True)

# Submit button

if json_data['status'] == "Open":
    if st.button('Submit'):
        json_data['responses'] = responses
        st.write("Submission Successful!")
        json_data['status'] = "Submitted"
else:
    st.button('Submit',disabled=True)

# Display JSON data
st.write("Updated JSON data:")
todo_db.updateById(pk=res['id'][0],new_data=json_data)

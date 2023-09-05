import streamlit as st
import pandas as pd
import json
from pysondb import getDb
# Sample JSON data
todo_db = getDb('questions.json')
json_data = todo_db.getAll()
# Extract relevant data from JSON
data = []
for item in json_data:
    data.append({
        'Name': item['Candidate_Name'],
        'Job_ID': item['Job_id'],
        'ID': item['id'],
        'Status': item['status'],
        'Report':item['id']
    })

# Create a DataFrame
df = pd.DataFrame(data)

# Define cell background color function
def set_cell_color(value):
    if value == 'Open':
        return 'background-color: orange'
    elif value == 'Submitted':
        return 'background-color: green'
    else:
        return ''
def make_clickable(val):
   
    link = f'<a href="http://localhost:8501/questions_link?id={val}" target="_blank">Link</a>'
   
    return link

# Apply cell background color to DataFrame
styled_df = df.style.applymap(set_cell_color, subset=['Status'])

df['Report'] = df['Report'].apply(make_clickable)

# Streamlit app
st.title('Candidate Status Table')
#df = st.dataframe(styled_df,use_container_width=True)   #.to_html(escape=False, index=False)
df = df.to_html(escape=False)
#st.dataframe(styled_df,use_container_width=True)
#st.write(, unsafe_allow_html=True)
def make_clickable(link):

    return f'<a target="_blank" href= "http://localhost:8501/questions_link?id={link}">{link}</a>'

# link is the column with hyperlinks

st.write(df, unsafe_allow_html=True)
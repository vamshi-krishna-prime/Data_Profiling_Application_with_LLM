# import packages/libraries
import os
import warnings
import sys

import streamlit as st
import streamlit.components.v1 as components
from streamlit_chat import message, NO_AVATAR

from streamlit.runtime.legacy_caching.hashing import _CodeHasher
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx as get_report_ctx
from streamlit.web.server import Server

import google.generativeai as genai 

from pathlib import Path 
import sqlite3
import os
import math
import time
from time import sleep
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from IPython.display import HTML
from pathlib import Path

# importing Ydata Profiling
from nbformat import write
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO


# Set page configuration
st.set_page_config(
    page_title="DPT", # String or None. Strings get appended with "• Streamlit". 
    page_icon=":bar_chart:", # String, anything supported by st.image, or None.
    layout="wide", # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="expanded", # Can be "auto", "expanded", "collapsed"
)

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
genai_version = genai.__version__
## load the API key from the environment variable
load_dotenv()
gemini_api_key = os.getenv("GEMINI_KEY")
genai.configure(api_key = gemini_api_key)


# Set up the model 
generation_config = { "temperature": 0.4, 
                     "top_p": 1, 
                     "top_k": 32, 
                     "max_output_tokens": 4096, } 

safety_settings = [ { "category": "HARM_CATEGORY_HARASSMENT", 
                     "threshold": "BLOCK_MEDIUM_AND_ABOVE" }, 
                    { "category": "HARM_CATEGORY_HATE_SPEECH", 
                      "threshold": "BLOCK_MEDIUM_AND_ABOVE" }, 
                    { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
                     "threshold": "BLOCK_MEDIUM_AND_ABOVE" }, 
                    { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", 
                     "threshold": "BLOCK_MEDIUM_AND_ABOVE" } ]

model = genai.GenerativeModel(model_name = "gemini-pro", 
                              generation_config = generation_config, 
                              safety_settings = safety_settings)


def read_sql_query(sql, db): 
    conn = sqlite3.connect(db) 
    cur = conn.cursor() 
    cur.execute(sql) 
    rows = cur.fetchall() 
    for row in rows: 
        print(row) 
    conn.close()


def run_query(q, db):
    with sqlite3.connect(db) as conn:
        return pd.read_sql(q,conn)


def run_command(c, db):
    with sqlite3.connect(db) as conn:
        conn.isolation_level = None
        conn.execute(c) 


def show_tables(db):
    q = '''
        SELECT
            name,
            type
        FROM sqlite_master
        WHERE type IN ("table","view");
        '''
    return run_query(q, db)


def database_details(db):
    q = '''
        SELECT *
        FROM sqlite_master
        '''
    return run_query(q, db)


db = 'chinook.db'

table_info = show_tables(db)
# st.write(table_info)
# result = run_query('SELECT * FROM Album LIMIT 10;', db=database)
# st.write(result)

database_string = ""
database_info = database_details(db)
for line in database_info['sql'].to_list():
    database_string += ';' + str(line)

# -------------------------------------
prompt_parts_1 = [f"""You are an expert in converting English questions to SQL code! \
                  The SQL database has the name {db} and has the following information - {table_info}.\n\nFor example,\nExample 1 - Find the most purchased song genre?, the SQL command will be something like \
                  this\n``` SELECT g.Name AS genre_type, COUNT(*) AS num_purchases FROM Genre g JOIN Track t ON g.GenreId = t.GenreId JOIN InvoiceLine il ON t.TrackId = il.TrackId GROUP BY 1 ORDER BY 2 DESC;\n```\n\nExample 2 - \
                  Classify the customers domain based on business orientation and on which section does the company should focus their customer services?, the SQL command will be something like this\n```\nWITH customer_type AS (SELECT CASE WHEN Company IS NULL THEN 1 ELSE 0 END AS domestic_customer, CASE WHEN Company IS NULL THEN 0 ELSE 1 END AS business_customer FROM Customer) SELECT SUM(domestic_customer) AS domestic_customers, SUM(business_customer) AS business_customers FROM customer_type;\n```\n\nExample 3 \
                  - Identify the best sales agent and analyze his annual performance, the SQL command will be something like this\n```\nWITH best_sales_agent AS (SELECT e.FirstName,  e.LastName,  CASE WHEN COUNT(*) IS NULL THEN "0" ELSE COUNT(*) END AS assists FROM Employee e JOIN Customer c  ON e.EmployeeId = c.SupportRepId WHERE e.Title = "Sales Support Agent" GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 1) SELECT  e.FirstName,  e.LastName,  STRFTIME('%Y', i.InvoiceDate) AS year, COUNT(*) AS sales FROM Employee e JOIN Customer c  ON e.EmployeeId = c.SupportRepId JOIN Invoice i  ON c.CustomerId = i.CustomerId JOIN best_sales_agent bsa  ON e.FirstName = bsa.FirstName  AND e.LastName = bsa.LastName GROUP BY 1, 2, 3 ORDER BY 3, 1, 2 ;\n```\n\
                  \nDont include ``` and \n in the output""", ]

prompt_parts_2 = [f"""You are an expert in converting English questions to SQL code! \
                  The SQL database has the name {db} and has the following information - {table_info}. Remove any text in the response output such as 'sql', except the sql query.""", ]

prompt_parts_3 = [f"""You are an expert in converting English questions to SQL code! \
                  The SQL database has the name {db} and has the following information - {database_string}. Remove any text in the response output such as 'sql', except the sql query.""", ]


def generate_gemini_response(question, input_prompt, database): 
    prompt_parts = [input_prompt, question] 
    response = model.generate_content(prompt_parts) 
    return response.text
    # output = read_sql_query(response.text, database) 
    # return output
# -------------------------------------


# st.title("Database ChatBot")
# col1, col2 = st.columns([3, 1])
# with col2:
#     st.button("Reset Chat", on_click=None)


# Display user prompts and responses
# prompt = st.chat_input("Enter the prompt")
# if prompt:
#     st.write(f"User prompt: {prompt}")
#     try:
#         response = generate_gemini_response(prompt, prompt_parts_3[0], db)
#         response = response.replace('sql', '').replace('```','  ')
#         output = run_query(response, db) 
#         message = st.chat_message("assistant")
#         message.write("Hello user")
#         message.write(f"SQL QUERY:\n")
#         message.code(response)
#         message.write(f"RESULT:\n")
#         message.write(output)
#         message.write('-'*50)
#     except:
#         response = generate_gemini_response(question=prompt, input_prompt='', database=None)
#         response = response.replace('sql', '').replace('```','  ')
#         output = run_query(response, db) 
#         message = st.chat_message("assistant")
#         message.write("Hello user")
#         message.write(f"RESULT:\n")
#         message.write(response)
#         message.write('-'*50)


# prompt = st.chat_input("Enter the prompt")
# if prompt:
#     st.write(f"User prompt: {prompt}")
#     if col1.button('Ask Database'):
#         response = generate_gemini_response(prompt, prompt_parts_3[0], db)
#         response = response.replace('sql', '').replace('```','  ')
#         output = run_query(response, db) 
#         message = st.chat_message("assistant")
#         message.write("Hello user")
#         message.write(f"SQL QUERY:\n")
#         message.code(response)
#         message.write(f"RESULT:\n")
#         message.write(output)
#         message.write('-'*50)

#     if col2.button('Ask GPT'):
#         response = generate_gemini_response(question=prompt, input_prompt='', database=None)
#         message = st.chat_message("assistant")
#         message.write("Hello user")
#         message.write(f"RESULT:\n")
#         message.write(response)
#         message.write('-'*50)


# prompt = st.chat_input("Enter the prompt")
# if prompt:
#     # st.write(f"User prompt: {prompt}")
#     message = st.chat_message("user")
#     message.write(f"User prompt: \n{prompt}")
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         st.button("Ask Database", on_click=scenario_1)
#     with col2:
#         st.button("Ask GPT", on_click=scenario_2)

# streamlit page
def main():
    pages = {
        "Home": home_page,
        # "Chatbot": llm_chatbot,
        # "Databot": database_chatbot,
        # "beta": beta,
        "Data Profiler": pandas_profiling,
        "BI Visualizer": bi_tool,
        "Insight Generator": data_bot,
        "Database Wisperer": sql_bot,
    }

    # st.sidebar.title(":bookmark_tabs: Navigation")
    image1 = Image.open("images/Indegene_Logo_png.png")
    st.sidebar.image(image1, width=200)
    st.sidebar.write('')
    page = st.sidebar.radio("Task selection:", tuple(pages.keys()),)
    pages[page]()


def home_page():
    st.title("DATA PROFILING APPLICATION")
    st.markdown('`In-house application to analyse the quality and integrity of the data.`')
    st.write('----')
    st.markdown('<p style="text-align: justify;">The quality  and reliability of the prediction \
                made by any analytical tool or machine learning model is directly impacted by the \
                quality of the data used.</p>', unsafe_allow_html=True)
    
    st.session_state.setdefault(
        'df_uploaded', 
        []
    )

    st.title("DATABASE CHATBOT APPLICATION")
    st.markdown('`In-house application to generate SQL and explore results using Natural Language.`')
    st.write('----')
    st.markdown('<p style="text-align: justify;">Chat-based web application that can talk to the Postgres \
                database through the Langchain framework. This chatbot will be able to understand natural \
                language and use it to create and run SQL queries. Enhanced the chatbots ability to \
                execute the SQL queries to fetch the results. Supports multiple languages.</p>', 
                unsafe_allow_html=True)


def pandas_profiling():
    st.title("📝 Data Profiler")
    if not st.session_state.df_uploaded :
        try:
            uploaded_file = st.file_uploader("Upload the data:", type=("csv", "xls", "xlsx", "txt", "md")) 
            try:
                uploaded_file = pd.read_excel(uploaded_file)
            except: 
                uploaded_file = pd.read_csv(uploaded_file)
            # st.dataframe(uploaded_file, use_container_width=True)
            # print(uploaded_file.shape[0])
            # if not uploaded_file.shape[0]:
                # uploaded_file = pd.read_csv(uploaded_file)
            uploaded_file = pd.DataFrame(uploaded_file)
            st.write('Upload successful')
            st.write('----')
            
        except:
            st.write('Upload a file')
            uploaded_file = pd.DataFrame()
            pass

        if uploaded_file.shape[0]:
            profile = ProfileReport(uploaded_file, title="New Data for profiling", explorative=True)
            st.subheader("Detailed Report of the Data Used:")
            st.dataframe(uploaded_file, use_container_width=True)
            st_profile_report(profile)
            
            def create_report():
                profile.to_file("report.html")
            
            st.markdown('## Download the report')
            st.button('Create Report', on_click=create_report)

            def download_report():
                with open("report.html", 'r') as f:
                    text = f.read()
                return text
            
            # report_file = download_report()

            st.download_button(label="Download Report",
                                data=download_report(),
                                file_name="test_report.html")
    
    else:
        st.write("You did not upload the new file")


def bi_tool():
    st.title("📝 Data Profiling")
    if not st.session_state.df_uploaded :
        try:
            uploaded_file = st.file_uploader("Upload the data:", type=("csv", "xls", "xlsx", "txt", "md")) 
            try:
                uploaded_file = pd.read_excel(uploaded_file)
            except: 
                uploaded_file = pd.read_csv(uploaded_file)
            uploaded_file = pd.DataFrame(uploaded_file)
            st.write('Upload successful')
        except:
            st.write('Upload a file')
            uploaded_file = pd.DataFrame()
            pass

        st.write('----')
        st.title("BI tool for data exploration and visualization:")
        # Establish communication between pygwalker and streamlit
        init_streamlit_comm()
        
        # Get an instance of pygwalker's renderer. You should cache this instance to effectively prevent the growth of in-process memory.
        @st.cache_resource
        def get_pyg_renderer() -> "StreamlitRenderer":
            # df = pd.read_csv("https://kanaries-app.s3.ap-northeast-1.amazonaws.com/public-datasets/bike_sharing_dc.csv")
            # When you need to publish your app to the public, you should set the debug parameter to False to prevent other users from writing to your chart configuration file.
            return StreamlitRenderer(uploaded_file, spec="./gw_config.json", debug=False)
        
        renderer = get_pyg_renderer()
        # Render your data exploration interface. Developers can use it to build charts by drag and drop.
        renderer.render_explore()

    else:
        st.write("You did not upload the new file")


def llm_chatbot():
    prompt = st.chat_input("Enter the prompt")
    if prompt:
        message = st.chat_message("human")
        message.write(f"User prompt:")
        message.write(prompt)
        scenario_1(prompt=prompt)


def database_chatbot():
    prompt = st.chat_input("Enter the prompt")
    if prompt:
        message = st.chat_message("user")
        message.write(f"User prompt:")
        message.write(prompt)
        scenario_2(prompt=prompt, prompt_prefix=prompt_parts_3[0], db=db)


def scenario_1(prompt):
    response = generate_gemini_response(question=prompt, input_prompt='', database=None)
    message = st.chat_message("ai")
    message.write("Hello user")
    message.write(f"RESULT:\n")
    message.write(response)


def scenario_2(prompt, prompt_prefix, db):
    response = generate_gemini_response(question=prompt, input_prompt=prompt_prefix, database=db)
    response = response.replace('sql', '').replace('```','  ')
    message = st.chat_message("assistant")
    message.write("Hello user")
    message.write(f"SQL QUERY:\n")
    message.code(response)
    # execute SQL query response
    with st.expander("Execute the SQL query"):
        # st.button('Run SQL', on_click=run_sql, args=[message, response, db])
        run_sql(message, response, db)


def run_sql(message, response, db):
    output = run_query(response, db) 
    message.write(f"QUERY RESULT:\n")
    message.write(output)


def scenario_1_beta(prompt):
    response = generate_gemini_response(question=prompt, input_prompt='', database=None)
    return response


def scenario_2_beta(prompt, prompt_prefix, db):
    response = generate_gemini_response(question=prompt, input_prompt=prompt_prefix, database=db)
    response = '```\n' + response.replace('sql', '') + '\n```'
    return response


def beta():
    if 'generated' not in st.session_state:
      st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    def on_input_change():
        user_input = st.session_state.user_input
        st.session_state.past.append(user_input)
        response = scenario_2_beta(prompt=user_input, prompt_prefix=prompt_parts_3[0], db=db)
        st.session_state.generated.append(response)

    def on_btn_click():
        del st.session_state.past[:]
        del st.session_state.generated[:]

    st.session_state.setdefault(
        'past', 
        []
    )

    st.session_state.setdefault(
        'generated', 
        []
    )

    st.title("Chat placeholder")

    chat_placeholder = st.empty()

    with chat_placeholder.container():    
        for i in range(len(st.session_state['generated'])):                
            message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
            message(
                st.session_state['generated'][i], 
                key=f"{i}", 
                allow_html=True,
                avatar_style=NO_AVATAR
            )
        st.button("Clear message", on_click=on_btn_click)

    with st.container():
        st.text_input("User Input:", on_change=on_input_change, key="user_input")



def sql_bot():
    if 'generated' not in st.session_state:
      st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    def on_input_change():
        user_input = st.session_state.user_input
        st.session_state.past.append(user_input)
        response = scenario_2_beta(prompt=user_input, prompt_prefix=prompt_parts_3[0], db=db)
        st.session_state.generated.append(response)

    def on_btn_click():
        del st.session_state.past[:]
        del st.session_state.generated[:]

    st.session_state.setdefault(
        'past', 
        []
    )

    st.session_state.setdefault(
        'generated', 
        []
    )

    st.title("Chat placeholder")

    chat_placeholder = st.empty()

    with chat_placeholder.container():    
        for i in range(len(st.session_state['generated'])):                
            message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
            if st.session_state['generated'][i]['type'] == 'code':
                message2 = st.chat_message("assistant")
                # message2.code(st.session_state['generated'][i]['data'])
                # query = st.session_state['generated'][i]['data'].split("-"*30)[0]
                # table = st.session_state['generated'][i]['data'].split("-"*30)[1]
                query = st.session_state['generated'][i]['data']
                message2.markdown('###### SQL QUERY:\n')
                message2.code(query)
                message2.markdown('###### QUERY RESULT:\n')
                # TESTDATA = StringIO("Index\t" + table)
                # df_table = pd.read_csv(TESTDATA, sep="\t")
                # message2.code(df_table)
                result = run_query(query, db) 
                message2.dataframe(result)
            else: 
                message(
                    st.session_state['generated'][i]['data'], 
                    key=f"{i}", 
                    allow_html=True,
                    avatar_style="assistant"
                )
        st.button("Clear message", on_click=on_btn_click)

    def on_input_change_2():
        user_input = st.session_state.user_input
        st.session_state.past.append(user_input)
        response = scenario_2_beta(prompt=user_input, prompt_prefix=prompt_parts_3[0], db=db)
        response = str(response.replace('sql', '').replace('```','  '))
        # output = run_query(response, db) 
        # output = '# SQL QUERY:\n' + str(response) + '\n' + str('-'*30) + '\n\n# RESULT:\n\n' + str(output)
        # st.session_state.generated.append({'type': 'code', 'data': f'{output}'})
        st.session_state.generated.append({'type': 'code', 'data': f'{response}'})
        st.session_state["user_input"] = ""
    with st.container():
        output = st.text_input("User Input:", on_change=on_input_change_2, key="user_input")


def data_bot():
    if not st.session_state.df_uploaded :
        try:
            uploaded_file = st.file_uploader("Upload the data:", type=("csv", "xls", "xlsx", "txt", "md")) 
            try:
                uploaded_file = pd.read_excel(uploaded_file)
            except: 
                uploaded_file = pd.read_csv(uploaded_file)
            uploaded_file = pd.DataFrame(uploaded_file)
            st.write('Upload successful')
        except:
            st.write('Upload a file')
            uploaded_file = pd.DataFrame()
            pass
    st.write(uploaded_file)

    json_df = uploaded_file.to_json() 


    prompt_parts_4 = [f"""You are an expert in data analytics and drawing insights from the data. \
                  Use the following information - {uploaded_file}. Let the user know if the question is not relatd to the data uploaded.""", ]
    prompt_parts_5 = [f"""You are an expert in data analytics and drawing insights from the data. \
                  Use the following information - {json_df}. """, ]

    prompt = st.chat_input("Enter the prompt")
    if prompt:
        message = st.chat_message("user")
        message.write(f"User prompt:")
        message.write(prompt)
        scenario_3(prompt=prompt, prompt_prefix=prompt_parts_5[0], db=db)


def scenario_3(prompt, prompt_prefix, db):
    response = generate_gemini_response(question=prompt, input_prompt=prompt_prefix, database=None)
    message = st.chat_message("ai")
    message.write("Hello user")
    message.write(f"REPLY:\n")
    message.write(response)


if __name__ == "__main__":
    main()
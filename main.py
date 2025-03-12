import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from pygwalker.api.streamlit import StreamlitRenderer
from langchain_groq.chat_models import ChatGroq
from pandasai import SmartDataframe

# Configure the page
st.set_page_config(page_title="Exploratory Data Analytics App", layout="wide")

# Main header
st.title("Exploratory Data Analytics App")

# Sidebar: File uploader for inventory data
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Process uploaded file
data = None
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

# API Key and LLM configuration for Chat with your Data
api_key = st.secrets["api_keys"]["GROQ_API_KEY"]
llm = ChatGroq(
    model_name="deepseek-r1-distill-qwen-32b",
    api_key=api_key,
    temperature=0.6
)

# Create tabs: Data Profiling, Data Exploration, Chat with your Data
tabs = st.tabs(["Data Profiling", "Data Exploration", "Chat with your Data"])

# Data Profiling Tab using ydata-profiling with a click button to generate the report
with tabs[0]:
    st.header("Data Profiling")
    if data is not None:
        if st.button("Generate Profiling Report"):
            with st.spinner("Generating data profiling report..."):
                profile = ProfileReport(data, title="Data Profiling Report", explorative=False)
                # Embed the interactive HTML report using Streamlit components
                st.components.v1.html(profile.to_html(), height=800, scrolling=True)
        else:
            st.info("Click the button above to generate the profiling report.")
    else:
        st.info("Please upload a CSV file from the sidebar to generate the profiling report.")

# Data Exploration Tab using pygwalker
with tabs[1]:
    st.header("Data Exploration")
    if data is not None:
        pyg_app = StreamlitRenderer(data)
        pyg_app.explorer()
    else:
        st.info("Please upload a CSV file from the sidebar to explore your data.")

# Chat with your Data Tab using PandasAI
with tabs[2]:
    st.header("Chat with your Data")
    if data is not None:
        with st.expander("🔎 Dataframe Preview"):
            st.write(data.head(5))
        query = st.text_area("🗣️ Chat with Dataframe")
        st.write(query)
        if query:
            df = SmartDataframe(data, config = {"llm":llm})
            result = df.chat(query)
            st.write(result)
    else:
        st.info("Please upload a CSV file from the sidebar to chat with your data.")


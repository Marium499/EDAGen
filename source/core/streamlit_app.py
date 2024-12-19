import streamlit as st
from openai import OpenAI
import pandas as pd
from eda_prog import eda_main
from PIL import Image

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    e2b_api_key = st.text_input("E2B API Key", key="e2b_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("EDAGEN: An Agentic RAG for Autonomous Exploratory Data Analysis")

# if "messages" not in st.session_state:
#     st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# if "data" not in st.session_state:
#     st.session_state["data"] = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

uploaded_file = st.file_uploader("Choose a CSV file", type=("csv"))

if uploaded_file:
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # save file to path
    filepath = "../data/"+uploaded_file.name
    data = pd.read_csv(uploaded_file)
    data.to_csv(filepath)
    eda_analysis, metadata_summary = eda_main(api_key=openai_api_key, e2b_key=e2b_api_key, file_path=filepath)
    st.chat_message("assistant").write(metadata_summary)

    st.chat_message("assistant").write("Here is some detailed EDA of the dataset...")
    for item in eda_analysis:
        st.chat_message("assistant").write(item["question"])
        ans = item["code_answer"]
        st.chat_message("assistant").write(ans["text_output"].replace('item:', ''))
        if ans["images"] is not None:
            with st.chat_message("assistant"):
                for img in ans["images"]:
                    image_path = img["image_path"]
                    # image = Image.open(image_path)
                    st.image(image_path, caption='Image output')
        ans_analysis = item["answer_analysis"]
        st.chat_message("assistant").write(ans_analysis)
        

    




    # client = OpenAI(api_key=openai_api_key)
    
    # if uploaded_files:
        # prompt = str(" and provided context below, answer the user query: "+prompt)
        # for file in uploaded_files:
            # data = pd.read_csv(file)
    # st.session_state.data.append(data)
        #     prompt = str(f"""\n\n<\data> {data}\n <\data>  """+prompt)
        # prompt = str(" Based on the relevant data: "+prompt)
        # st.session_state.messages.append({"role": "user", "content": prompt})
        # st.chat_message("user").write(prompt)
    # response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    # msg = response.choices[0].message.content
    # st.session_state.messages.append({"role": "assistant", "content": msg})
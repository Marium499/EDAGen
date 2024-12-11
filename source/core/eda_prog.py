import pandas as pd
from agents import ChatLLMOpenAI, CodeLLMOpenAI
from ci_agent import get_data_summary
from cc_agent import identify_dataset_topic
from cc_agent import identify_dataset_type
from cc_agent import generate_de_prompt
from utils import random_sample_dataframe
from utils import extract_bullets
from de_agent import generate_eda_questions
from ci_agent import process_eda_questions
from utils import process_outputs
from de_agent import analyse_eda_output
from cc_agent import summarize_metadata


def eda_main(api_key, e2b_key, file_path):

    role_prompts = pd.read_csv("../data/role_prompts.csv", header=0)
    cc_prompt = role_prompts.loc[role_prompts["function"]=="init_cc_agent", "prompt"].values[0]
    cc_prompt = """You are part of an agentic RAG application that performs EDA on datasets. The following text describes your role as an LLM helper agent. Understand your roles and responsibilities then answer the proceeding prompts as per the instructions. """+ cc_prompt
    ci_agent = CodeLLMOpenAI(api_key, e2b_key, temp=0.7)
    sbx_file_path = ci_agent.upload_file_sandbox(file_path)

    metadata = get_data_summary(ci_agent, sbx_file_path)

    cc_agent = ChatLLMOpenAI(api_key, role_prompt=cc_prompt)

    data_topic = identify_dataset_topic(cc_agent, metadata = metadata)
    dataset_type = identify_dataset_type(cc_agent, metadata = metadata)
    de_prompt = generate_de_prompt(cc_agent, domain=data_topic)
    

    df_sample = random_sample_dataframe(file_path, n=15)
    metadata[0]["dataset topic and domain"]= data_topic
    metadata[0]["dataset type (Time-series,Observational/cross-sectional,Labelled/supervised)"]=dataset_type


    metadata[0]["Random subset of the dataset to understand its structure"]= df_sample

    de_agent = ChatLLMOpenAI(api_key, de_prompt)
    questions = generate_eda_questions(de_agent, df_sample, metadata)
    q_list = extract_bullets(questions, pat = "numbered_bullet_pattern")
    qa_list = process_eda_questions(ci_agent, q_list, sbx_file_path, metadata) # dict_keys = ["question", "answer"]

    for item in qa_list:

        answer_dict = process_outputs(item["answer"]) #dict_keys = ["text_output", "image_output", "image_base64"]
        item["answer"] = answer_dict

    analysis_list = analyse_eda_output(de_agent,qa_list) # dict_keys= ["question", "code_answer", "answer_analysis"}
    summarizer_agent = ChatLLMOpenAI(api_key)
    metadata_summary = summarize_metadata(summarizer_agent, metadata)


    return analysis_list, metadata_summary



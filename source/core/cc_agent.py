# import openai
# from llama_index.agent.openai import OpenAIAgent
# import os
# from langchain.memory import ConversationBufferMemory
# from llama_index import GPTVectorStoreIndex
# from langchain_openai import ChatOpenAI
import logging
# from io import BytesIO
import sys

# from typing import List, Dict, Union

# from langchain.chains import LLMChain
# from langchain.memory import ConversationBufferMemory
# from langchain_core.messages import SystemMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.prompts.chat import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     MessagesPlaceholder,
# )
# from langchain_openai import ChatOpenAI

from agents import ChatLLMOpenAI

# import pandas as pd

logger = logging.getLogger("AppLogger")




# prompt = ChatPromptTemplate(
#     [
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessagePromptTemplate.from_template("{text}"),
#     ]
# )

        


# def init_cc_agent(user_api_key: str) -> ChatLLMOpenAI: 

#     """
#     Initialize the central coordinator agent with langchain memory and user provided api key.
#     Assign role to agent via prompt engineering
#     Args:
#         user_api_key (str): The api key entered by the user
#     Returns:
#         ChatLLMOpenAI: the cc agent  
    
#     Raises:
#         Exception: Logs and exits the program if an error occurs during execution.

#     """
    
    # try:
        
#         # Initialize agent role
#         agent_prompt = """You are the central coordinator and manager agent in a multi-agent system for Exploratory Data Analysis (EDA).
#             Your Role:
#             You are responsible for receiving all user queries and the dataset. Based on the query, decide whether to:
#             Respond directly to general questions about the dataset, its description, or its purpose.
#             Delegate specific tasks to other agents such as the Code Interpreter agent or the Domain Expert agent.
#             Responsibilities:
#             Use your knowledge of data analysis to answer general questions or provide insights about the dataset when appropriate.
#             Categorize user queries into appropriate types (e.g., code-related, domain-related, or general).
#             Route specialized queries to the most relevant agent and ensure the response is consistent and contextually accurate.
#             Collaborate with other agents by initiating role-specific prompts that clearly explain their tasks.
#             Analyze datasets to understand their content, structure, and relationships. For example:  
#                 - If given multiple files, identify the main dataset containing rows and columns of primary interest.  
#                 - Detect and differentiate metadata or supporting files from the core dataset.  
#             Guidelines:
#             Use the dataset and the ongoing conversation as context for all decisions.
#             Be flexible in understanding diverse user queries and adapt your responses or routing accordingly.
#             Maintain a smooth and coherent user experience by managing query flow effectively.
#             Your overarching goal is to facilitate seamless interaction and collaboration between agents and provide meaningful insights to the user."""

#         # Initialize agent
#         cc_agent = ChatLLMOpenAI(user_api_key=user_api_key, role_prompt=agent_prompt)
        
#         return cc_agent
#     except Exception as e:
#         logger.error("An error has occurred in initializing cc agent: ", e)
#         # logging.info(e)

#         sys.exit()

def identify_dataset_topic(agent: ChatLLMOpenAI, metadata) -> str:
    """
    Identify the dataset topic and context
    Args:
    agent: the AI agent to be used for the task
    files: List of dictionary containing `name` (str) and `content` (BytesIO) attribute
    metadata: List of dictionary containing `attribute` and `data` elements

    Returns:
    dataset_topic (str): the topic of dataset

    Raises:
        Exception: Logs and exits the program if an error occurs during execution.
    """

    

    try:
        prompt = f"""
            I have metadata about a dataset that includes column names, a data summary, descriptive statistics, and information about unique values, qualitative and quantitative measurements. Using this metadata, I need your help to infer the likely topic domain of the dataset.

            Here is the metadata for reference: 
            <metadata>{metadata}</metadata>

            Based on your knowledge and, if necessary, browsing the internet, analyze the provided metadata to deduce the specific topic or niche that the dataset represents. Your response should be concise and consist of just a few words describing the topic or purpose of the dataset. Please aim for specificity to identify the dataset's niche or primary purpose.
            """

        response = agent.prompt_agent(prompt)
        return response
    except Exception as e:
        logger.error("An error has occured in identifying dataset topic: ", e)
        sys.exit()

def generate_de_prompt(agent: ChatLLMOpenAI, domain: str, prompt: str = "") -> str:

    """
    Generates a detailed role explanation prompt for the Domain Expert (DE) agent.

    This function interacts with a CC (Coordinator) agent to create a detailed prompt 
    that defines the responsibilities, capabilities, and context for the Domain Expert 
    agent in the given domain or topic. The prompt ensures the DE agent can effectively 
    guide coders and perform Exploratory Data Analysis (EDA) tasks.

    Args:
        agent (ChatLLMOpenAI): The CC agent responsible for generating the DE agent's prompt.
        domain (str): The domain or topic of the user's data, providing context for the DE agent.
        prompt (str): An optional base prompt to customize the CC agent's task (default is "").

    Returns:
        str: The generated prompt for the Domain Expert agent.

    Raises:
        Exception: Logs the error and exits the program if an error occurs during execution.
    """
    try:
        prompt = f"""Create a detailed prompt for an AI agent that defines its role as a Domain Expert specializing in {domain}, 
            with expertise in data science and exploratory data analysis (EDA). The agent should also possess the ability to understand and analyze data visualizations to derive meaningful insights.

            Context: The agent operates as a domain expert in the given topic area ({domain}) and guides a team of coders.
            Responsibilities:
            Define data science and EDA tasks relevant to the dataset and the topic.
            Specify appropriate visualizations and analysis techniques to explore the data.
            Interpret the analysis results and visualizations provided by the coders to extract insights.
            Communicate findings clearly, highlighting patterns, trends, or anomalies in the dataset.
            Capabilities:
            Analyze datasets to understand their content, structure, and relationships. For example:  
                - If given multiple files, identify the main dataset containing rows and columns of primary interest.  
                - Detect and differentiate metadata or supporting files from the core dataset.
            Understand and interpret various types of data visualizations (e.g., scatter plots, histograms, box plots).
            Utilize domain-specific knowledge ({domain}) to guide data exploration and analysis.
            Collaborate effectively with coders, ensuring tasks are well-defined and align with the analytical goals.
            The domain expert agent should prioritize accuracy, clarity, and actionable insights when reviewing and interpreting the results from the coders."""
        response = agent.prompt_agent(prompt)
        return response
    except Exception as e:
        logger.error("An error has occurred in the de prompt generation task: ", e)
        sys.exit()

def identify_dataset_type(agent: ChatLLMOpenAI, metadata) -> str:
    """
    Identifies the type of dataset (observational/cross-sectional or labelled/supervised).

    This function uses an AI agent to analyze the provided dataset files and metadata 
    to determine whether the dataset is observational/cross-sectional or labelled/supervised. 
    The agent considers the structure, attributes, and descriptions of the dataset.

    Args:
        agent (ChatLLMOpenAI): The AI agent responsible for analyzing the dataset.
        files (List[Dict[str, BytesIO]]): A list of file metadata, where each file is represented 
                                          as a dictionary with keys:
                                          - "name" (str): The name of the file.
                                          - "content" (BytesIO): The file content.
        metadata (List[Dict[str, str]]): A list of dictionaries containing additional dataset 
                                         metadata, including attributes and data summaries.

    Returns:
        str: The identified dataset type, either "observational/cross-sectional" or "labelled/supervised".

    Raises:
        Exception: Logs the error and exits the program if an error occurs during execution.
    """
    try:
        prompt = f"""
        Based on the data description, columns, and the qualitative and quantitative attributes measured in the dataset provided below: 
        <metadata>{metadata}</metadata> 

        Analyze the dataset type carefully, considering the following:
        1. **Time-series data** typically includes a column with timestamps or dates where values are recorded sequentially over time. It focuses on trends or changes across time intervals.
        2. **Observational/cross-sectional data** involves measurements taken at a single point or over a short period, focusing on relationships or distributions rather than temporal trends.
        3. **Labelled/supervised data** usually includes explicitly defined target variables used for prediction tasks.

        Use the metadata to decide which category the dataset fits best. If there is no evidence of trends over time or prediction tasks, the data is likely observational/cross-sectional.

        Provide your answer in **one or two words only** (e.g., "Time-series," "Observational/cross-sectional," or "Labelled/supervised").
        """
        response = agent.prompt_agent(prompt)
        return response
    except Exception as e:
        logger.error("An error has occurred in the identify dataset type task: ", e)
        sys.exit()


def summarize_metadata(agent, metadata):

    try:
        prompt = f""" You are an intelligent technical summarizer tasked with converting metadata information that includes text + code output into a comprehensive and user-friendly text summary. Below is the metadata dictionary containing detailed information about a dataset. Your task is to create a well-structured and informative summary based on this metadata.

            - Ensure you include every detail from the metadata without omitting anything.
            - Write the output in clear, complete sentences organized into coherent paragraphs.
            - Structure the summary logically, starting with an overview of the dataset and followed by detailed descriptions of its features, dimensions, and other statistics.

            Here is the metadata dictionary: \n\n<metadata>{metadata}</metadata>\n\n

            """
        response = agent.prompt_agent(prompt)
        return response
    except Exception as e:
        logger.error("An error has occurred in the summarize metadata task: ", e)
        sys.exit()
        





            


    

        


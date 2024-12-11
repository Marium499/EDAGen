import openai
from llama_index.agent.openai import OpenAIAgent
# import os
# from langchain.memory import ConversationBufferMemory
# from llama_index import GPTVectorStoreIndex
# from langchain.chat_models import OpenAI
import logging
from io import BytesIO
import sys

from typing import List, Dict, Union
from io import BytesIO
import base64

from llama_index.agent.openai import OpenAIAgent
# from llama_index.tools.code_interpreter.base import CodeInterpreterToolSpec

from agents import ChatLLMOpenAI, CodeLLMOpenAI


logger = logging.getLogger("AppLogger")

def init_cc_agent(user_api_key: str) -> OpenAIAgent: 

    """
    Initialize the code interpreter agent with llama index code interpreter tools and user provided api key.
    Assign role to agent via prompt engineering
    Args:
        user_api_key (str): The api key entered by the user
    Returns:
        OpenAIAgent: the ci agent  
    
    Raises:
        Exception: Logs and exits the program if an error occurs during execution.

    """

    
    try:
        # Initialize agent and tools
        code_spec = CodeInterpreterToolSpec()

        tools = code_spec.to_tool_list()

        #create open ai agent with tools
        ci_agent = OpenAIAgent.from_tools(openai_api_key = user_api_key, tools=tools, verbose=True)
        # Initialize agent role
        agent_prompt = f"""You are a **Code Interpreter Agent** in a multi-agent Exploratory Data Analysis (EDA) system. Your primary role is to handle all coding-related tasks during the EDA process while also having the ability to understand dataset content and make informed decisions.  

            - **Responsibilities:**  
            1. Write efficient, accurate, and optimized code to perform tasks such as data cleaning, feature engineering, statistical analysis, and visualization.  
            2. Execute the defined code, ensuring that the outputs are correct and meaningful.  
            3. Generate and customize visualizations (e.g., scatter plots, histograms, box plots) to aid in data understanding.  
            4. Provide results in a structured format (e.g., tables, graphs) for further interpretation by other agents, such as the Domain Expert Agent.  
            5. Analyze datasets to understand their content, structure, and relationships. For example:  
                - If given multiple files, identify the main dataset containing rows and columns of primary interest.  
                - Detect and differentiate metadata or supporting files from the core dataset.  

            - **Guidelines:**  
            - Collaborate effectively with other agents by taking their inputs and instructions to generate the required outputs.  
            - Ensure that the code adheres to best practices for readability, modularity, and performance.  
            - Include comments and explanations where necessary to make the code understandable.  
            - Handle errors gracefully and provide clear feedback if the task cannot be completed.  
            - Use logical reasoning and content analysis to identify the structure and purpose of datasets.  

            - **Capabilities:**  
            - Use Python libraries like Pandas, NumPy, Matplotlib, and Seaborn to perform EDA tasks.  
            - Read and process datasets in various formats (e.g., CSV, Excel).  
            - Understand dataset content to classify and select relevant files (e.g., determining which file is the primary dataset).  
            - Generate and execute code snippets based on task descriptions provided by other agents or the user.  
            - Handle multi-file scenarios by analyzing file contents, comparing column structures, and identifying relationships between files.  

            Your goal is to act as the coding backbone of the system, executing EDA tasks accurately and efficiently, understanding dataset content deeply, and providing outputs that other agents can use to derive insights.
            """
        
        r = ci_agent.chat(agent_prompt)
        return ci_agent
    except Exception as e:
        logger.error("An error has occurred in initializing ci agent: ", e)
        # logging.info(e)

        sys.exit()

def get_data_summary(agent: CodeLLMOpenAI, files): # -> List[Dict[str, str]]:

    """
    Generate a summary of the main dataset from the provided data files using an AI agent.

    This function takes a list of data files and interacts with the AI agent to:
    1. Identify the main dataset containing rows and columns of primary interest.
    2. Extract information about the dataset's columns.
    3. Determine what is being measured or expressed in the dataset (qualitative and quantitative elements).
    4. Summarize the column information of the dataset.

    Args:
        agent (OpenAIAgent): The AI agent responsible for analyzing the dataset and generating responses.
        files (List[Dict[str, BytesIO]]): A list of file metadata, where each file is represented 
                                          as a dictionary with keys 'name' (str) and 'content' (BytesIO).

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing:
            - "columns": Information about the dataset's columns.
            - "data summary": A summary of the column details.
            - "query": The question about what is being measured.
            - "answer": The agent's response to the measurement query.

    Raises:
        Exception: Logs and exits the program if an error occurs during execution.
    """
    # try:

    prompt1 = f"""Based on the csv data file located in the path \n\n <data_path>"{files}"</data_path>\n\n
    write and execute code to print to console what columns does it have? Also print the dtypes of all the columns"""

    res1 = agent.prompt_client(prompt1)

    # prompt3a = f"""Based on the csv data file located in \n\n <data_path>"{files}"</data_path>\n\n
    # First, summarize the main dataset by providing column names and data types."""


    # res3a = agent.prompt_client(prompt3a)

    prompt2 = f"""Based on the csv data file located in the path \n\n <data_path>"{files}"</data_path>\n\n
    print a dictionary containing the number of rows and columns."""

    res2 = agent.prompt_client(prompt2)

    prompt3 = f"""
        Based on the details and summary of the dataset columns and their datatypes: 
        <column_types>{res1}</column_types>

        And the CSV data file located at: 
        <data_path>{files}</data_path>

        Perform the following tasks:
        1. Identify the categorical and numeric columns in the dataset and store them in a Python dictionary. The dictionary should have the format:
        {{
            "categorical_columns": ["col1", "col2", ...],
            "numeric_columns": ["col3", "col4", ...]
        }}
        2. For each categorical column, include up to 10 unique values and their respective counts in the dictionary under the key "categorical_column_details". The format should be:
        {{
            "column_name": {{"value1": count1, "value2": count2, ...}}
        }}
        3. For numeric columns, calculate descriptive statistics (mean, median, quantiles) and include them in the dictionary under the key "numeric_column_statistics". The format should be:
        {{
            "column_name": {{"mean": value, "median": value, "quantiles": {{"0.25": value, "0.5": value, "0.75": value}}}}
        }}

        Finally, return the complete dictionary as the output of the code (not just printed).
        """

    # prompt3 = f"""Based on the details and summary of the dataset columns and their datatypes \n<column_types>{res1}</column_types>\n 
    # and the csv data file located in \n\n <data_path>"{files}"</data_path>\n\n identify categorical and numeric columns, store them in a python dictionary and return that dictionary.
    # For categorical columns, print up to 10 the unique values and their respective counts in the columns,
    # and descriptive statistics (mean, median, quantiles) for numeric columns. Return all the information"""


    res3 = agent.prompt_client(prompt3, temp=0.5)

    metadata = [
        {"number of rows and columns": res2, 
         "columns and datatypes in the dataset": res1, 
         "dataset statistics": res3}
    ]
    


    prompt4 = f"""
        Using the dataset metadata provided below, which includes information about columns, data types, unique values, and column statistics: 
        <metadata>{metadata}</metadata> 

        I need you to analyze the dataset and explain what is being measured or expressed. Your response should:
        1. Include both qualitative and quantitative aspects based on the metadata and unique values in the columns.
        2. Provide a detailed description of each column, specifying the quantities or qualities being measured.
        3. Be specific in your descriptions—avoid generalizations.
        4. Avoid generating code; instead, focus on interpreting the information presented.

        Please highlight key measurements and express your analysis clearly.
        """

    
    res4 = agent.prompt_client(prompt4,temp=0.7)


    # [-1]["content"]
    # if isinstance(res3[-1], dict):
    #     res3 = res3[-1]["content"]
    # else:
    #     try:
    #         res3 = res3[-1].content
    #     except:
    #         res3 = res3[-1]


    # prompt4 = f"""Based on the provided data files in this list \n\n <data_files>{files}</data_files>\n\n
    # discern which one is the main dataset containing rows and columns of primary interest and 
    # summarise the column information."""

    # res4 = agent.prompt_client(prompt4)[-1]["content"]

    metadata.append([{"question": prompt4, "answer":res4}])
    

    return metadata
    # except Exception as e:
    #     logger.error("An error has occurred in the data summary task: ", e)
    #     sys.exit()


def process_eda_questions(agent: CodeLLMOpenAI, questions: List[str], file, metadata): # -> List[Dict[str, str]]:

    """
    Function to handle questions and provide flexible outputs, including text, graphs, or other file outputs.

    Args:
        agent: OpenAIAgent instance to handle queries.
        questions: List of questions to process.
        files: List of file metadata in the format {"name": str, "content": BytesIO}.
    
    Returns:
        List of dictionaries with questions, outputs, and output metadata.

    Raises:
        Exception: Logs and exits the program if an error occurs during execution.
    """

    qa_list = []
    for q in questions:

        prompt = f"""
        You are tasked with analyzing a dataset and answering the relevant question using Python code.

        The dataset is located at: 
        <data_path>{file}</data_path>

        To assist you, relevant metadata has been provided, which includes details about the dataset and a random sample of rows to help you understand its structure:
        <metadata>{metadata}</metadata>

        Your task:
        - Write Python code that directly answers the following question: 
        <question>{q}</question>

        Important Instructions:
        1. Use the random subset included in the metadata to understand the structure, columns, and data types of the dataset.
        2. Do NOT use exploratory commands like `df.head()` or similar functions to explore the data structure, as the required information is already provided in the metadata.
        3. Avoid printing anything. Instead, return all outputs as structured text and/or plots. 
        4. Combine any computed values with explanatory text. For example, if you calculate a covariance of `x` between two columns `abc` and `xyz`, return a descriptive string such as: "The covariance of abc with xyz is x."
        5. Include visualizations or plots in your output when relevant, accompanied by clear textual explanations.

        Ensure your solution is complete, providing a well-structured response that directly addresses the question using the provided metadata.
        """

    # If your answer involves code execution, return the results in the appropriate format.
    #     For graphs or visualizations, encode the graph as Base64 and include a description.
    # """
        
        res = agent.prompt_client(prompt)
        # res_last = res[-1]["content"]

        # if isinstance(res, str):
            # Simple textual response
        qa_list.append({
            "question": q,
            # "output_type": "text",
            "answer": res
        })
        
        # elif isinstance(res, dict) and res.get("type") == "graph":
        #     # Graph output with Base64 encoding
        #     qa_list.append({
        #         "question": q,
        #         "output_type": "graph",
        #         "description": res.get("description"),
        #         "content": res.get("content")  # Base64-encoded graph
        #     })
        # elif isinstance(res, dict) and res.get("type") == "file":
        #     # File output
        #     qa_list.append({
        #         "question": q,
        #         "output_type": "file",
        #         "description": res.get("description"),
        #         "file_name": res.get("file_name"),
        #         "file_content": res.get("file_content")  # Base64-encoded file
        #     })
        # else:
        #     # Handle unexpected responses
        #     qa_list.append({
        #         "question": q,
        #         "output_type": "unknown",
        #         "output": "Unexpected response format."
        #     })

    return qa_list







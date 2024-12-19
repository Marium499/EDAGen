
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
from langchain_community.tools import E2BDataAnalysisTool
import os
import json
import base64
import time
from langchain_community.tools import tool
from PIL import Image




from openai import OpenAI
from e2b_code_interpreter import Sandbox
from fastapi.encoders import jsonable_encoder

from langchain.agents import AgentType, initialize_agent
# from langchain_openai import ChatOpenAI

from llama_index.agent.openai import OpenAIAgent
# from llama_index.tools.code_interpreter.base import CodeInterpreterToolSpec

import uuid
import logging

logger = logging.getLogger("AppLogger")

@tool
def load_secret_image():
    "load a secret image that I have prepared for you"

    img = Image.open("secret_image.png")
    return img


class ChatLLMOpenAI():
    """
    A class to manage a chat-based LLM workflow using OpenAI's API.

    This class integrates an OpenAI chat model into a state-based graph workflow, enabling
    interaction with an LLM for conversational tasks. It provides functionality to initialize
    the workflow, handle memory for conversation threads, and define a role-specific prompt 
    for the agent.

    Attributes:
        workflow (StateGraph): A state graph object managing the flow of conversation states.
        agent (ChatOpenAI): The OpenAI chat model for generating responses.
        memory (MemorySaver): A memory saver to store conversation states and checkpoints.
        app (Callable): The compiled state graph application for managing interactions.
        config (dict): Configuration for conversation threads, including a unique thread ID.

    Methods:
        __init__(user_api_key, role_prompt=None):
            Initializes the chat workflow, OpenAI agent, and memory system.
        
        call_model(state: MessagesState) -> dict:
            Invokes the OpenAI chat model using the current state messages and returns the response.

        prompt_agent(msg: str) -> list:
            Sends a role-specific prompt message to the agent and returns the updated conversation 
            messages.

    Args:
        user_api_key (str): The API key for accessing OpenAI's API.
        role_prompt (str, optional): An optional role-specific prompt for initializing the agent's behavior.
    """
    def __init__(self, user_api_key, role_prompt=None) -> None:
        
        """
        Initializes the ChatLLMOpenAI class.

        Args:
            user_api_key (str): The API key for accessing OpenAI's API.
            role_prompt (str, optional): An optional role-specific prompt to define the agent's behavior.

        Sets up:
        - A state-based workflow using `StateGraph`.
        - An OpenAI chat model with the provided API key.
        - A memory saver to manage conversation states.
        - A unique thread ID for managing conversation threads.

        Raises:
            Exception: Logs the error
        """
        try:
            # define graph
            self.workflow = StateGraph(state_schema=MessagesState)

            # Define chat agent
            self.agent = ChatOpenAI(openai_api_key=user_api_key, verbose=True, model="gpt-4o")

            # Define the two nodes we will cycle between
            self.workflow.add_edge(START, "model")
            self.workflow.add_node("model", self.call_model)

            self.memory = MemorySaver()

            self.app = self.workflow.compile(
                checkpointer=self.memory
            )
            # threads for multiple conversations : unlikely in my usecase
            thread_id = uuid.uuid4()
            self.config = {"configurable": {"thread_id": thread_id}}
            if role_prompt:
                self.prompt_agent(role_prompt)

        except Exception as e:
            logger.error("An error has occurred initialising agent: ", e)
            # ADD LOGIC: Add safe exception handling function that displays user session variables
            raise(e)
            

    def call_model(self, state: MessagesState):
        """
        Invokes the chat model with the given conversation state.

        Args:
            state (MessagesState): The current state containing conversation messages.

        Returns:
            dict: The updated state containing the model's response messages.
        
        Raises:
            Exception: Logs the error
        """
        response = self.agent.invoke(state["messages"])
        # We return a list, because this will get added to the existing list
        return {"messages": response}
    
    def prompt_agent(self,msg):
        """
        Sends a prompt message to the agent.

        Args:
            msg (str): The prompt message for initializing the agent's behavior.

        Returns:
            list: The updated list of conversation messages after processing the prompt.
        
        Raises:
            Exception: Logs the error
        """
        try:

            input_message = HumanMessage(content=msg)
            for event in self.app.stream({"messages": [input_message]}, self.config, stream_mode="values"):
                event["messages"][-1].pretty_print()
            system_prompt = event["messages"][0]
            event["messages"] = system_prompt + event["messages"][-1]
            return event["messages"][-1].content

        except Exception as e:
            logger.error("An error has occurred in prompt agent: ", e)


class CodeLLMOpenAI():

    def __init__(self, openai_key, e2b_key, role_prompt=None, model ="gpt-4o", temp=0.7) -> None:

        self.client = OpenAI(api_key=openai_key)
        self.e2b_key = e2b_key
        self.temp = temp

        self.sbx = Sandbox(api_key=self.e2b_key, timeout=3599)


        if role_prompt is None:
            
            self.role_prompt="""You are an expert data analysis agent who writes code to perform data analysis and data visualizations using python.
            You will be provided with data encapsulated within prompts or paths to csv files that you can read using python pandas library.
            You should be able to deal with numeric and categorical variables in datasets according to best practices in data analysis in python.
            You must return the outputs always be it text output and/or display plots as the task requires. Your code will be run in the e2b interpreter sandbox environment."""# The code must be implemented as a python script, do not assume an ipy kernel."""
            # Also assume independent execution of each prompt code, that means variables and libraries from a previous code snippet will not be stored for use in another code execution.
            # You code will be run in the e2b interpreter sandbox environment."""
        else:
            self.role_prompt = role_prompt
        self.messages = [
                {"role": "system", "content": self.role_prompt}
            ]
        
        self.model = model
        self.full_code = ""

        self.tools = [{
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Execute python code in a Jupyter notebook cell and return result",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The python code to execute in a single cell"
                            }
                        },
                        "required": ["code"]
                    }
                }
            }]



    def prompt_client(self, msg, temp=None):
        messages = self.messages + [{"role": "user", "content": msg}]
        if temp is not None:
            self.temp=temp

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages= messages,
                # [{"role": "system", "content": self.role_prompt},{"role": "user", "content": msg}],
                tools=self.tools,
                temperature=self.temp,
            ) 
            print(len(response.choices))
        except Exception as error:
            print(f"Failed in createChatCompletion call {error}")

        response_message = response.choices[0].message
        # print(response_message)

        messages.append({
            "role": response_message.role,
            "content": response_message.content,
            "function_call" : response_message.function_call,
            "tool_calls": response_message.tool_calls,
        })

        
        content_str = ""
        # Execute the tool if it's called by the model
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                if tool_call.function.name == "execute_python":
                    # Create a sandbox and execute the code

                    code_to_run = json.loads(tool_call.function.arguments)['code']

                    print(code_to_run)
                    
                    execution = self.sbx.run_code(code_to_run)

                    # print(execution)
                    results = execution.results
                    # print(results)

                    
                    for res in results:
                        try:
                            if res.png:
                            # Save the png to a file. The png is in base64 format.
                                id = uuid.uuid4()
                                with open(f'../charts/{id}_chart.png', 'wb') as f:
                                    f.write(base64.b64decode(res.png))
                                print(f'Chart saved as {id}_chart.png')
                                res = jsonable_encoder(res)
                                content_str = content_str +" item: "+ f"Read chart from path ../charts/{id}_chart.png\n\n"

                            
                            else:
                                print("not png output")
                                res = jsonable_encoder(res)
                                content_str = content_str +" item: " + res["text"] +"\n\n"
                        except:
                            print("here in except clause")
                            res = jsonable_encoder(res)
                            content_str = content_str +" item: " + str(res) +"\n\n"

                    # print(f"content str: {content_str}")
                    messages.append({
                        "role": "tool",
                        "name": "execute_python",
                        "content": content_str,
                        "tool_call_id": tool_call.id,
                    })
        else:
            content_str = response_message.content
        

        # self.messages = messages

        return content_str
    
    def upload_file_sandbox(self, file_path):
    
        # Read local file
        # for file in file_paths:
        with open(file_path, "rb") as file:
        # Upload file to the sandbox to path '/home/user/my-file'
            self.dataset_path_in_sandbox = self.sbx.files.write("dataset.csv", file)

        return self.dataset_path_in_sandbox


    def save_artifact(self, artifact, filename=uuid.uuid4()):
        print("New matplotlib chart generated:", artifact.name)
        # Download the artifact as `bytes` and leave it up to the user to display them (on frontend, for example)
        file = artifact.download()
        basename = os.path.basename(artifact.name)

        # Save the chart to the `charts` directory
        with open(f"./charts/{basename+"_"+filename}", "wb") as f:
            f.write(file)
        return f"./charts/{basename+"_"+filename}"







# class CodeLLMOpenAI():

#     def __init__(self, user_api_key, e2b_api_key,  model=None, role_prompt=None,) -> None:

#         try:
            
        
#             e2b_data_analysis_tool = E2BDataAnalysisTool(
#                 api_key = e2b_api_key,
#             # Pass environment variables to the sandbox
#                 # env_vars={"MY_SECRET": "secret_value"},
#                 on_stdout=lambda stdout: print("stdout:", stdout),
#                 on_stderr=lambda stderr: print("stderr:", stderr),
#                 on_artifact=self.save_artifact,
#             )

#             #create open ai agent with tools
#             tools = [e2b_data_analysis_tool.as_tool()]

#             llm = ChatOpenAI(openai_api_key = user_api_key,model="gpt-4", temperature=0)

#             self.agent = initialize_agent(
#                 tools,
#                 llm,
#                 agent=AgentType.OPENAI_FUNCTIONS,
#                 verbose=True,
#                 handle_parsing_errors=True,
#             )
#             if role_prompt is not None:
#                 r = self.agent.prompt_agent(role_prompt)

#         except Exception as e:
#             logger.error("An error has occurred in initializing code agent: ", e)

#             # ADD LOGIC: Add safe exception handling function that displays user session variables
#             raise(e)
        
#     def save_artifact(self, artifact):
#         print("New matplotlib chart generated:", artifact.name)
#         # Download the artifact as `bytes` and leave it up to the user to display them (on frontend, for example)
#         file = artifact.download()
#         basename = os.path.basename(artifact.name)

#         # Save the chart to the `charts` directory
#         with open(f"./charts/{basename}", "wb") as f:
#             f.write(file)

#     def prompt_agent(self,msg):
#         """
#         Sends a prompt message to the agent.

#         Args:
#             msg (str): The prompt message for initializing the agent's behavior.

#         Returns:
#             list: The updated list of conversation messages after processing the prompt.
        
#         Raises:
#             Exception: Logs the error
#         """
#         try:
#             input_message = HumanMessage(content=msg)
#             for event in self.app.stream({"messages": [input_message]}, self.config, stream_mode="values"):
#                 event["messages"][-1].pretty_print()
#             return event["messages"]
#         except Exception as e:
#             logger.error("An error has occurred in prompt agent: ", e)
            
    
        







# Autonomous LLM-Powered EDA Assistant

## Overview
The Autonomous EDA Assistant is an innovative project leveraging **Large Language Models (LLMs)** and a multi-agent architecture to perform **Exploratory Data Analysis (EDA)** on uploaded datasets. Designed as a minimal viable product (MVP), this application dynamically analyzes datasets, generates visualizations, identifies trends, and provides actionable insights—all without requiring explicit user instructions.

The system uses **Streamlit** as its interface, **LangChain** for LLM integration, and the **e2b sandbox** to securely execute Python code. Future plans include enhancements for preprocessing, dynamic routing, and multi-user scalability.

---

## Key Features
- Automated detection of dataset structure and domain.
- Generation of dynamic visualizations and textual summaries.
- Multi-agent collaboration for tasks like summarization, domain detection, and question generation.
- Secure and scalable Python code execution in the e2b sandbox.
- User-friendly interface powered by Streamlit.

---

## Technologies Used

### 1. **LangChain**
LangChain is a framework for developing applications powered by **Language Models (LLMs)**. It facilitates structured prompts, tool usage, and chaining of agents for complex tasks.
- **Applications in this project**: 
  - Prompt engineering.
  - Agent chaining with LangGraph for managing chat history and workflows.
  - Dynamic generation of insights and responses.

### 2. **e2b Sandbox**
The e2b sandbox is a secure environment for running Python code generated by LLMs.
- **Applications in this project**:
  - Executes Python scripts for data analysis and visualization.
  - Manages outputs such as visualizations and logs errors during execution.
  - Ensures data privacy by handling dataset processing in isolated sessions.

### 3. **Streamlit**
Streamlit is an open-source app framework used to create web applications for machine learning and data science projects.
- **Applications in this project**:
  - Provides the front-end interface for dataset upload and API key input.
  - Displays outputs including visualizations, insights, and summaries in a conversational format.

### 4. **Prompt Engineering**
Prompt engineering involves designing inputs to LLMs to elicit desired responses.
- **Applications in this project**:
  - Restricting unique dataset values in prompts to manage token limits.
  - Tailoring agents' roles to specific tasks, such as summarization or visualization.

### 5. **Chain-of-Thought Prompting**
This technique enables LLMs to reason through multi-step problems by explicitly prompting step-by-step reasoning.
- **Applications in this project**:
  - Generating EDA questions and corresponding analyses.
  - Breaking down complex data insights into simpler, interpretable steps.

### 6. **LangGraph**
LangGraph is used to manage state and maintain chat histories for agent workflows.
- **Applications in this project**:
  - Tracks conversation states between the Manager Agent and Domain Expert Agent.
  - Enables smooth transitions between tasks in the multi-agent system.

---

## Architecture
Below is the program flow for the architecture:

1. **Streamlit Interface**:
   - Accepts API keys (OpenAI, e2b) and a CSV dataset from the user.
   - Initiates the EDA process upon successful validation.

2. **Dataset Storage**:
   - Uploaded datasets are securely stored in a user-specific e2b sandbox environment.

3. **Manager Agent**:
   - Initializes with system prompts and orchestrates workflows.
   - Detects dataset domain and generates prompts for the Domain Expert Agent.

4. **Code Interpreter Agent**:
   - Analyzes the dataset to generate metadata and summaries.
   - Executes Python code for visualizations and data exploration in the e2b sandbox.

5. **Domain Expert Agent**:
   - Receives dataset metadata and generates insightful EDA questions.
   - Provides explanations for patterns and relationships in the data.

6. **Summarizer Agent**:
   - Synthesizes all outputs into a concise summary for the user.

7. **Streamlit Output**:
   - Displays EDA questions, visualizations, insights, and summaries in a user-friendly format.

---

## Workflow
1. User uploads a dataset and enters API keys.
2. Manager and Code Interpreter Agents are initialized.
3. Code Interpreter generates dataset summaries and metadata.
4. Manager Agent identifies the dataset domain and initializes the Domain Expert Agent with a system prompt.
5. Domain Expert Agent formulates relevant EDA questions.
6. Code Interpreter creates visualizations and analyses for the questions.
7. Domain Expert provides observations, and Summarizer Agent generates a final summary.
8. Streamlit displays the results.

---

## Limitations
- Supports only CSV file uploads.
- No preprocessing for missing or malformed data.
- Lacks dynamic routing between agents.
- Single-user focused in the current MVP.
- Logs and session data are not yet secured.

---

## Future Features
- **Data Preprocessing**: Handle missing data and preprocess datasets.
- **Dynamic Routing**: Enable intelligent query routing between agents.
- **Multi-User Support**: Add session-based scalability.
- **Interactive Visualizations**: Generate dynamic, interactive plots.
- **Domain Detection Models**: Introduce ML/statistical models for better domain identification.
- **Chat Interface**: Add conversational interaction for diverse queries.
- **Open Source Library**: Transition the project into a publicly available library.

---

## Key Challenges and Solutions
1. **Max Tokens Issue**:
   - Fixed by limiting unique values in prompts to 10.
2. **Agent Response Variability**:
   - Addressed by adjusting temperature settings.
3. **Cost Analysis**:
   - Optimized for affordable execution using efficient API calls.

---

## Setup and Usage

### Prerequisites
- Python 3.8+
- OpenAI and e2b API keys.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/eda-assistant.git
   cd eda-assistant
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Usage
1. Enter OpenAI and e2b API keys in the Streamlit app.
2. Upload a CSV dataset.
3. View dynamic visualizations, insights, and summaries directly on the app.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## References
- [LangChain Documentation](https://langchain.readthedocs.io/)
- [e2b Sandbox Documentation](https://e2b.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## Contact
For questions or feedback, reach out to **Marium Aslam** at m2aslam@uwaterloo.ca.



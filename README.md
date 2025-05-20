# Build an Agent with Tool-Calling Superpowers Using `smolagents`

This project demonstrates how to create powerful AI agents using the `smolagents` library. These agents, powered by large language models (LLMs), can perform advanced tasks like web browsing, image generation, retrieval-augmented generation (RAG), and code debugging by integrating specialized tools.

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Multimodal + Web-Browsing Assistant](#1-multimodal--web-browsing-assistant)
  - [2. RAG with Iterative Query Refinement & Source Selection](#2-rag-with-iterative-query-refinement--source-selection)
  - [3. Debug Python Code](#3-debug-python-code)
- [Contributing](#contributing)
- [License](#license)


## Introduction
**Agents** are LLM-powered systems that use tools to tackle complex tasks. These tools enhance the LLM’s capabilities, allowing it to perform functions it couldn’t handle alone, such as web searches, image generation, or code execution.

**`smolagents`** is a library that provides the building blocks to craft custom agents. This project showcases three exciting use cases:
1. A multimodal agent that browses the web and generates images.
2. A RAG system with dynamic query refinement and source selection.
3. A code debugging agent that fixes and runs Python code.


## Features
- **Web Search & Image Generation**: Seamlessly combine web browsing and image creation.
- **Dynamic RAG**: Retrieve knowledge with iterative query adjustments and source filtering.
- **Code Debugging**: Automatically identify and fix errors in Python code.
- **Tool Integration**: Easily incorporate tools like search engines, retrievers, and interpreters.
- **LLM-Powered**: Leverages advanced models like `Qwen/Qwen2.5-72B-Instruct` via Hugging Face Inference API.


## Installation
To set up the project, install the necessary dependencies:

```bash
pip install smolagents datasets langchain sentence-transformers faiss-cpu duckduckgo-search openai langchain-community --upgrade -q
```

Next, log in to Hugging Face to use the Inference API:

```python
from huggingface_hub import notebook_login
notebook_login()
```


## Usage
Here are the three main use cases with code examples and explanations.

### 1. Multimodal + Web-Browsing Assistant
This agent can search the web and generate images based on user input.

#### Example
```python
from smolagents import load_tool, CodeAgent, InferenceClientModel, DuckDuckGoSearchTool

# Load image generation tool
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

# Use built-in web search tool
search_tool = DuckDuckGoSearchTool()

# Initialize the agent
model = InferenceClientModel("Qwen/Qwen2.5-72B-Instruct")
agent = CodeAgent(tools=[image_generation_tool, search_tool], model=model)

# Run the agent
result = agent.run("Generate a photo of the car James Bond drove in the latest movie.")
print(result)
```

#### Output
The agent searches for details (e.g., an Aston Martin DB5) and generates an image.


### 2. RAG with Iterative Query Refinement & Source Selection
This agent performs Retrieval-Augmented Generation (RAG) with dynamic control over retrieval parameters.

#### Setup
1. Load and prepare the knowledge base:
```python
import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
source_docs = [Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]}) for doc in knowledge_base]
docs_processed = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(source_docs)[:1000]

embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
vectordb = FAISS.from_documents(documents=docs_processed, embedding=embedding_model)
```

2. Define the `RetrieverTool`:
```python
from smolagents import Tool
import json

class RetrieverTool(Tool):
    name = "retriever"
    description = "Retrieves documents from the knowledge base based on the query."
    inputs = {
        "query": {"type": "string", "description": "The search query."},
        "source": {"type": "string", "description": "Sources to search (e.g., ['transformers', 'blog'])."},
        "number_of_documents": {"type": "string", "description": "Number of documents to retrieve."}
    }
    output_type = "string"

    def __init__(self, vectordb, all_sources, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb
        self.all_sources = all_sources

    def forward(self, query: str, source: str = None, number_of_documents=7) -> str:
        # Implementation details...
```

3. Run the agent:
```python
from smolagents import ToolCallingAgent

model = InferenceClientModel("Qwen/Qwen2.5-72B-Instruct")
retriever_tool = RetrieverTool(vectordb=vectordb, all_sources=all_sources)
agent = ToolCallingAgent(tools=[retriever_tool], model=model)
agent_output = agent.run("Show me a LORA finetuning script")
print(agent_output)
```

#### Output
The agent retrieves relevant documents, adjusts the query if needed, and provides an answer.

---

### 3. Debug Python Code
This agent fixes errors in Python code using a built-in interpreter.

#### Example
```python
from smolagents import CodeAgent

agent = CodeAgent(tools=[], model=InferenceClientModel("Qwen/Qwen2.5-72B-Instruct"))

faulty_code = """
numbers = [0, 1, 2]
for i in range(4):
    print(numbers(i))
"""

final_answer = agent.run(
    "Debug this code and return the corrected version.",
    additional_args=dict(code=faulty_code)
)
print(final_answer)
```

#### Output
The agent corrects the code to:
```python
numbers = [0, 1, 2]
for i in range(len(numbers)):
    print(numbers[i])
```


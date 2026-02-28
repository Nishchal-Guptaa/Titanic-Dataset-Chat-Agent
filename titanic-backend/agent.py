# agent.py

import matplotlib
matplotlib.use("Agg")

import os
from dotenv import load_dotenv
from typing import Tuple, Optional, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.agents import create_pandas_dataframe_agent

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0
)

# --------------------------------------------------
# Load Titanic Dataset (Pandas)
# --------------------------------------------------
CSV_PATH = "titanic-backend/Titanic-Dataset.csv"
df = pd.read_csv(CSV_PATH)

# --------------------------------------------------
# Vector Store Setup (for semantic retrieval)
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="titanic_collection",
    embedding_function=embeddings,
)

# Load CSV into vector store (only once)
if vector_store._collection.count() == 0:
    loader = CSVLoader(CSV_PATH)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    splits = text_splitter.split_documents(documents)
    vector_store.add_documents(splits)


# --------------------------------------------------
# Tool 1: Retrieval Tool
# --------------------------------------------------
@tool(response_format="content_and_artifact")
def retrieve_context(query: str) -> Tuple[str, Optional[Any]]:
    """Retrieve row-level passenger info using semantic search."""
    docs = vector_store.similarity_search(query, k=2)

    text = "\n\n".join(
        f"Metadata: {doc.metadata}\nContent: {doc.page_content}"
        for doc in docs
    )

    return text, None


# --------------------------------------------------
# Tool 2: Data Analysis Tool
# --------------------------------------------------
pandas_agent = create_pandas_dataframe_agent(
    model,
    df,
    verbose=True,
    allow_dangerous_code=True,
)

@tool(response_format="content_and_artifact")
def analyze_data(query: str):
    """
    Uses intelligent pandas agent to answer statistical
    and visualization queries dynamically.
    """

    response = pandas_agent.invoke(query)

    return response["output"], None



# --------------------------------------------------
# Agent Setup
# --------------------------------------------------
tools = [retrieve_context, analyze_data]

system_prompt = """
You are a Titanic dataset analysis assistant.

If the question involves:
- percentages
- averages
- counts
- statistics
- survival rate
- fare
- embarked ports
- visualizations

ALWAYS use the analyze_data tool.

If the question asks about specific passenger details,
use the retrieve_context tool.

Respond clearly and concisely.

Do not answer questions that are not related to the Titanic dataset. If user asks unrelated questions, politely decline and remind them that you can only answer questions about the Titanic dataset even if the user insists. Do not use any tool for unrelated questions, just respond with a polite refusal.

If user asks for visualizations, generate the plot using the analyze_data tool and return a description of the plot in the text response.

If user asks for generating a visualization then generate the plot only one time per query. Do not generate multiple plots for a single query unless a user asks for multiple visualizations in a single query.
"""

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
)


# --------------------------------------------------
# Public Function for FastAPI
# --------------------------------------------------
def run_agent(query: str):
    """
    This function will be called from FastAPI.
    Returns final text and optional artifact.
    """
    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )

    return response

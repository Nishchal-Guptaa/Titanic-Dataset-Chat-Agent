# agent.py

import matplotlib
matplotlib.use("Agg")

import os
from dotenv import load_dotenv
from typing import Tuple, Optional, Any

import pandas as pd

from langchain_groq import ChatGroq

from langchain.tools import tool
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")


# --------------------------------------------------
# Load Model (lightweight, no large download)
# --------------------------------------------------

model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    max_tokens=4096,
)

# --------------------------------------------------
# Load Titanic Dataset (Pandas) - small CSV, fine at import
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Titanic-Dataset.csv")

df = pd.read_csv(CSV_PATH)


# --------------------------------------------------
# Lazy-loaded heavy components to reduce startup memory
# --------------------------------------------------
_embeddings = None
_vector_store = None
_pandas_agent = None
_agent = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        )
    return _embeddings


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        from langchain_chroma import Chroma
        embeddings = _get_embeddings()
        _vector_store = Chroma(
            collection_name="titanic_collection",
            embedding_function=embeddings,
        )
        # Load CSV into vector store (only once)
        if _vector_store._collection.count() == 0:
            loader = CSVLoader(CSV_PATH)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )

            splits = text_splitter.split_documents(documents)
            _vector_store.add_documents(splits)
    return _vector_store


def _get_pandas_agent():
    global _pandas_agent
    if _pandas_agent is None:
        from langchain_experimental.agents import create_pandas_dataframe_agent
        _pandas_agent = create_pandas_dataframe_agent(
            model,
            df,
            verbose=True,
            allow_dangerous_code=True,
        )
    return _pandas_agent


# --------------------------------------------------
# Tool 1: Retrieval Tool
# --------------------------------------------------
@tool(response_format="content_and_artifact")
def retrieve_context(query: str) -> Tuple[str, Optional[Any]]:
    """Retrieve row-level passenger info using semantic search."""
    vector_store = _get_vector_store()
    docs = vector_store.similarity_search(query, k=2)

    text = "\n\n".join(
        f"Metadata: {doc.metadata}\nContent: {doc.page_content}"
        for doc in docs
    )

    return text, None


# --------------------------------------------------
# Tool 2: Data Analysis Tool
# --------------------------------------------------
@tool(response_format="content_and_artifact")
def analyze_data(query: str):
    """
    Uses intelligent pandas agent to answer statistical
    and visualization queries dynamically.
    """
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    # Close any pre-existing figures so we only capture new ones
    plt.close("all")

    pandas_agent = _get_pandas_agent()
    response = pandas_agent.invoke(query)

    # Capture any matplotlib figures the agent created
    artifact = None
    figures = [plt.figure(i) for i in plt.get_fignums()]
    if figures:
        buf = BytesIO()
        figures[-1].savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        artifact = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close("all")

    return response["output"], artifact


# --------------------------------------------------
# Agent Setup (also lazy)
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


def _get_agent():
    global _agent
    if _agent is None:
        from langchain.agents import create_agent
        _agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
        )
    return _agent


# --------------------------------------------------
# Public Function for FastAPI
# --------------------------------------------------
def run_agent(query: str):
    """
    This function will be called from FastAPI.
    Returns final text and optional artifact.
    """
    agent = _get_agent()
    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )

    return response

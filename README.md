## RAG Demo for MAPELab

A basic RAG demo for MAPELAB using LangChain, Chroma, Cohere, and Sentence-Transformers on Streamlit which uses input data and Chroma databases stored in the local space, specifically the conversation dataset and two research papers.

To run:

``python -m venv ragdemo``

``source ragdemo/bin/activate``

``pip install -r requirements.txt``

``streamlit run streamlit.app.py``

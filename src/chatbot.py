import os
import pickle
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "vectorstore.pkl")

with open(file_path, "rb") as f:
    vectorstore = pickle.load(f)

# Set your Groq API key directly
os.environ["GROQ_API_KEY"] = "your_GROQ_API_key"

# Initialize the Groq LLM for generation
llm = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct")

# Create the Retriever
# The retriever's job is to fetch the relevant chunks from our vector store.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 most relevant chunks

# 3. Create the Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step-by-step and provide a detailed answer. If you don't know the answer,
just say that you don't have enough information from the provided documents.
DO NOT use any of your own knowledge.

<context>
{context}
</context>

Question: {input}
""")

# Create the "Stuff" Documents Chain
# This chain takes the retrieved documents and "stuffs" them into the prompt.
document_chain = create_stuff_documents_chain(llm, prompt)

# 5Create the Full Retrieval Chain
# This master chain orchestrates everything: it takes the user's query,
# uses the retriever to get the documents, and then passes them to the document_chain.
retrieval_chain = create_retrieval_chain(retriever, document_chain)

'''
 --- Test Case 1 ---
question = "what drug am i meant to take in the night"
print(f"Question: {question}")
response = retrieval_chain.invoke({"input": question})
print("\n--- Answer ---")
print(response["answer"])
'''

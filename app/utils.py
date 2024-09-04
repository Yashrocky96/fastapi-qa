# RAG Prompt Template
RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""


# Helper function to format the retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

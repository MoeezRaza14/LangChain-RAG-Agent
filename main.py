# from langchain_ollama.llms import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate
# from vector import retriever

# model = OllamaLLM(model="llama3.2:1b")

# template = """
# You are an expert in answering questions about a pizza restaurant

# Here are some relevant reviews: {reviews}

# Here is the question to answer: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)
# chain = prompt | model

# while True:
#     print("\n\n-------------------------------------")
#     question = input("Ask your question (q to quit): ")
#     print("\n\n-------------------------------------")
#     if question == "q":
#         break
#     docs = retriever.invoke(question)
#     print(f"[debug] #docs returned: {len(docs)}")
#     if len(docs) == 0:
#         reviews_text = "No relevant reviews found in DB."
#     else:
#         reviews_list = []
#         for d in docs:
#             md = d.metadata if hasattr(d, "metadata") else {}
#             rating = md.get("rating") if md else None
#             date = md.get("date") if md else None
#             header = f"Rating: {rating} | Date: {date}" if rating or date else ""
#             reviews_list.append(header + "\n" + d.page_content)
#         reviews_text = "\n\n---\n\n".join(reviews_list)

#     result = chain.invoke({"reviews": reviews_text, "question": question})
#     print(result)

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.memory.buffer import ConversationBufferMemory
from vector import retriever

# LLM
llm = OllamaLLM(model="llama3.2:1b")

# Memory (keeps full conversation)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Prompt with history + RAG context
prompt = ChatPromptTemplate.from_template("""
You are an expert in answering questions about a pizza restaurant.

Conversation history:
{chat_history}

Relevant customer reviews:
{reviews}

User question:
{question}

Answer strictly using the reviews and conversation context.
""")

def run_rag(question: str):
    # Retrieve docs
    docs = retriever.invoke(question)

    if len(docs) == 0:
        reviews_text = "No relevant reviews found."
    else:
        reviews_text = "\n\n---\n\n".join(
            [
                f"Rating: {d.metadata.get('rating')} | Date: {d.metadata.get('date')}\n{d.page_content}"
                for d in docs
            ]
        )

    # Load memory
    chat_history = memory.load_memory_variables({})["chat_history"]

    chain = prompt | llm

    response = chain.invoke({
        "chat_history": chat_history,
        "reviews": reviews_text,
        "question": question
    })

    # Save conversation
    memory.save_context(
        {"input": question},
        {"output": response}
    )

    return response
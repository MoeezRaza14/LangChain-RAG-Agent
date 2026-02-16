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


import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==================================================
# 1. MODELS - THE REASONING ENGINE
# ==================================================

from langchain_ollama import ChatOllama

# Use your local Qwen model through Ollama.
model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0
)

response = model.invoke(
    "What is LangChain in one sentence?"
)

print("\n=== Model Response ===")
print(response.content)


# ==================================================
# 2. PROMPT TEMPLATES - STEERING THE MODEL
# ==================================================

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
)

# A. Simple text prompt template
simple_template = PromptTemplate(
    input_variables=["topic"],
    template=(
        "Explain {topic} to a complete beginner "
        "in 2-3 sentences."
    )
)

formatted = simple_template.format(
    topic="AI agents"
)

print("\n=== Formatted Prompt ===")
print(formatted)


# B. Chat prompt template with message roles
chat_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful coding tutor. "
        "Keep answers short and clear."
    ),
    (
        "human",
        "Explain {concept} with a simple Python example."
    )
])

messages = chat_template.format_messages(
    concept="list comprehension"
)

print("\n=== Chat Messages ===")
for msg in messages:
    print(f"[{msg.type}]: {msg.content[:200]}")


# ==================================================
# 3. CHAINS - CONNECTING THE PIECES
# ==================================================

from langchain_core.output_parsers import StrOutputParser

# Pipeline: Prompt -> Model -> Output Parser
chain = chat_template | model | StrOutputParser()

result = chain.invoke({
    "concept": "for loops"
})

print("\n=== Chain Output ===")
print(result)


# ==================================================
# 4. MEMORY - GIVING THE MODEL CONTEXT
# ==================================================

from langchain_core.chat_history import (
    InMemoryChatMessageHistory
)
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

# Create an in-memory conversation history.
memory = InMemoryChatMessageHistory()

# Add an example conversation.
memory.add_message(
    HumanMessage(
        content="My name is Alex and I'm learning LangChain"
    )
)

memory.add_message(
    AIMessage(
        content="Nice to meet you, Alex! "
                "LangChain is a great choice."
    )
)

memory.add_message(
    HumanMessage(
        content="What tools should I learn first?"
    )
)

memory.add_message(
    AIMessage(
        content="Start with PromptTemplates and simple chains, "
                "then move to tools and agents."
    )
)

print("\n=== Memory Contents ===")
for msg in memory.messages:
    print(f"[{msg.type}]: {msg.content}")


# Insert conversation history into the next prompt.
chat_with_memory = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful tutor. Use the conversation "
        "history to personalize your responses."
    ),
    ("placeholder", "{history}"),
    ("human", "{question}")
])

# Build a chain that accepts history and a new question.
chain_with_memory = (
    chat_with_memory | model | StrOutputParser()
)

result = chain_with_memory.invoke({
    "history": memory.messages,
    "question": "What was my name again?"
})

print("\n=== Memory-Aware Response ===")
print(result)

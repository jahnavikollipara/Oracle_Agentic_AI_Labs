
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool

# ==========================================
# 1. INITIALIZE YOUR LOCAL MODEL
# ==========================================

model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0
)

# ==========================================
# 2. DEFINE TOOLS
# ==========================================

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide the first number by the second."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


# ==========================================
# 3. CREATE THE AGENT
# ==========================================

tools = [multiply, divide]

agent = create_agent(
    model=model,
    tools=tools
)

# ==========================================
# 4. RUN THE AGENT
# ==========================================

question = "What is 15 multiplied by 8, then divided by 3?"

result = agent.invoke({
    "messages": [
        ("user", question)
    ]
})

print("User:", question)
print("\nAgent response:")
print(result["messages"][-1].content)

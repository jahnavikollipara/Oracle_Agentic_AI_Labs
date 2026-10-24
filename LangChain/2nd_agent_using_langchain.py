
"""
==========================================================
LangChain for AI Agents - Companion Code
Using Qwen 2.5 1.5B through Ollama (local model)
==========================================================

Demonstrates:
- ReAct-style agent execution: Reason -> Act -> Observe
- Creating tools with @tool
- Building an agent using create_agent()
- Displaying tool calls and execution results
"""

# ==========================================================
# STEP 0: IMPORTS
# ==========================================================

import math

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool


# ==========================================================
# STEP 1: INITIALIZE THE MODEL (THE "BRAIN")
# ==========================================================

# Use your locally installed Qwen model through Ollama.
# No OpenAI API key is required.

model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0
)


# ==========================================================
# STEP 2: DEFINE YOUR TOOLS (THE "HANDS")
# ==========================================================

@tool
def add(a: float, b: float) -> float:
    """
    Add two numbers together.
    Use this tool for addition operations.
    """
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers together.
    Use this tool for multiplication operations.
    """
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """
    Divide the first number by the second.
    Raises an error if dividing by zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")

    return a / b


@tool
def square_root(number: float) -> float:
    """
    Calculate the square root of a number.
    Raises an error for negative inputs.
    """
    if number < 0:
        raise ValueError(
            "Cannot take the square root of a negative number"
        )

    return math.sqrt(number)


# Combine all tools into one list.
tools = [add, multiply, divide, square_root]


# Display the tools available to the agent.
print("=== Available Tools ===")

for t in tools:
    print(f"  - {t.name}: {t.description}")

print()


# ==========================================================
# STEP 3: CREATE THE AGENT (THE "LOOP")
# ==========================================================

# The agent decides whether to call a tool.
# LangChain manages the tool execution loop.

agent = create_agent(
    model=model,
    tools=tools
)


# ==========================================================
# STEP 4: RUN THE AGENT
# ==========================================================

def run_agent(question: str):
    """Run the agent and display its execution trace."""

    print(f"\nUser: {question}")
    print("-" * 60)

    # Send the question to the agent.
    result = agent.invoke({
        "messages": [
            ("user", question)
        ]
    })

    print("Agent Execution Trace")
    print("-" * 60)

    step = 1

    # Inspect the messages returned by the agent.
    for msg in result["messages"]:

        # 1. Original user message
        if msg.type == "human":
            print(f"\n{step}. User asked:")
            print(f"   {msg.content}")
            step += 1

        # 2. AI message containing tool calls
        elif msg.type == "ai" and getattr(
            msg, "tool_calls", None
        ):
            for tool_call in msg.tool_calls:
                print(f"\n{step}. Agent decision:")
                print(
                    f"   Tool selected: {tool_call['name']}"
                )
                print(
                    f"   Tool arguments: {tool_call['args']}"
                )
                step += 1

        # 3. Result returned by a tool
        elif msg.type == "tool":
            print(f"\n{step}. Tool observation:")
            print(f"   Tool returned: {msg.content}")
            step += 1

        # 4. Final AI response
        elif msg.type == "ai" and msg.content:
            print(f"\n{step}. Agent response:")
            print(f"   {msg.content}")
            step += 1

    print("\n" + "=" * 60)


# ==========================================================
# STEP 5: TEST THE AGENT
# ==========================================================

if __name__ == "__main__":

    run_agent("What is 42 + 58?")

    run_agent(
        "What is 15 multiplied by 8, then divided by 3?"
    )

    run_agent(
        "What is the square root of 144?"
    )

    run_agent(
        "A rectangle has width 12 and height 7. "
        "What is its area? Use multiplication."
    )


"""
LangChain Agent + MCP Math Server + Ollama Qwen 2.5
"""

import asyncio
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    # 1. Initialize your local Ollama model
    model = ChatOllama(
        model="qwen2.5:1.5b",
        temperature=0
    )

    # 2. Find the MCP server file in this same folder
    script_dir = Path(__file__).resolve().parent
    server_path = script_dir / "mcp_math_server.py"

    # 3. Connect to the MCP server
    client = MultiServerMCPClient({
        "math": {
            "command": "python",
            "args": [str(server_path)],
            "transport": "stdio",
        }
    })

    # 4. Automatically discover the server's tools
    tools = await client.get_tools()

    print("=== MCP Tools Discovered ===")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    # 5. Create the LangChain agent
    agent = create_agent(
        model=model,
        tools=tools
    )

    # 6. Ask the agent a question
    questions = [
        "What is 42 + 58?",
        "What is 15 multiplied by 8, then divided by 3?",
        "What is the square root of 144?"
    ]

    for question in questions:
        print("\n" + "=" * 50)
        print("User:", question)

        result = await agent.ainvoke({
            "messages": [
                {"role": "user", "content": question}
            ]
        })

        print("Agent:", result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())

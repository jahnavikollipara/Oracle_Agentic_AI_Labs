
"""
MCP Math Server
Provides addition, multiplication, division, and square root.
"""

import math
import sys
from mcp.server.fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("Math")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide the first number by the second."""
    if b == 0:
        return "Error: Cannot divide by zero"
    return a / b


@mcp.tool()
def square_root(number: float) -> float:
    """Calculate the square root of a number."""
    if number < 0:
        return "Error: Cannot take the square root of a negative number"
    return math.sqrt(number)


if __name__ == "__main__":
    # MCP uses stdout for communication.
    # Keep diagnostic messages on stderr, if needed.
    mcp.run(transport="stdio")

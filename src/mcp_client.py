import asyncio
import json
import sys
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.write = None
        self.stdio = None
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.gpt_client = OpenAI()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. If a relevant function is available, always use the function call system "
                    "instead of explaining it. Do not describe or suggest function calls — just call the function directly."
                )
            },
            {"role": "user", "content": query}
        ]

        # Step 1: List available tools
        response = await self.session.list_tools()
        functions = []

        for tool in response.tools:
            function = {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
                "strict": False,
            }
            functions.append(function)

        final_text = []

        while True:
            # Step 2: Get model response
            response = self.gpt_client.responses.create(
                model="gpt-4",
                input=messages,
                tools=functions,
                tool_choice="auto",
            )

            # Step 3: Handle model output
            output = response.output[0]  # Assuming one response per round

            if output.type == "message":
                message_content = output.content[0].text
                final_text.append(message_content)
                break

            elif output.type == "function_call":
                # 3a. Prepare assistant tool call message
                assistant_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": output.call_id,
                            "type": "function",
                            "function": {
                                "name": output.name,
                                "arguments": output.arguments
                            }
                        }
                    ]
                }
                messages.append(assistant_message)

                # 3b. Execute the actual tool
                tool_name = output.name
                tool_args = json.loads(output.arguments)
                result = await self.session.call_tool(tool_name, tool_args)

                # Extract text from tool result
                tool_result_text = ""
                for content in result.content:
                    if isinstance(content, TextContent):
                        tool_result_text += content.text
                    else:
                        tool_result_text += str(content)

                # 3c. Append tool output message
                tool_output_message = {
                    "role": "tool",
                    "tool_call_id": output.call_id,
                    "content": tool_result_text
                }
                messages.append(tool_output_message)

            else:
                raise RuntimeError(f"Unexpected output type: {output.type}")

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

# messages.append(
#     {
#         "role": "assistant",
#         "type": "function_call_output",
#         "content": {
#             "type": "output_text",
#             "tool_use_id": output.id,
#             "content": tool_message,
#         },
#     }
# )
# messages.append(
#     {
#         "role": "user",
#         "type": "function_call_output",
#         "content": [
#             {
#                 "type": "output_text",
#                 "tool_use_id": output.id,
#                 "content": tool_message,
#             }
#         ],
#     }
# )
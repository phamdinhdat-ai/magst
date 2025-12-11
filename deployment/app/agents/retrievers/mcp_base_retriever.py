# mcp_base_retriever.py ( client side)

import asyncio
from time import time
from typing import Dict, Any
from pydantic import BaseModel, Field as PydanticField
from loguru import logger

# --- MCP Client Imports ---
# Make sure you have the 'model-context-protocol' library installed:
# pip install model-context-protocol
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

# --- Your BaseAgentTool (assuming it's in a file named base_tool.py) ---
from app.agents.factory.tools.base import BaseAgentTool # Replace with the actual import path


# --- 1. Define the Input Schema for the Remote Tool ---
# Since this tool is a generic wrapper, its input is the dictionary of
# arguments that will be forwarded to the remote tool.
class RemoteToolInput(BaseModel):
    arguments: Dict[str, Any] = PydanticField(
        description="A dictionary containing the arguments for the remote tool."
    )


# --- 2. The Remote MCP Client Tool ---

class RemoteMCPTool(BaseAgentTool):
    """
    A generic agent tool that acts as a client to call a specific tool
    on a remote MCP (Model Context Protocol) server.
    """
    args_schema: type[BaseModel] = RemoteToolInput

    # --- Configuration for this specific tool instance ---
    mcp_server_url: str
    remote_tool_name: str

    def _run(self, arguments: Dict[str, Any]) -> str:
        """
        Synchronously calls the remote tool.
        
        Note: The underlying mcp library is async-native, so this method
        will run the async version in a new event loop.
        """
        logger.info(f"Initiating synchronous call to remote tool '{self.remote_tool_name}'...")
        try:
            # asyncio.run() is the standard way to call an async function from sync code.
            return asyncio.run(self._arun(arguments=arguments))
        except Exception as e:
            logger.error(f"Failed to run sync wrapper for remote tool call: {e}")
            return f"Error: Could not execute remote tool call. {e}"


    async def _arun(self, arguments: Dict[str, Any]) -> str:
        """
        Asynchronously connects to the MCP server and calls the specified remote tool
        with the given arguments.
        """
        logger.info(
            f"Connecting to MCP server at {self.mcp_server_url} to call '{self.remote_tool_name}'"
        )
        try:
            async with sse_client(url=self.mcp_server_url) as vectors:
                async with ClientSession(read_stream=vectors[0], write_stream=vectors[1]) as session:
                    # Initialize the session with the server
                    await session.initialize()
                    
                    # Call the specific tool on the remote server
                    logger.debug(f"Calling remote tool '{self.remote_tool_name}' with args: {arguments}")
                    response = await session.call_tool(
                        self.remote_tool_name,
                        arguments=arguments
                    )

                    # Process the response from the server
                    logger.info(f"Results_response: {response.content}")
                    if response and response.content and len(response.content) > 0:
                        # Assuming the primary content is in the first text block
                        for result in response.content:
                            result_text = result.text
                            logger.debug(f"Tool '{self.remote_tool_name}' returned: {result_text}")
                            logger.success(f"Successfully received response from '{self.remote_tool_name}'")
                            return result_text
                    else:
                        logger.warning(f"Remote tool '{self.remote_tool_name}' returned no content.")
                        return f"Remote tool '{self.remote_tool_name}' executed but returned no content."

        except ConnectionRefusedError:
            error_msg = f"Error: Connection refused. Is the MCP server running at {self.mcp_server_url}?"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"An unexpected error occurred while calling remote tool '{self.remote_tool_name}': {e}"
            logger.error(error_msg)
            return error_msg


# --- 3. Example Usage: How to Create and Use the Tools ---

async def main():
    SERVER_URL = "http://localhost:50051/sse"

    # --- Tool 1: A client for the 'retrieve_vectorestore' remote tool ---
    vector_retrieval_tool = RemoteMCPTool(
        name="remote_vector_search",
        description=(
            "Retrieves data from a remote vector database. "
            "The 'arguments' dictionary must contain a 'query' (string) and "
            "an optional 'collection' (string, e.g., 'customer_data')."
        ),
        mcp_server_url=SERVER_URL,
        remote_tool_name="retrieve_vectorestore"  # This must match the name on the server
    )
    
    # --- Tool 2: A client for the 'file_to_documents' remote tool ---
    file_ingest_tool = RemoteMCPTool(
        name="remote_file_ingestion",
        description=(
            "Sends a file path to a remote server to be ingested into a vector database. "
            "The 'arguments' dictionary must contain a 'file_path' (string)."
        ),
        mcp_server_url=SERVER_URL,
        remote_tool_name="ingest_documents"
    )

    print("--- Using the Remote Vector Search Tool ---")
    query_args = {
        "query": "Thien Huong Tinh cach cua toi",
        "collection": "customer_data2"
    }
    
    # Use the tool's async method
    start_time = time()
    result = await vector_retrieval_tool.arun(arguments=query_args)
    end_time = time()
    print(f"Time taken for retrieval: {end_time - start_time:.4f}s")

    print(f"Query: {query_args['query']}")
    print("\n--- Result from Server ---")
    print(result)
    print("--------------------------\n")

    # A LangChain Agent would now have [vector_retrieval_tool, file_ingest_tool]
    # in its list of available tools.


if __name__ == "__main__":
    asyncio.run(main())
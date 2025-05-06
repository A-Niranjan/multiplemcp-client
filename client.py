import asyncio
import os
import sys
from typing import Optional, List, Dict, Any, Tuple
from contextlib import AsyncExitStack
import json
from os.path import basename # Import basename for cleaner aliases
import hashlib # For unique alias generation
import traceback # For detailed exception printing

# MCP Imports
from mcp import ClientSession, StdioServerParameters, Tool as McpTool
from mcp.client.stdio import stdio_client

# Gemini Imports
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, Tool as GeminiTool, FunctionDeclaration

# Env variable loader
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

# --- Gemini Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables or .env file.")
genai.configure(api_key=GOOGLE_API_KEY)
# --- End Gemini Configuration ---

# Helper to create a unique alias for Gemini tools
def create_server_alias(server_path: str) -> str:
    # Get the immediate parent directory name and the filename without extension
    parent_dir_name = os.path.basename(os.path.dirname(server_path))
    fname_no_ext = basename(server_path).split('.')[0]

    # Combine them, cleaning up characters
    raw_alias = f"{parent_dir_name}_{fname_no_ext}".replace('.', '_').replace(' ', '_').replace('-', '_')
    
    path_hash = hashlib.md5(server_path.encode()).hexdigest()[:6] # Use 6 chars of md5 hash
    
    # Make sure alias is valid (alphanumeric and underscores, not too long if Gemini has limits)
    # For simplicity, let's assume Gemini is flexible. If not, add more cleaning/truncation.
    final_alias = f"{raw_alias}_{path_hash}"
    # Replace any remaining invalid characters (though underscores should be fine)
    final_alias = "".join(c if c.isalnum() or c == '_' else '_' for c in final_alias)
    return final_alias


def convert_mcp_tool_to_gemini(mcp_tool: McpTool, server_alias: str) -> FunctionDeclaration:
    prefixed_name = f"{server_alias}__{mcp_tool.name}"
    gemini_params = {
        "type": "object",
        "properties": mcp_tool.inputSchema.get("properties", {}),
    }
    if "required" in mcp_tool.inputSchema:
        required_fields = mcp_tool.inputSchema["required"]
        if isinstance(required_fields, list) and required_fields:
            gemini_params["required"] = required_fields
        elif isinstance(required_fields, list) and not required_fields:
            pass
        else:
            print(f"Warning: 'required' field in schema for tool '{mcp_tool.name}' from server '{server_alias}' is not a list, skipping.")
    func_decl = FunctionDeclaration(
        name=prefixed_name,
        description=f"[{server_alias}] {mcp_tool.description or f'Tool named {mcp_tool.name}'}",
        parameters=gemini_params
    )
    return func_decl

class MultiMCPClient:
    def __init__(self):
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.tool_map: Dict[str, Tuple[str, str]] = {}
        self.model = genai.GenerativeModel(
            'gemini-1.5-flash-latest',
            generation_config=GenerationConfig(temperature=0.7)
        )
        self.chat_session = None

    async def connect_to_server(self, server_script_path: str):
        if server_script_path in self.servers:
            print(f"Already connected to server: {server_script_path}")
            return

        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            print(f"Warning: Server script '{server_script_path}' is not a .py or .js file. Skipping.")
            return

        command = "python" if is_python else "node"
        print(f"\nAttempting to connect to server: {server_script_path} using '{command}'...")
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        exit_stack = AsyncExitStack()
        try:
            print(f"DEBUG: [{server_script_path}] stdio_client about to be called.")
            stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
            stdio_reader, stdio_writer = stdio_transport
            print(f"DEBUG: [{server_script_path}] ClientSession about to be called.")
            session = await exit_stack.enter_async_context(ClientSession(stdio_reader, stdio_writer))
            print(f"DEBUG: [{server_script_path}] session.initialize about to be called.")
            await session.initialize()
            print(f"DEBUG: [{server_script_path}] session.initialize successful.")

            print(f"DEBUG: [{server_script_path}] session.list_tools about to be called.")
            response = await session.list_tools()
            mcp_tools = response.tools
            print(f"DEBUG: [{server_script_path}] session.list_tools successful, found {len(mcp_tools)} tools.")


            server_alias = create_server_alias(server_script_path)
            print(f"DEBUG: [{server_script_path}] Generated alias: {server_alias}")
            gemini_func_declarations = []
            server_tools_count = 0
            for tool in mcp_tools:
                try:
                    gemini_func = convert_mcp_tool_to_gemini(tool, server_alias)
                    gemini_func_declarations.append(gemini_func)
                    # Check for alias collision before assignment
                    if gemini_func.name in self.tool_map:
                        print(f"CRITICAL WARNING: Tool name collision for '{gemini_func.name}'. Overwriting previous entry from server '{self.tool_map[gemini_func.name][0]}' with new entry from '{server_script_path}'. This indicates an issue with alias generation or identical tool names from different servers that alias didn't uniquely separate.")
                    self.tool_map[gemini_func.name] = (server_script_path, tool.name)
                    server_tools_count += 1
                except Exception as conversion_err:
                     print(f"Error converting tool '{tool.name}' from server '{server_script_path}': {conversion_err}")

            self.servers[server_script_path] = {
                "session": session,
                "exit_stack": exit_stack,
                "alias": server_alias,
                "gemini_declarations": gemini_func_declarations,
                "stdio_reader": stdio_reader,
                "stdio_writer": stdio_writer,
            }
            print(f"Successfully connected to '{server_script_path}' (alias: '{server_alias}'). Found {server_tools_count} tools.")

            if not self.chat_session:
                print("\nInitializing Gemini chat session...")
                self.chat_session = self.model.start_chat(enable_automatic_function_calling=False)
                print("Gemini chat session initialized.")

        except Exception as e:
            print(f"\nError connecting to or initializing server '{server_script_path}':")
            traceback.print_exc() # Print full traceback for connection errors
            await exit_stack.aclose() # Ensure cleanup on partial failure
            raise # Re-raise to indicate connection failure for this server

    def _get_all_gemini_tools(self) -> Optional[List[GeminiTool]]:
        all_declarations = []
        for server_path, data in self.servers.items():
            all_declarations.extend(data.get("gemini_declarations", []))
        if not all_declarations:
            return None
        return [GeminiTool(function_declarations=all_declarations)]

    async def process_query(self, query: str) -> str:
        if not self.servers:
            return "Error: Not connected to any MCP servers."
        if not self.chat_session:
            return "Error: Gemini chat session not initialized."

        gemini_tools = self._get_all_gemini_tools()
        try:
            response = await self.chat_session.send_message_async(
                query,
                tools=gemini_tools
            )
            while response.candidates and response.candidates[0].content.parts and response.candidates[0].content.parts[0].function_call.name:
                function_call = response.candidates[0].content.parts[0].function_call
                prefixed_tool_name = function_call.name
                tool_args = dict(function_call.args)

                if prefixed_tool_name not in self.tool_map:
                    print(f"Error: Gemini requested unknown tool '{prefixed_tool_name}'.")
                    error_message = f"Tool '{prefixed_tool_name}' not found among registered tools."
                    tool_response_part = genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=prefixed_tool_name, response={"error": error_message}))
                else:
                    server_path, original_tool_name = self.tool_map[prefixed_tool_name]
                    if server_path not in self.servers:
                         print(f"Error: Server '{server_path}' for tool '{prefixed_tool_name}' is not connected.")
                         error_message = f"Server '{server_path}' required for tool '{original_tool_name}' is disconnected."
                         tool_response_part = genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=prefixed_tool_name, response={"error": error_message}))
                    else:
                         server_data = self.servers[server_path]
                         session = server_data["session"]
                         server_alias = server_data["alias"]
                         try:
                             mcp_result = await session.call_tool(original_tool_name, tool_args)
                             extracted_data = None
                             json_string = None
                             if (isinstance(mcp_result.content, list) and
                                 len(mcp_result.content) == 1 and
                                 type(mcp_result.content[0]).__name__ == 'TextContent' and
                                 hasattr(mcp_result.content[0], 'text')):
                                 json_string = mcp_result.content[0].text
                             elif mcp_result.content is not None:
                                 print(f">>> WARNING ({server_alias}): Unexpected content type: {type(mcp_result.content)}. Attempting direct use.")
                                 if isinstance(mcp_result.content, (dict, list, str, int, float, bool)):
                                     extracted_data = mcp_result.content
                                 else:
                                     error_msg = f"Received unserializable content type {type(mcp_result.content)} from tool."
                                     print(f">>> ERROR ({server_alias}): {error_msg}")
                                     extracted_data = {"error": error_msg}
                             if json_string:
                                 try:
                                     extracted_data = json.loads(json_string)
                                 except json.JSONDecodeError as json_err:
                                     print(f">>> WARNING ({server_alias}): Failed to parse JSON: {json_err}. Passing raw string.")
                                     extracted_data = json_string
                             api_response = {"content": extracted_data}
                             tool_response_part = genai.protos.Part(
                                 function_response=genai.protos.FunctionResponse(
                                     name=prefixed_tool_name, response=api_response))
                         except Exception as tool_error:
                             print(f"Error during MCP tool execution or result processing for '{original_tool_name}' on '{server_alias}': {tool_error}")
                             tool_response_part = genai.protos.Part(
                                 function_response=genai.protos.FunctionResponse(
                                     name=prefixed_tool_name,
                                     response={"error": f"MCP client failed to execute tool '{original_tool_name}' or process its result: {str(tool_error)}"}))
                try:
                    response = await self.chat_session.send_message_async(
                        tool_response_part, tools=gemini_tools)
                except Exception as send_error:
                     print(f"Error sending tool response/error back to Gemini: {send_error}")
                     return f"Error: Failed to send tool result back to Gemini ({send_error}). Cannot get final response."
            final_text = ""
            if response and response.candidates:
                 final_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            return final_text
        except Exception as e:
            print(f"\nError during Gemini interaction:")
            traceback.print_exc() # Print full traceback for Gemini errors
            return f"Error processing query with Gemini. See console for details."

    async def chat_loop(self):
        if not self.servers:
             print("\nWarning: No servers connected successfully. Cannot start chat.")
             return
        print("\n--- Multi-Server MCP Client with Gemini Started! ---")
        print(f"Connected to {len(self.servers)} server(s).")
        print("Type your queries or 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query: continue
                if query.lower() == 'quit':
                    print("Exiting...")
                    break
                response = await self.process_query(query)
                print("\nResponse:")
                print(response)
            except KeyboardInterrupt:
                 print("\nExiting...")
                 break
            except Exception as e:
                print(f"\nAn unexpected error occurred in the chat loop:")
                traceback.print_exc() # This will print the full traceback

    async def cleanup(self):
        print("\nCleaning up resources...")
        if not self.servers:
             print("No servers were connected.")
             return
        server_paths = list(self.servers.keys())
        for server_path in server_paths:
            print(f"Closing connection to '{server_path}'...")
            server_data = self.servers.pop(server_path, None)
            if server_data and "exit_stack" in server_data:
                try:
                    await server_data["exit_stack"].aclose()
                    print(f"Connection to '{server_path}' closed.")
                except Exception as e:
                    print(f"Error cleaning up resources for '{server_path}': {e}")
            else:
                 print(f"No exit stack found for '{server_path}' during cleanup.")
        self.tool_map.clear()
        print("All server resources cleaned up.")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server1.[py|js]> [path_to_server2.[py|js]] ...")
        sys.exit(1)
    server_paths = sys.argv[1:]
    client = MultiMCPClient()
    connected_count = 0
    try:
        print(f"Attempting to connect to {len(server_paths)} server(s)...")
        for i, server_path in enumerate(server_paths):
             print(f"\n--- Connecting to server {i+1}/{len(server_paths)}: {server_path} ---")
             try:
                 await client.connect_to_server(server_path)
                 connected_count +=1
             except Exception as e:
                  # Error is already printed with traceback in connect_to_server
                  print(f"--- Failed to connect to server: {server_path}. Continuing with next server if any. ---")
        if connected_count > 0:
            await client.chat_loop()
        else:
            print("\nNo servers connected successfully. Exiting.")
    except Exception as e:
        print(f"\nFatal error during setup or chat:")
        traceback.print_exc()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nApplication terminated with error:")
        traceback.print_exc()
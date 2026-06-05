"""
Notes MCP Server — Python implementation
=========================================
A minimal but complete MCP server that demonstrates:
  - Tool registration (add_note, list_notes)
  - Resource exposure (notes://all)
  - Prompt templates (summarize_notes)
  - stdio transport (default for local servers)

Requirements:
    pip install mcp

Run directly (stdio transport — connect from Claude Desktop or mcp CLI):
    python notes_mcp_server.py

Test with the MCP CLI:
    pip install mcp[cli]
    mcp dev notes_mcp_server.py
"""

from datetime import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import asyncio

# ─────────────────────────────────────────────
# 1. In-memory data store
# ─────────────────────────────────────────────

notes: list[dict] = []  # Each note: { id, title, content, created_at }


def _next_id() -> int:
    return len(notes) + 1


# ─────────────────────────────────────────────
# 2. Create the MCP Server instance
# ─────────────────────────────────────────────

app = Server("notes-server")


# ─────────────────────────────────────────────
# 3. Tool definitions
#    The LLM sees the name + description + inputSchema
#    and decides when to call them.
# ─────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """Advertise available tools to the MCP client."""
    return [
        types.Tool(
            name="add_note",
            description=(
                "Save a new note with a title and content. "
                "Use this whenever the user wants to remember something."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "A short, descriptive title for the note.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full body/content of the note.",
                    },
                },
                "required": ["title", "content"],
            },
        ),
        types.Tool(
            name="list_notes",
            description=(
                "Return a summary list of all saved notes (id, title, timestamp). "
                "Use this to show the user what notes exist."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        types.Tool(
            name="get_note",
            description=(
                "Retrieve the full content of a specific note by its id."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The numeric id of the note to retrieve.",
                    }
                },
                "required": ["id"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Dispatch incoming tool calls to handler functions.
    Must return a list of content blocks.
    """
    if name == "add_note":
        title = arguments["title"]
        content = arguments["content"]
        note = {
            "id": _next_id(),
            "title": title,
            "content": content,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        notes.append(note)
        return [
            types.TextContent(
                type="text",
                text=f"✓ Note saved (id={note['id']}): "{title}"",
            )
        ]

    elif name == "list_notes":
        if not notes:
            return [types.TextContent(type="text", text="No notes yet.")]
        lines = [f"id={n['id']} | {n['created_at']} | {n['title']}" for n in notes]
        return [types.TextContent(type="text", text="\n".join(lines))]

    elif name == "get_note":
        note_id = arguments["id"]
        note = next((n for n in notes if n["id"] == note_id), None)
        if note is None:
            return [types.TextContent(type="text", text=f"Note id={note_id} not found.")]
        text = f"# {note['title']}\n_Saved: {note['created_at']}_\n\n{note['content']}"
        return [types.TextContent(type="text", text=text)]

    else:
        raise ValueError(f"Unknown tool: {name}")


# ─────────────────────────────────────────────
# 4. Resource definitions
#    Resources are data sources the client can
#    read and inject into the model's context.
# ─────────────────────────────────────────────

@app.list_resources()
async def list_resources() -> list[types.Resource]:
    """Advertise available resources."""
    return [
        types.Resource(
            uri="notes://all",
            name="All notes",
            description="Every saved note as a Markdown document.",
            mimeType="text/markdown",
        )
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Return the content of a requested resource URI."""
    if uri == "notes://all":
        if not notes:
            return "_No notes saved yet._"
        sections = []
        for n in notes:
            sections.append(
                f"## {n['title']}\n"
                f"_id={n['id']} | saved: {n['created_at']}_\n\n"
                f"{n['content']}"
            )
        return "\n\n---\n\n".join(sections)
    raise ValueError(f"Unknown resource URI: {uri}")


# ─────────────────────────────────────────────
# 5. Prompt templates
#    Pre-built prompt patterns the client can
#    fetch and inject into the conversation.
# ─────────────────────────────────────────────

@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    """Advertise available prompt templates."""
    return [
        types.Prompt(
            name="summarize_notes",
            description="Produce a concise summary of all saved notes.",
            arguments=[],
        )
    ]


@app.get_prompt()
async def get_prompt(
    name: str, arguments: dict | None
) -> types.GetPromptResult:
    """Return a rendered prompt template."""
    if name == "summarize_notes":
        all_notes_md = await read_resource("notes://all")
        return types.GetPromptResult(
            description="Summarize all notes",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=(
                            "Here are my notes:\n\n"
                            f"{all_notes_md}\n\n"
                            "Please give me a concise bullet-point summary."
                        ),
                    ),
                )
            ],
        )
    raise ValueError(f"Unknown prompt: {name}")


# ─────────────────────────────────────────────
# 6. Entry point — run over stdio transport
# ─────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())

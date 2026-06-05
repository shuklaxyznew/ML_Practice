# Notes MCP Server — Setup & Testing Guide

## Python Setup

### Install dependencies
```bash
pip install mcp
# or, for the dev CLI:
pip install "mcp[cli]"
```

### Run the server (for Claude Desktop)
```bash
python notes_mcp_server.py
```

### Test interactively with the MCP dev inspector
```bash
mcp dev notes_mcp_server.py
```
This opens a browser UI where you can call tools, read resources, and get prompts manually.

### Connect to Claude Desktop
Add this block to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "notes": {
      "command": "python",
      "args": ["/absolute/path/to/notes_mcp_server.py"]
    }
  }
}
```

Restart Claude Desktop. You'll see the notes server listed in the MCP panel.

---

## Java Setup

### pom.xml dependency
```xml
<dependency>
    <groupId>io.modelcontextprotocol.sdk</groupId>
    <artifactId>mcp</artifactId>
    <version>0.9.0</version>
</dependency>
```

### Build and run
```bash
mvn compile
mvn exec:java -Dexec.mainClass="NotesMcpServer"
```

### Connect to Claude Desktop
```json
{
  "mcpServers": {
    "notes-java": {
      "command": "java",
      "args": ["-cp", "/path/to/target/classes", "NotesMcpServer"]
    }
  }
}
```

---

## What you can test

Once connected, ask Claude:

| Prompt | What triggers |
|--------|---------------|
| "Save a note called 'MCP ideas' with content 'Build a database server'" | `add_note` tool |
| "What notes do I have?" | `list_notes` tool |
| "Show me note number 1 in full" | `get_note` tool |
| "Read my notes resource and summarize" | `notes://all` resource |
| "Use the summarize_notes prompt" | `summarize_notes` prompt template |

---

## How the MCP handshake works (sequence)

```
Client → Server:   initialize (client info, protocol version)
Server → Client:   initialize result (server info, capabilities)
Client → Server:   initialized (acknowledgement)
Client → Server:   tools/list
Server → Client:   [add_note, list_notes, get_note]
Client → Server:   resources/list
Server → Client:   [notes://all]
...
User asks Claude a question
Claude decides to call add_note
Client → Server:   tools/call  { name: "add_note", arguments: {...} }
Server → Client:   CallToolResult { content: [TextContent] }
Claude uses result in its response
```

---

## Next steps to extend this server

1. **Persistence** — swap the in-memory list for SQLite (`sqlite3` in Python, JDBC in Java)
2. **Search tool** — add a `search_notes(query)` tool using simple string matching or embedding similarity
3. **Tags** — extend the Note schema with `tags: list[str]` and add a `filter_by_tag` tool
4. **SSE transport** — expose the server over HTTP for remote/shared access
5. **Authentication** — add OAuth 2.0 token validation at the transport layer

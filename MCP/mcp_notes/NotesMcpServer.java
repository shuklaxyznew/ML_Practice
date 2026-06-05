/**
 * Notes MCP Server — Java implementation
 * ========================================
 * A minimal but complete MCP server that demonstrates:
 *   - Tool registration  (add_note, list_notes, get_note)
 *   - Resource exposure  (notes://all)
 *   - Prompt templates   (summarize_notes)
 *   - stdio transport    (default for local servers)
 *
 * Dependencies (Maven):
 * ─────────────────────
 * <dependency>
 *     <groupId>io.modelcontextprotocol.sdk</groupId>
 *     <artifactId>mcp</artifactId>
 *     <version>0.9.0</version>   <!-- use latest from Maven Central -->
 * </dependency>
 *
 * Build & run:
 *   mvn compile exec:java -Dexec.mainClass="NotesMcpServer"
 *
 * Or with the MCP Inspector (requires Node):
 *   npx @modelcontextprotocol/inspector java -cp target/classes NotesMcpServer
 */

import io.modelcontextprotocol.server.McpServer;
import io.modelcontextprotocol.server.McpSyncServer;
import io.modelcontextprotocol.server.transport.StdioServerTransportProvider;
import io.modelcontextprotocol.spec.McpSchema;

import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public class NotesMcpServer {

    // ─────────────────────────────────────────────
    // 1. In-memory data model
    // ─────────────────────────────────────────────

    record Note(int id, String title, String content, String createdAt) {}

    private static final List<Note> notes = new ArrayList<>();
    private static final AtomicInteger idCounter = new AtomicInteger(0);

    private static String now() {
        return DateTimeFormatter.ISO_INSTANT.format(Instant.now());
    }

    // ─────────────────────────────────────────────
    // 2. Helper: build the full notes Markdown
    // ─────────────────────────────────────────────

    private static String buildAllNotesMarkdown() {
        if (notes.isEmpty()) return "_No notes saved yet._";
        return notes.stream()
                .map(n -> "## " + n.title() + "\n"
                        + "_id=" + n.id() + " | saved: " + n.createdAt() + "_\n\n"
                        + n.content())
                .collect(Collectors.joining("\n\n---\n\n"));
    }

    // ─────────────────────────────────────────────
    // 3. Tool handlers
    // ─────────────────────────────────────────────

    /** add_note: saves a new note, returns confirmation. */
    private static McpSchema.CallToolResult handleAddNote(Map<String, Object> args) {
        String title   = (String) args.get("title");
        String content = (String) args.get("content");

        if (title == null || content == null) {
            return errorResult("Both 'title' and 'content' are required.");
        }

        Note note = new Note(idCounter.incrementAndGet(), title, content, now());
        notes.add(note);

        return McpSchema.CallToolResult.builder()
                .addTextContent("✓ Note saved (id=" + note.id() + "): \"" + title + "\"")
                .build();
    }

    /** list_notes: returns a summary table of all notes. */
    private static McpSchema.CallToolResult handleListNotes() {
        if (notes.isEmpty()) {
            return McpSchema.CallToolResult.builder()
                    .addTextContent("No notes yet.")
                    .build();
        }
        String summary = notes.stream()
                .map(n -> "id=" + n.id() + " | " + n.createdAt() + " | " + n.title())
                .collect(Collectors.joining("\n"));
        return McpSchema.CallToolResult.builder()
                .addTextContent(summary)
                .build();
    }

    /** get_note: returns full content of a note by id. */
    private static McpSchema.CallToolResult handleGetNote(Map<String, Object> args) {
        Object idArg = args.get("id");
        if (idArg == null) return errorResult("'id' is required.");

        int id = ((Number) idArg).intValue();
        return notes.stream()
                .filter(n -> n.id() == id)
                .findFirst()
                .map(n -> McpSchema.CallToolResult.builder()
                        .addTextContent("# " + n.title() + "\n_Saved: "
                                + n.createdAt() + "_\n\n" + n.content())
                        .build())
                .orElse(errorResult("Note id=" + id + " not found."));
    }

    private static McpSchema.CallToolResult errorResult(String message) {
        return McpSchema.CallToolResult.builder()
                .addTextContent("Error: " + message)
                .isError(true)
                .build();
    }

    // ─────────────────────────────────────────────
    // 4. main — wire up and start the server
    // ─────────────────────────────────────────────

    public static void main(String[] args) {

        // 4a. Transport: stdio (stdin → server, server → stdout)
        var transportProvider = new StdioServerTransportProvider();

        // 4b. Server info advertised during initialization handshake
        var serverInfo = new McpSchema.Implementation("notes-server", "1.0.0");

        // 4c. Build the synchronous MCP server
        //     (McpSyncServer wraps async internals with a simpler blocking API)
        McpSyncServer server = McpServer.sync(transportProvider)
                .serverInfo(serverInfo)

                // ── TOOLS ───────────────────────────────────────────────────

                .tool(
                    McpSchema.Tool.builder()
                        .name("add_note")
                        .description("Save a new note with a title and content. " +
                                     "Use whenever the user wants to remember something.")
                        .inputSchema(McpSchema.JsonSchema.builder()
                            .type("object")
                            .properties(Map.of(
                                "title",   Map.of("type", "string",
                                                  "description", "Short descriptive title."),
                                "content", Map.of("type", "string",
                                                  "description", "Full body of the note.")
                            ))
                            .required(List.of("title", "content"))
                            .build())
                        .build(),
                    (exchange, toolArgs) -> handleAddNote(toolArgs.arguments())
                )

                .tool(
                    McpSchema.Tool.builder()
                        .name("list_notes")
                        .description("Return a summary of all saved notes (id, title, timestamp).")
                        .inputSchema(McpSchema.JsonSchema.builder()
                            .type("object")
                            .properties(Map.of())
                            .required(List.of())
                            .build())
                        .build(),
                    (exchange, toolArgs) -> handleListNotes()
                )

                .tool(
                    McpSchema.Tool.builder()
                        .name("get_note")
                        .description("Retrieve the full content of a note by its id.")
                        .inputSchema(McpSchema.JsonSchema.builder()
                            .type("object")
                            .properties(Map.of(
                                "id", Map.of("type", "integer",
                                             "description", "Numeric id of the note.")
                            ))
                            .required(List.of("id"))
                            .build())
                        .build(),
                    (exchange, toolArgs) -> handleGetNote(toolArgs.arguments())
                )

                // ── RESOURCES ───────────────────────────────────────────────

                .resource(
                    McpSchema.Resource.builder()
                        .uri("notes://all")
                        .name("All notes")
                        .description("Every saved note rendered as a Markdown document.")
                        .mimeType("text/markdown")
                        .build(),
                    (exchange, request) -> {
                        String markdown = buildAllNotesMarkdown();
                        return McpSchema.ReadResourceResult.builder()
                                .addTextContent(request.uri(), "text/markdown", markdown)
                                .build();
                    }
                )

                // ── PROMPT TEMPLATES ─────────────────────────────────────────

                .prompt(
                    McpSchema.Prompt.builder()
                        .name("summarize_notes")
                        .description("Produce a concise summary of all saved notes.")
                        .arguments(List.of())
                        .build(),
                    (exchange, request) -> {
                        String allNotes = buildAllNotesMarkdown();
                        return McpSchema.GetPromptResult.builder()
                                .description("Summarize all notes")
                                .addUserMessage(
                                    "Here are my notes:\n\n" + allNotes +
                                    "\n\nPlease give me a concise bullet-point summary."
                                )
                                .build();
                    }
                )

                .build();

        // 4d. Keep the server running until stdin closes
        //     (the MCP client controls the lifecycle)
        System.err.println("Notes MCP Server running on stdio. Waiting for client...");
        server.closeGracefully();
    }
}

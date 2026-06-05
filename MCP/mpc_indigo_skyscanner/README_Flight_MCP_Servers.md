# ✈️ Flight MCP Servers — Indigo & Skyscanner

A pair of MCP (Model Context Protocol) servers that let Claude search and book flights via **IndiGo Airlines** and **Skyscanner** (multi-airline aggregator).

---

## 📁 Project Structure

```
flight-mcp/
├── indigo_mcp_server.py        # IndiGo-specific MCP server
├── skyscanner_mcp_server.py    # Skyscanner multi-airline MCP server
├── flight_servers_setup.py     # Claude Desktop config + test script
└── README.md                   # This file
```

---

## ⚙️ Requirements

- Python 3.10+
- pip packages:

```bash
pip install mcp httpx
```

---

## 🔑 API Keys

### IndiGo API
1. Register at https://developer.goindigo.in
2. Get your API key from the developer dashboard
3. Set environment variable:
```bash
export INDIGO_API_KEY="your_key_here"
```

### Skyscanner API (via RapidAPI)
1. Register at https://rapidapi.com
2. Subscribe to **Skyscanner Flight Search** API
3. Set environment variable:
```bash
export RAPIDAPI_KEY="your_key_here"
```

> **Note:** Both servers work with mock data out of the box — you don't need real API keys to test locally.

---

## 🚀 Running the Servers

### Option 1 — MCP Dev Inspector (recommended for testing)
Opens a browser UI where you can call tools manually.

```bash
# Install MCP CLI
pip install "mcp[cli]"

# Run Indigo server
mcp dev indigo_mcp_server.py

# Run Skyscanner server (new terminal)
mcp dev skyscanner_mcp_server.py
```

### Option 2 — stdio (for Claude Desktop)
```bash
python indigo_mcp_server.py
python skyscanner_mcp_server.py
```

---

## 🖥️ Connect to Claude Desktop

### Step 1 — Find the config file
| OS      | Path |
|---------|------|
| macOS   | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux   | `~/.config/Claude/claude_desktop_config.json` |

### Step 2 — Add both servers to the config

```json
{
  "mcpServers": {
    "indigo": {
      "command": "python",
      "args": ["/absolute/path/to/indigo_mcp_server.py"],
      "env": {
        "INDIGO_API_KEY": "your_indigo_api_key_here"
      }
    },
    "skyscanner": {
      "command": "python",
      "args": ["/absolute/path/to/skyscanner_mcp_server.py"],
      "env": {
        "RAPIDAPI_KEY": "your_rapidapi_key_here"
      }
    }
  }
}
```

> ⚠️ Use **absolute paths** — relative paths will not work.

### Step 3 — Restart Claude Desktop
Both servers will appear in the MCP panel (🔨 hammer icon).

---

## 🛠️ Available Tools

### IndiGo Server (`indigo_mcp_server.py`)

| Tool | Description | Required Inputs |
|------|-------------|-----------------|
| `indigo_search_flights` | Search IndiGo flights by route and date | `origin`, `destination`, `date` |
| `indigo_book_flight` | Book a flight and get a booking reference | `flight_id`, `passenger_name`, `passenger_email` |
| `indigo_list_bookings` | List all bookings made in this session | none |

### Skyscanner Server (`skyscanner_mcp_server.py`)

| Tool | Description | Required Inputs |
|------|-------------|-----------------|
| `sky_search_flights` | Search all airlines, sort by price or duration | `origin`, `destination`, `date` |
| `sky_get_prices` | Full fare breakdown for a specific itinerary | `itinerary_id` |
| `sky_compare_flights` | Compare cheapest vs fastest flight side-by-side | `origin`, `destination`, `date` |

---

## 📦 Available Resources

| Server | URI | Content |
|--------|-----|---------|
| IndiGo | `indigo://bookings` | All confirmed bookings as JSON |
| Skyscanner | `skyscanner://quotes` | All fetched price quotes as JSON |

---

## 💬 Example Prompts in Claude

Once both servers are connected, try these:

```
"Search IndiGo flights from Bangalore to Delhi on July 10th"

"Compare all airlines from BLR to DEL — cheapest vs fastest"

"Show only non-stop flights from Mumbai to Chennai"

"Book flight 6E-2401 for Rahul Verma, email rahul@example.com"

"What are the taxes and baggage policy for the Vistara option?"

"List all my IndiGo bookings"
```

---

## 🧪 Run Tests Locally (no Claude needed)

```bash
python test_flight_servers.py
```

Expected output:
```
==================================================
INDIGO SERVER TESTS
==================================================

[1] Search BLR → DEL
  6E-2401 | ₹4,850 | 165min
  6E-5512 | ₹5,200 | 160min

[2] Book flight 6E-2401
  Booking ref: 6E-0001

[3] Book invalid flight
  Result: {'error': 'Flight 6E-9999 not found.'}

==================================================
SKYSCANNER SERVER TESTS
==================================================

[1] Search BLR → DEL (sorted by price)
  SKY-BLR-DEL-003 | ['SpiceJet', 'GoAir'] | ₹3,200 | 1 stops
  SKY-BLR-DEL-001 | ['IndiGo'] | ₹4,850 | 0 stops
  SKY-BLR-DEL-002 | ['Air India'] | ₹5,600 | 0 stops

[2] Get prices for SKY-BLR-DEL-001
  Total: ₹4,850
  Baggage: 7 kg cabin. 15 kg check-in included.

[3] Compare BLR → DEL
  Cheapest: ₹3,200 | ['SpiceJet', 'GoAir']
  Fastest:  160min | ['IndiGo']

[4] Non-stop BLR → DEL only
  Found 2 non-stop options

✅ All tests passed.
```

---

## 🔌 Swapping Mock Data for Real API Calls

Both servers are pre-wired for real API calls — just uncomment the `httpx` sections.

### IndiGo (`indigo_mcp_server.py`)
Find the comment block in `_search_indigo()` and `_book_indigo()` and replace mock returns with:
```python
import httpx
async with httpx.AsyncClient() as client:
    resp = await client.get(
        f"https://api.goindigo.in/v1/flights/search",
        headers={"X-API-Key": INDIGO_API_KEY},
        params={"origin": origin, "dest": destination, "date": date}
    )
    return resp.json()["flights"]
```

### Skyscanner (`skyscanner_mcp_server.py`)
Replace mock return in `_sky_search()` with:
```python
import httpx
async with httpx.AsyncClient() as client:
    resp = await client.get(
        "https://skyscanner-flight-search.p.rapidapi.com/apiservices/browsequotes/v1.0/IN/INR/en-IN/{origin}/{destination}/{date}",
        headers={
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": "skyscanner-flight-search.p.rapidapi.com"
        }
    )
    return resp.json()["Quotes"]
```

---

## 🗺️ IATA Airport Codes (India)

| City | Code |
|------|------|
| Bengaluru | BLR |
| Delhi | DEL |
| Mumbai | BOM |
| Chennai | MAA |
| Hyderabad | HYD |
| Kolkata | CCU |
| Pune | PNQ |
| Ahmedabad | AMD |
| Goa | GOI |
| Kochi | COK |

---

## 🔮 Extending the Servers

| Feature | How |
|---------|-----|
| Round-trip search | Add `return_date` param to search tools |
| Seat selection | New tool: `indigo_select_seat(booking_ref, seat)` |
| Price alerts | Background task polling Skyscanner, notify via resource update |
| Multi-city | Extend input schema with a `legs: list` parameter |
| SSE transport | Replace `stdio_server` with `SseServerTransport` for remote access |
| Auth | Add OAuth2 token validation middleware at transport layer |

---

## 📄 License

MIT — free to use, modify, and distribute.

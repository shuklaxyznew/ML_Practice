"""
IndiGo Airlines MCP Server
============================
Exposes IndiGo flight search and booking as MCP tools.

In production, replace the mock functions with real IndiGo API calls.
IndiGo official API: https://developer.goindigo.in  (requires API key)

Install:
    pip install mcp httpx

Run (stdio — used by Claude Desktop):
    python indigo_mcp_server.py

Environment variables:
    INDIGO_API_KEY   — your IndiGo developer API key
"""

import asyncio
import os
import json
from datetime import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ── optional: real HTTP client ─────────────────────────────────────
# import httpx
# INDIGO_API_KEY = os.getenv("INDIGO_API_KEY", "")
# INDIGO_BASE    = "https://api.goindigo.in/v1"

app = Server("indigo-server")

# ─────────────────────────────────────────────────────────────────────
# MOCK DATA  (replace with real httpx calls against IndiGo API)
# ─────────────────────────────────────────────────────────────────────

MOCK_FLIGHTS = [
    {
        "flight_id":   "6E-2401",
        "airline":     "IndiGo",
        "origin":      "BLR",
        "destination": "DEL",
        "departure":   "2025-07-10T06:00:00",
        "arrival":     "2025-07-10T08:45:00",
        "duration_min": 165,
        "price_inr":   4850,
        "seats_left":  12,
        "class":       "Economy",
    },
    {
        "flight_id":   "6E-5512",
        "airline":     "IndiGo",
        "origin":      "BLR",
        "destination": "DEL",
        "departure":   "2025-07-10T14:30:00",
        "arrival":     "2025-07-10T17:10:00",
        "duration_min": 160,
        "price_inr":   5200,
        "seats_left":  4,
        "class":       "Economy",
    },
    {
        "flight_id":   "6E-8810",
        "airline":     "IndiGo",
        "origin":      "BLR",
        "destination": "BOM",
        "departure":   "2025-07-10T08:00:00",
        "arrival":     "2025-07-10T09:40:00",
        "duration_min": 100,
        "price_inr":   3100,
        "seats_left":  22,
        "class":       "Economy",
    },
]

# Store confirmed bookings in memory
bookings: list[dict] = []


def _search_indigo(origin: str, destination: str, date: str) -> list[dict]:
    """
    Mock flight search. Replace with:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{INDIGO_BASE}/flights/search",
                headers={"X-API-Key": INDIGO_API_KEY},
                params={"origin": origin, "dest": destination, "date": date}
            )
            return resp.json()["flights"]
    """
    origin = origin.upper()
    destination = destination.upper()
    return [
        f for f in MOCK_FLIGHTS
        if f["origin"] == origin and f["destination"] == destination
    ]


def _book_indigo(flight_id: str, passenger_name: str, passenger_email: str) -> dict:
    """
    Mock booking. Replace with a POST to the IndiGo bookings endpoint.
    """
    flight = next((f for f in MOCK_FLIGHTS if f["flight_id"] == flight_id), None)
    if not flight:
        return {"error": f"Flight {flight_id} not found."}

    booking_ref = f"6E-{len(bookings)+1:04d}"
    booking = {
        "booking_ref":      booking_ref,
        "flight_id":        flight_id,
        "passenger_name":   passenger_name,
        "passenger_email":  passenger_email,
        "origin":           flight["origin"],
        "destination":      flight["destination"],
        "departure":        flight["departure"],
        "price_inr":        flight["price_inr"],
        "booked_at":        datetime.now().isoformat(timespec="seconds"),
        "status":           "CONFIRMED",
    }
    bookings.append(booking)
    return booking


# ─────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="indigo_search_flights",
            description=(
                "Search IndiGo airline flights between two Indian airports on a given date. "
                "Returns available flights with price in INR, duration, and seats left. "
                "Use IATA airport codes (e.g. BLR, DEL, BOM, MAA, HYD, CCU)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "3-letter IATA code of departure airport. E.g. 'BLR'.",
                    },
                    "destination": {
                        "type": "string",
                        "description": "3-letter IATA code of arrival airport. E.g. 'DEL'.",
                    },
                    "date": {
                        "type": "string",
                        "description": "Travel date in YYYY-MM-DD format.",
                    },
                },
                "required": ["origin", "destination", "date"],
            },
        ),
        types.Tool(
            name="indigo_book_flight",
            description=(
                "Book an IndiGo flight using its flight_id. "
                "Requires passenger name and email. Returns a booking reference."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "flight_id": {
                        "type": "string",
                        "description": "Flight ID from indigo_search_flights results. E.g. '6E-2401'.",
                    },
                    "passenger_name": {
                        "type": "string",
                        "description": "Full name of the passenger.",
                    },
                    "passenger_email": {
                        "type": "string",
                        "description": "Passenger's email for e-ticket delivery.",
                    },
                },
                "required": ["flight_id", "passenger_name", "passenger_email"],
            },
        ),
        types.Tool(
            name="indigo_list_bookings",
            description="List all IndiGo flight bookings made in this session.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "indigo_search_flights":
        flights = _search_indigo(
            arguments["origin"],
            arguments["destination"],
            arguments["date"],
        )
        if not flights:
            return [types.TextContent(
                type="text",
                text=f"No IndiGo flights found from {arguments['origin']} "
                     f"to {arguments['destination']} on {arguments['date']}."
            )]
        # Format as readable table
        lines = ["IndiGo flights found:\n"]
        for f in flights:
            lines.append(
                f"✈ {f['flight_id']}  |  "
                f"{f['departure'][11:16]} → {f['arrival'][11:16]}  |  "
                f"{f['duration_min']} min  |  "
                f"₹{f['price_inr']:,}  |  "
                f"{f['seats_left']} seats left"
            )
        return [types.TextContent(type="text", text="\n".join(lines))]

    elif name == "indigo_book_flight":
        result = _book_indigo(
            arguments["flight_id"],
            arguments["passenger_name"],
            arguments["passenger_email"],
        )
        if "error" in result:
            return [types.TextContent(type="text", text=f"Booking failed: {result['error']}")]
        return [types.TextContent(
            type="text",
            text=(
                f"✅ IndiGo booking CONFIRMED!\n"
                f"Booking ref:  {result['booking_ref']}\n"
                f"Flight:       {result['flight_id']}\n"
                f"Route:        {result['origin']} → {result['destination']}\n"
                f"Departure:    {result['departure']}\n"
                f"Passenger:    {result['passenger_name']}\n"
                f"E-ticket to:  {result['passenger_email']}\n"
                f"Amount paid:  ₹{result['price_inr']:,}\n"
                f"Status:       {result['status']}"
            )
        )]

    elif name == "indigo_list_bookings":
        if not bookings:
            return [types.TextContent(type="text", text="No IndiGo bookings yet.")]
        lines = [f"{b['booking_ref']} | {b['flight_id']} | "
                 f"{b['origin']}→{b['destination']} | {b['passenger_name']}"
                 for b in bookings]
        return [types.TextContent(type="text", text="\n".join(lines))]

    else:
        raise ValueError(f"Unknown tool: {name}")


# ─────────────────────────────────────────────────────────────────────
# RESOURCES
# ─────────────────────────────────────────────────────────────────────

@app.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="indigo://bookings",
            name="My IndiGo bookings",
            description="All confirmed IndiGo bookings as JSON.",
            mimeType="application/json",
        )
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "indigo://bookings":
        return json.dumps(bookings, indent=2) if bookings else "[]"
    raise ValueError(f"Unknown resource: {uri}")


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())

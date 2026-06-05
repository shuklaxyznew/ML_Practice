"""
Skyscanner MCP Server
======================
Exposes Skyscanner multi-airline flight search as MCP tools.

Skyscanner has a public API via RapidAPI:
    https://rapidapi.com/skyscanner/api/skyscanner-flight-search

Install:
    pip install mcp httpx

Run (stdio — used by Claude Desktop):
    python skyscanner_mcp_server.py

Environment variables:
    RAPIDAPI_KEY   — your RapidAPI key for Skyscanner API
"""

import asyncio
import os
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ── optional: real HTTP client ──────────────────────────────────────────
# import httpx
# RAPIDAPI_KEY  = os.getenv("RAPIDAPI_KEY", "")
# SKY_BASE      = "https://skyscanner-flight-search.p.rapidapi.com/apiservices"
# SKY_HEADERS   = {
#     "X-RapidAPI-Key":  RAPIDAPI_KEY,
#     "X-RapidAPI-Host": "skyscanner-flight-search.p.rapidapi.com",
# }

app = Server("skyscanner-server")

# ─────────────────────────────────────────────────────────────────────
# MOCK DATA  (replace with real RapidAPI / Skyscanner SDK calls)
# ─────────────────────────────────────────────────────────────────────

MOCK_RESULTS = [
    {
        "itinerary_id": "SKY-BLR-DEL-001",
        "carriers":     ["IndiGo"],
        "origin":       "BLR",
        "destination":  "DEL",
        "departure":    "2025-07-10T06:00:00",
        "arrival":      "2025-07-10T08:45:00",
        "stops":        0,
        "duration_min": 165,
        "price_inr":    4850,
        "price_usd":    58.2,
        "deep_link":    "https://www.skyscanner.co.in/...",
    },
    {
        "itinerary_id": "SKY-BLR-DEL-002",
        "carriers":     ["Air India"],
        "origin":       "BLR",
        "destination":  "DEL",
        "departure":    "2025-07-10T10:00:00",
        "arrival":      "2025-07-10T12:55:00",
        "stops":        0,
        "duration_min": 175,
        "price_inr":    5600,
        "price_usd":    67.2,
        "deep_link":    "https://www.skyscanner.co.in/...",
    },
    {
        "itinerary_id": "SKY-BLR-DEL-003",
        "carriers":     ["SpiceJet", "GoAir"],
        "origin":       "BLR",
        "destination":  "DEL",
        "departure":    "2025-07-10T07:30:00",
        "arrival":      "2025-07-10T13:10:00",
        "stops":        1,
        "duration_min": 340,
        "price_inr":    3200,
        "price_usd":    38.4,
        "deep_link":    "https://www.skyscanner.co.in/...",
    },
    {
        "itinerary_id": "SKY-BLR-BOM-001",
        "carriers":     ["Vistara"],
        "origin":       "BLR",
        "destination":  "BOM",
        "departure":    "2025-07-10T09:15:00",
        "arrival":      "2025-07-10T10:55:00",
        "stops":        0,
        "duration_min": 100,
        "price_inr":    3800,
        "price_usd":    45.6,
        "deep_link":    "https://www.skyscanner.co.in/...",
    },
]

# Store price quotes (simulate "lock price" feature)
price_quotes: dict[str, dict] = {}


def _sky_search(origin: str, destination: str, date: str,
                max_stops: int = 2, sort_by: str = "price") -> list[dict]:
    """
    Mock multi-airline search. Replace with:
        async with httpx.AsyncClient() as client:
            # Step 1: create session
            resp = await client.get(
                f"{SKY_BASE}/browsequotes/v1.0/IN/INR/en-IN/{origin}/{destination}/{date}",
                headers=SKY_HEADERS,
            )
            return resp.json()["Quotes"]
    """
    origin = origin.upper()
    destination = destination.upper()
    results = [
        r for r in MOCK_RESULTS
        if r["origin"] == origin
        and r["destination"] == destination
        and r["stops"] <= max_stops
    ]
    if sort_by == "price":
        results.sort(key=lambda r: r["price_inr"])
    elif sort_by == "duration":
        results.sort(key=lambda r: r["duration_min"])
    return results


def _sky_get_prices(itinerary_id: str) -> dict:
    """
    Mock price detail fetch. In production:
        Hit Skyscanner live pricing session endpoint.
    """
    result = next((r for r in MOCK_RESULTS if r["itinerary_id"] == itinerary_id), None)
    if not result:
        return {"error": f"Itinerary {itinerary_id} not found."}
    quote = {
        **result,
        "breakdown": {
            "base_fare_inr":  result["price_inr"] - 520,
            "taxes_inr":      360,
            "convenience_inr": 160,
            "total_inr":      result["price_inr"],
        },
        "fare_rules": "Non-refundable. Changes allowed with ₹3,500 fee.",
        "baggage":    "7 kg cabin. 15 kg check-in included.",
        "quoted_at":  "2025-06-05T10:00:00",
    }
    price_quotes[itinerary_id] = quote
    return quote


def _sky_compare(origin: str, destination: str, date: str) -> dict:
    """
    Compare cheapest option vs fastest option.
    """
    results = _sky_search(origin, destination, date)
    if not results:
        return {"error": "No flights found."}
    cheapest = min(results, key=lambda r: r["price_inr"])
    fastest  = min(results, key=lambda r: r["duration_min"])
    return {
        "cheapest": cheapest,
        "fastest":  fastest,
        "all_count": len(results),
    }


# ─────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="sky_search_flights",
            description=(
                "Search flights across ALL airlines on Skyscanner between two airports. "
                "Returns results sorted by price (default) or duration. "
                "Covers IndiGo, Air India, Vistara, SpiceJet, GoAir, Emirates, and more. "
                "Use IATA codes (BLR, DEL, BOM, MAA, DXB, LHR, etc.)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "IATA code of departure airport. E.g. 'BLR'.",
                    },
                    "destination": {
                        "type": "string",
                        "description": "IATA code of arrival airport. E.g. 'DEL'.",
                    },
                    "date": {
                        "type": "string",
                        "description": "Travel date in YYYY-MM-DD format.",
                    },
                    "max_stops": {
                        "type": "integer",
                        "description": "Maximum number of stops. 0 = non-stop only. Default: 2.",
                        "default": 2,
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["price", "duration"],
                        "description": "Sort results by 'price' (cheapest first) or 'duration' (fastest first). Default: price.",
                        "default": "price",
                    },
                },
                "required": ["origin", "destination", "date"],
            },
        ),
        types.Tool(
            name="sky_get_prices",
            description=(
                "Get detailed fare breakdown, baggage policy, and fare rules "
                "for a specific Skyscanner itinerary_id from sky_search_flights results."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "itinerary_id": {
                        "type": "string",
                        "description": "Itinerary ID from sky_search_flights results.",
                    }
                },
                "required": ["itinerary_id"],
            },
        ),
        types.Tool(
            name="sky_compare_flights",
            description=(
                "Compare the cheapest vs fastest flight between two airports. "
                "Great for helping users decide between saving money and saving time."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "origin":      {"type": "string", "description": "IATA departure code."},
                    "destination": {"type": "string", "description": "IATA arrival code."},
                    "date":        {"type": "string", "description": "Date in YYYY-MM-DD."},
                },
                "required": ["origin", "destination", "date"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "sky_search_flights":
        results = _sky_search(
            arguments["origin"],
            arguments["destination"],
            arguments["date"],
            max_stops=arguments.get("max_stops", 2),
            sort_by=arguments.get("sort_by", "price"),
        )
        if not results:
            return [types.TextContent(
                type="text",
                text=f"No flights found on Skyscanner from "
                     f"{arguments['origin']} to {arguments['destination']} "
                     f"on {arguments['date']}."
            )]
        lines = ["Skyscanner results (multi-airline):\n"]
        for r in results:
            stops_str = "non-stop" if r["stops"] == 0 else f"{r['stops']} stop"
            lines.append(
                f"🌐 [{r['itinerary_id']}]  "
                f"{', '.join(r['carriers'])}  |  "
                f"{r['departure'][11:16]} → {r['arrival'][11:16]}  |  "
                f"{r['duration_min']} min  |  "
                f"{stops_str}  |  "
                f"₹{r['price_inr']:,}  (${r['price_usd']})"
            )
        lines.append("\nUse sky_get_prices with an itinerary_id for full fare details.")
        return [types.TextContent(type="text", text="\n".join(lines))]

    elif name == "sky_get_prices":
        quote = _sky_get_prices(arguments["itinerary_id"])
        if "error" in quote:
            return [types.TextContent(type="text", text=f"Error: {quote['error']}")]
        b = quote["breakdown"]
        return [types.TextContent(
            type="text",
            text=(
                f"💰 Fare breakdown for {quote['itinerary_id']}\n"
                f"Carrier(s):   {', '.join(quote['carriers'])}\n"
                f"Route:        {quote['origin']} → {quote['destination']}\n"
                f"Departure:    {quote['departure']}\n\n"
                f"Base fare:    ₹{b['base_fare_inr']:,}\n"
                f"Taxes:        ₹{b['taxes_inr']:,}\n"
                f"Convenience:  ₹{b['convenience_inr']:,}\n"
                f"─────────────────────\n"
                f"Total:        ₹{b['total_inr']:,}\n\n"
                f"Baggage:      {quote['baggage']}\n"
                f"Fare rules:   {quote['fare_rules']}\n"
                f"Book at:      {quote['deep_link']}"
            )
        )]

    elif name == "sky_compare_flights":
        comparison = _sky_compare(
            arguments["origin"],
            arguments["destination"],
            arguments["date"],
        )
        if "error" in comparison:
            return [types.TextContent(type="text", text=comparison["error"])]
        c = comparison["cheapest"]
        f = comparison["fastest"]
        return [types.TextContent(
            type="text",
            text=(
                f"📊 Cheapest vs Fastest — {arguments['origin']} → {arguments['destination']}\n\n"
                f"💸 Cheapest: {', '.join(c['carriers'])} | ₹{c['price_inr']:,} | "
                f"{c['duration_min']} min | {'non-stop' if c['stops']==0 else str(c['stops'])+' stop'}\n"
                f"⚡ Fastest:  {', '.join(f['carriers'])} | ₹{f['price_inr']:,} | "
                f"{f['duration_min']} min | {'non-stop' if f['stops']==0 else str(f['stops'])+' stop'}\n\n"
                f"Total options found: {comparison['all_count']}"
            )
        )]

    else:
        raise ValueError(f"Unknown tool: {name}")


# ─────────────────────────────────────────────────────────────────────
# RESOURCES
# ─────────────────────────────────────────────────────────────────────

@app.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="skyscanner://quotes",
            name="Skyscanner price quotes",
            description="All fetched price quotes from this session as JSON.",
            mimeType="application/json",
        )
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "skyscanner://quotes":
        return json.dumps(price_quotes, indent=2) if price_quotes else "{}"
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

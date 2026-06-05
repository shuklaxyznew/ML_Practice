# Connecting Both Flight Servers to Claude Desktop
# ==================================================

# ── 1. claude_desktop_config.json ──────────────────────────────────────────
# Save this at:
#   macOS:   ~/Library/Application Support/Claude/claude_desktop_config.json
#   Windows: %APPDATA%\Claude\claude_desktop_config.json
#
# Replace /absolute/path/to/ with your actual directory path.

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

# After saving, restart Claude Desktop.
# You will see both servers listed in the MCP panel (hammer icon).


# ── 2. Test without Claude Desktop (manual Python test script) ──────────────
# Save as: test_flight_servers.py
# Run with: python test_flight_servers.py

"""
test_flight_servers.py — Quick sanity check for both MCP servers.
Calls the tool handler functions directly, without a real MCP client.
"""

import asyncio
import sys
sys.path.insert(0, ".")

async def test_indigo():
    print("=" * 50)
    print("INDIGO SERVER TESTS")
    print("=" * 50)

    # Import the server modules
    import indigo_mcp_server as indigo

    # Test 1: Search flights
    print("\n[1] Search BLR → DEL")
    results = indigo._search_indigo("BLR", "DEL", "2025-07-10")
    for f in results:
        print(f"  {f['flight_id']} | ₹{f['price_inr']:,} | {f['duration_min']}min")

    # Test 2: Book a flight
    print("\n[2] Book flight 6E-2401")
    booking = indigo._book_indigo("6E-2401", "Rohan Sharma", "rohan@example.com")
    print(f"  Booking ref: {booking.get('booking_ref', booking.get('error'))}")

    # Test 3: Book non-existent flight
    print("\n[3] Book invalid flight")
    bad = indigo._book_indigo("6E-9999", "Test User", "test@test.com")
    print(f"  Result: {bad}")


async def test_skyscanner():
    print("\n" + "=" * 50)
    print("SKYSCANNER SERVER TESTS")
    print("=" * 50)

    import skyscanner_mcp_server as sky

    # Test 1: Search all airlines
    print("\n[1] Search BLR → DEL (sorted by price)")
    results = sky._sky_search("BLR", "DEL", "2025-07-10", sort_by="price")
    for r in results:
        print(f"  {r['itinerary_id']} | {r['carriers']} | ₹{r['price_inr']:,} | {r['stops']} stops")

    # Test 2: Get price details
    print("\n[2] Get prices for SKY-BLR-DEL-001")
    quote = sky._sky_get_prices("SKY-BLR-DEL-001")
    print(f"  Total: ₹{quote['breakdown']['total_inr']:,}")
    print(f"  Baggage: {quote['baggage']}")

    # Test 3: Compare cheapest vs fastest
    print("\n[3] Compare BLR → DEL")
    comp = sky._sky_compare("BLR", "DEL", "2025-07-10")
    c, f = comp["cheapest"], comp["fastest"]
    print(f"  Cheapest: ₹{c['price_inr']:,} | {c['carriers']}")
    print(f"  Fastest:  {f['duration_min']}min | {f['carriers']}")

    # Test 4: Non-stop only
    print("\n[4] Non-stop BLR → DEL only")
    nonstop = sky._sky_search("BLR", "DEL", "2025-07-10", max_stops=0)
    print(f"  Found {len(nonstop)} non-stop options")


async def main():
    await test_indigo()
    await test_skyscanner()
    print("\n✅ All tests passed.")

if __name__ == "__main__":
    asyncio.run(main())

"""
test_flight_servers.py
======================
Quick sanity check for both Indigo and Skyscanner MCP servers.
Calls the handler functions directly — no real MCP client needed.

Usage:
    python test_flight_servers.py

Make sure indigo_mcp_server.py and skyscanner_mcp_server.py
are in the same directory before running.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import indigo_mcp_server as indigo
import skyscanner_mcp_server as sky


def separator(title):
    print("\n" + "=" * 52)
    print(f"  {title}")
    print("=" * 52)


async def test_indigo():
    separator("INDIGO SERVER TESTS")

    print("\n[1] Search flights: BLR → DEL")
    results = indigo._search_indigo("BLR", "DEL", "2025-07-10")
    if results:
        for f in results:
            print(f"    {f['flight_id']}  |  "
                  f"{f['departure'][11:16]} → {f['arrival'][11:16]}  |  "
                  f"₹{f['price_inr']:,}  |  {f['seats_left']} seats")
    else:
        print("    No flights found.")

    print("\n[2] Search flights: BLR → BOM")
    results2 = indigo._search_indigo("BLR", "BOM", "2025-07-10")
    for f in results2:
        print(f"    {f['flight_id']}  |  ₹{f['price_inr']:,}  |  {f['duration_min']} min")

    print("\n[3] Book flight 6E-2401")
    booking = indigo._book_indigo("6E-2401", "Rohan Sharma", "rohan@example.com")
    if "error" in booking:
        print(f"    ERROR: {booking['error']}")
    else:
        print(f"    Booking ref : {booking['booking_ref']}")
        print(f"    Flight      : {booking['flight_id']}")
        print(f"    Route       : {booking['origin']} → {booking['destination']}")
        print(f"    Passenger   : {booking['passenger_name']}")
        print(f"    Status      : {booking['status']}")

    print("\n[4] Book a second flight 6E-5512")
    booking2 = indigo._book_indigo("6E-5512", "Priya Patel", "priya@example.com")
    print(f"    Booking ref : {booking2.get('booking_ref', booking2.get('error'))}")

    print("\n[5] Book invalid flight ID")
    bad = indigo._book_indigo("6E-9999", "Test User", "test@test.com")
    print(f"    Result: {bad}")

    print("\n[6] All bookings in session")
    for b in indigo.bookings:
        print(f"    {b['booking_ref']}  |  {b['flight_id']}  |  {b['passenger_name']}")


async def test_skyscanner():
    separator("SKYSCANNER SERVER TESTS")

    print("\n[1] Search BLR → DEL (sort by price)")
    results = sky._sky_search("BLR", "DEL", "2025-07-10", sort_by="price")
    for r in results:
        stops = "non-stop" if r["stops"] == 0 else f"{r['stops']} stop"
        print(f"    {r['itinerary_id']}  |  "
              f"{', '.join(r['carriers'])}  |  "
              f"₹{r['price_inr']:,}  |  {r['duration_min']} min  |  {stops}")

    print("\n[2] Search BLR → DEL (sort by duration)")
    results2 = sky._sky_search("BLR", "DEL", "2025-07-10", sort_by="duration")
    for r in results2:
        print(f"    {r['itinerary_id']}  |  {r['duration_min']} min  |  ₹{r['price_inr']:,}")

    print("\n[3] Non-stop only (max_stops=0)")
    nonstop = sky._sky_search("BLR", "DEL", "2025-07-10", max_stops=0)
    print(f"    Found {len(nonstop)} non-stop option(s)")
    for r in nonstop:
        print(f"    {r['itinerary_id']}  |  {', '.join(r['carriers'])}")

    print("\n[4] Get fare breakdown: SKY-BLR-DEL-001")
    quote = sky._sky_get_prices("SKY-BLR-DEL-001")
    if "error" in quote:
        print(f"    ERROR: {quote['error']}")
    else:
        b = quote["breakdown"]
        print(f"    Base fare    : ₹{b['base_fare_inr']:,}")
        print(f"    Taxes        : ₹{b['taxes_inr']:,}")
        print(f"    Convenience  : ₹{b['convenience_inr']:,}")
        print(f"    Total        : ₹{b['total_inr']:,}")
        print(f"    Baggage      : {quote['baggage']}")
        print(f"    Fare rules   : {quote['fare_rules']}")

    print("\n[5] Compare cheapest vs fastest: BLR → DEL")
    comp = sky._sky_compare("BLR", "DEL", "2025-07-10")
    if "error" in comp:
        print(f"    ERROR: {comp['error']}")
    else:
        c, f = comp["cheapest"], comp["fastest"]
        print(f"    Cheapest : {', '.join(c['carriers'])}  |  ₹{c['price_inr']:,}  |  {c['duration_min']} min")
        print(f"    Fastest  : {', '.join(f['carriers'])}  |  ₹{f['price_inr']:,}  |  {f['duration_min']} min")
        print(f"    Total options found : {comp['all_count']}")

    print("\n[6] Search BLR → BOM")
    bom = sky._sky_search("BLR", "BOM", "2025-07-10")
    for r in bom:
        print(f"    {r['itinerary_id']}  |  {', '.join(r['carriers'])}  |  ₹{r['price_inr']:,}")

    print("\n[7] Invalid route (no results expected)")
    none_ = sky._sky_search("DEL", "GOI", "2025-07-10")
    print(f"    Results found: {len(none_)}  (expected 0)")


async def main():
    await test_indigo()
    await test_skyscanner()

    print("\n" + "=" * 52)
    print("  All tests completed.")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

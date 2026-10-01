"""Flight Agent - searches outbound and return flights and scores convenience."""

from datetime import timedelta

from agents.base import Agent
from core import inventory


def convenience_score(f: dict) -> float:
    """0..1: shorter, non-stop, sociable departure hours score higher."""
    dur = max(0.0, 1 - (f["duration_min"] - 60) / 600)
    stops = 1.0 if f["stops"] == 0 else 0.55
    hour = int(f["depart"][:2])
    timing = 1.0 if 7 <= hour <= 20 else 0.6
    return round(0.45 * dur + 0.35 * stops + 0.20 * timing, 3)


class FlightAgent(Agent):
    name = "Flight Agent"
    role = "Find outbound and return flights; score them on price and convenience."
    tools = ["search_flights (inventory API)"]

    def run(self, spec: dict, today=None) -> dict:
        o, d, n = spec["origin"], spec["destination"], spec["travellers"]
        start, end = spec["start_date"], spec["end_date"]
        self.say("Flight API", "tool_call", f"search_flights({o}->{d}, {start}, pax={n})")
        out = self.timed(inventory.search_flights, o, d, start, n, today)
        self.say("Flight API", "tool_call", f"search_flights({d}->{o}, {end}, pax={n})")
        ret = self.timed(inventory.search_flights, d, o, end, n, today)
        for f in out + ret:
            f["convenience"] = convenience_score(f)
        result = {"outbound": out, "return": ret}
        cheapest = out[0]["price_pp"] + ret[0]["price_pp"]
        self.say("Orchestrator", "result",
                 f"{len(out)} outbound + {len(ret)} return options. Cheapest round trip "
                 f"INR {cheapest:,} per person ({out[0]['airline']} / {ret[0]['airline']}).")
        self.board.put("flights", result)
        return result

    def flexible_dates(self, spec: dict, window: int = 2, today=None) -> list[dict]:
        """Check +/- `window` days for cheaper round trips (used when over budget)."""
        o, d = spec["origin"], spec["destination"]
        options = []
        for shift in range(-window, window + 1):
            if shift == 0:
                continue
            s = spec["start_date"] + timedelta(days=shift)
            e = spec["end_date"] + timedelta(days=shift)
            if today and s <= today:
                continue
            a = inventory.search_flights(o, d, s, 1, today)[0]["price_pp"]
            b = inventory.search_flights(d, o, e, 1, today)[0]["price_pp"]
            options.append({"shift_days": shift, "start": s.isoformat(), "end": e.isoformat(),
                            "round_trip_pp": a + b})
        options.sort(key=lambda x: x["round_trip_pp"])
        self.say("Budget Agent", "result",
                 f"Flexible-date scan (+/-{window} days): cheapest is shift {options[0]['shift_days']:+d} "
                 f"days at INR {options[0]['round_trip_pp']:,} pp." if options else "No alternative dates.")
        return options

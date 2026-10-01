"""Hotel Agent - searches hotels, filters by travel style, scores quality."""

from agents.base import Agent
from core import inventory
from core.data import STYLE_STARS


def quality_score(h: dict, style: str) -> float:
    """0..1 from guest rating, review volume and fit with the requested style."""
    lo, hi = STYLE_STARS[style]
    rating = (h["rating"] - 3.0) / 2.0
    fit = 1.0 if lo <= h["stars"] <= hi else (0.6 if abs(h["stars"] - (lo + hi) / 2) <= 1.5 else 0.3)
    volume = min(h["reviews"] / 2000, 1.0)
    bonus = 0.05 * ("Breakfast included" in h["amenities"]) + 0.05 * h["free_cancellation"]
    return round(min(1.0, 0.5 * rating + 0.3 * fit + 0.1 * volume + bonus), 3)


class HotelAgent(Agent):
    name = "Hotel Agent"
    role = "Find hotels that fit the travel style; widen the search if the Budget Agent asks."
    tools = ["search_hotels (inventory API)"]

    def run(self, spec: dict, widen: bool = False) -> list[dict]:
        city, style = spec["destination"], spec["style"]
        self.say("Hotel API", "tool_call",
                 f"search_hotels({city}, {spec['start_date']} to {spec['end_date']}, pax={spec['travellers']})")
        hotels = self.timed(inventory.search_hotels, city, spec["start_date"], spec["end_date"], spec["travellers"])
        for h in hotels:
            h["quality"] = quality_score(h, style)
        lo, hi = STYLE_STARS[style]
        if widen:
            shortlist = hotels
            self.say("Budget Agent", "result", f"Widened search to all {len(hotels)} hotels (2-5 star).")
        else:
            shortlist = [h for h in hotels if lo - 1 <= h["stars"] <= hi]
            self.say("Orchestrator", "result",
                     f"{len(shortlist)} hotels fit '{style}' ({lo}-{hi} star, one band lower allowed). "
                     f"Top rated: {max(shortlist, key=lambda h: h['rating'])['name']}.")
        self.board.put("hotels", shortlist)
        return shortlist

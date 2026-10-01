"""
Orchestrator (Planner) agent - coordinates the multi-agent workflow.

  1. UNDERSTAND : parse free text into a trip spec (LLM, or regex fallback)
  2. GUARD      : screen text (PII / injection) and validate the spec
  3. DISPATCH   : Flight, Hotel and Weather agents run IN PARALLEL
  4. OPTIMISE   : Budget Agent picks the best combination
  5. NEGOTIATE  : if over budget -> Hotel Agent widens, Flight Agent scans dates,
                  Budget Agent re-optimises and proposes trade-offs
  6. PLAN       : Itinerary Agent builds a RAG-grounded, weather-aware plan
  7. CRITIQUE   : Critic checks constraints before anything is shown
  8. RESPOND    : summary for the traveller (LLM or template)
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

from agents.base import Agent, Blackboard
from agents.budget_agent import BudgetAgent
from agents.flight_agent import FlightAgent
from agents.guardrails import DISCLAIMER, screen_text, validate_spec
from agents.hotel_agent import HotelAgent
from agents.itinerary_agent import ItineraryAgent
from agents.weather_agent import WeatherAgent
from core import prompts
from core.data import CITIES, INTERESTS, destinations, is_international, origins
from core.llm import LLMError

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
                                       "sep", "oct", "nov", "dec"], start=1)}
INTEREST_WORDS = {
    "beach": ["beach", "sea", "coast", "island"], "heritage": ["heritage", "history", "fort", "palace", "monument"],
    "culture": ["culture", "local life", "tradition"], "food": ["food", "seafood", "street food", "cuisine", "eat"],
    "nature": ["nature", "lake", "garden", "waterfall", "backwater", "scenic"],
    "adventure": ["adventure", "trek", "safari", "water sports", "thrill"],
    "shopping": ["shopping", "market", "bazaar", "mall"], "nightlife": ["nightlife", "party", "club", "bar"],
    "spiritual": ["temple", "spiritual", "church", "mosque", "pilgrim", "ghat"],
    "art": ["art", "museum", "gallery"], "family": ["kids", "family", "children", "parents"],
}
CITY_ALIASES = {"bombay": "Mumbai", "bangalore": "Bengaluru", "new delhi": "Delhi", "cochin": "Kochi",
                "kerala": "Kochi", "banaras": "Varanasi", "kashi": "Varanasi", "madras": "Chennai"}


def regex_parse(text: str, today: date) -> dict:
    """Offline fallback for natural-language trip requests."""
    t = text.lower()
    spec = {"origin": None, "destination": None, "start_date": None, "nights": None, "travellers": None,
            "budget_inr": None, "style": None, "interests": [], "assumptions": []}
    names = {c.lower(): c for c in CITIES} | CITY_ALIASES
    m = re.search(r"from\s+([a-z ]+?)(?:\s+to|\s*,|\s+for|\s+on|\s+in|$)", t)
    if m and m.group(1).strip() in names:
        spec["origin"] = names[m.group(1).strip()]
    for alias, city in sorted(names.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{alias}\b", t) and city != spec["origin"] and CITIES[city]["dest"]:
            if not (spec["origin"] is None and CITIES[city]["origin"] and re.search(rf"from\s+{alias}", t)):
                spec["destination"] = city
                break
    if spec["origin"] is None:
        for alias, city in names.items():
            if re.search(rf"from\s+{alias}\b", t):
                spec["origin"] = city
    if m := re.search(r"(\d+)\s*(?:nights?|n\b)", t):
        spec["nights"] = int(m.group(1))
    elif m := re.search(r"(\d+)\s*days?", t):
        spec["nights"] = max(int(m.group(1)) - 1, 1)
        spec["assumptions"].append(f"{m.group(1)} days read as {spec['nights']} nights")
    elif "weekend" in t:
        spec["nights"] = 2
    if m := re.search(r"(\d+)\s*(?:people|persons|pax|adults|travellers|travelers|friends|of us)", t):
        spec["travellers"] = int(m.group(1))
    elif m := re.search(r"family of (\d+)", t):
        spec["travellers"] = int(m.group(1))
    elif re.search(r"\b(wife|husband|partner|girlfriend|boyfriend|couple|honeymoon)\b", t):
        spec["travellers"] = 2
    elif re.search(r"\b(solo|alone|myself)\b", t):
        spec["travellers"] = 1
    if m := re.search(r"(\d+(?:\.\d+)?)\s*(lakh|lac|l\b|k\b|thousand)", t):
        mult = 100000 if m.group(2) in ("lakh", "lac", "l") else 1000
        spec["budget_inr"] = int(float(m.group(1)) * mult)
    elif m := re.search(r"(?:₹|rs\.?|inr)\s*([\d,]{4,})", t):
        spec["budget_inr"] = int(m.group(1).replace(",", ""))
    if m := re.search(r"(\d{4}-\d{2}-\d{2})", t):
        spec["start_date"] = m.group(1)
    elif m := re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*", t):
        d, mo = int(m.group(1)), MONTHS[m.group(2)]
        yr = today.year + (1 if (mo, d) < (today.month, today.day) else 0)
        spec["start_date"] = date(yr, mo, d).isoformat()
    elif m := re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", t):
        mo = MONTHS[m.group(1)]
        yr = today.year + (1 if mo < today.month else 0)
        spec["start_date"] = date(yr, mo, 10).isoformat()
        spec["assumptions"].append("Only a month was given; assumed the 10th")
    if re.search(r"luxury|5[- ]star|premium|lavish", t):
        spec["style"] = "Luxury"
    elif re.search(r"\bbudget (?:trip|travel|hotel|stay|holiday)|on a (?:tight )?budget|cheap|backpack|affordable|low[- ]cost", t):
        spec["style"] = "Budget"
    for interest, words in INTEREST_WORDS.items():
        if any(w in t for w in words):
            spec["interests"].append(interest)
    return spec


class Orchestrator(Agent):
    name = "Orchestrator"
    role = "Understand the request, delegate to specialist agents, resolve conflicts, check the final plan."

    def __init__(self, llm=None, allow_network: bool = True, today: date | None = None):
        self.board = Blackboard()
        super().__init__(self.board, llm)
        self.today = today or date.today()
        self.flight = FlightAgent(self.board, llm)
        self.hotel = HotelAgent(self.board, llm)
        self.weather = WeatherAgent(self.board, llm, allow_network=allow_network)
        self.budget = BudgetAgent(self.board, llm)
        self.itinerary = ItineraryAgent(self.board, llm)

    # ---------------- Step 1: understand ----------------
    def parse_request(self, text: str) -> tuple[dict, list[str]]:
        safe, warnings = screen_text(text)
        for w in warnings:
            self.say("User", "warning", w)
        spec = None
        if self.llm is not None and self.llm.enabled:
            try:
                user = prompts.PLANNER_FEWSHOT + "\n\n" + prompts.PLANNER_USER.format(
                    origins=origins(), destinations=destinations(), interests=INTERESTS,
                    today=self.today.isoformat(), request=safe)
                self.say("LLM", "tool_call", "parse trip request -> JSON spec")
                spec = self.llm.complete_json(prompts.PLANNER_SYSTEM, user)
                spec["interests"] = [i for i in spec.get("interests", []) if i in INTERESTS]
            except (LLMError, ValueError) as e:
                self.say("Orchestrator", "warning", f"LLM parse failed ({str(e)[:60]}); using rule-based parser.")
        if spec is None:
            spec = regex_parse(safe, self.today)
        self.say("User", "info", f"Understood request as: { {k: v for k, v in spec.items() if v and k != 'assumptions'} }")
        return spec, warnings

    # ---------------- Steps 2-8 ----------------
    def plan(self, spec: dict) -> dict:
        t0 = time.time()
        spec = dict(spec)
        spec.setdefault("style", "Moderate")
        spec["style"] = spec["style"] or "Moderate"
        spec["interests"] = spec.get("interests") or []
        errors = validate_spec(spec, self.today)
        if errors:
            self.say("User", "warning", "Cannot plan yet: " + "; ".join(errors))
            return {"ok": False, "errors": errors, "trace": self.board.trace()}
        if isinstance(spec["start_date"], str):
            spec["start_date"] = datetime.strptime(spec["start_date"], "%Y-%m-%d").date()
        spec["nights"], spec["travellers"], spec["budget_inr"] = int(spec["nights"]), int(spec["travellers"]), int(spec["budget_inr"])
        spec["end_date"] = spec["start_date"] + timedelta(days=spec["nights"])
        self.board.put("spec", spec)
        self.say("Flight Agent", "task", f"Find flights {spec['origin']} <-> {spec['destination']} for {spec['travellers']}.")
        self.say("Hotel Agent", "task", f"Find {spec['style']} hotels in {spec['destination']} for {spec['nights']} nights.")
        self.say("Weather Agent", "task", f"Weather outlook {spec['start_date']} to {spec['end_date']}.")

        with ThreadPoolExecutor(max_workers=3) as pool:
            f_flights = pool.submit(self.flight.run, spec, self.today)
            f_hotels = pool.submit(self.hotel.run, spec)
            f_weather = pool.submit(self.weather.run, spec, self.today)
            flights, hotels, weather = f_flights.result(), f_hotels.result(), f_weather.result()

        self.say("Budget Agent", "task", "Optimise flight + hotel combination within budget.")
        budget = self.budget.optimise(spec, flights, hotels)
        suggestions, negotiated = [], False
        if budget["status"] == "over_budget":
            negotiated = True
            hotels = self.hotel.run(spec, widen=True)
            budget = self.budget.optimise(spec, flights, hotels)
            if budget["status"] == "over_budget":
                dates = self.flight.flexible_dates(spec, today=self.today)
                suggestions = self.budget.suggestions(spec, budget, dates)
        plan = budget["plans"][0] if budget["status"] == "ok" else budget["cheapest"]

        self.say("Itinerary Agent", "task", "Build a day-wise plan using the knowledge base and weather.")
        itinerary = self.itinerary.run(spec, plan, weather)

        checks = self.critic(spec, plan, weather, itinerary)
        summary = self.summarise(spec, plan, weather, itinerary, budget)
        latency = round(time.time() - t0, 2)
        self.say("User", "result", f"Plan ready in {latency}s. {sum(c['passed'] for c in checks)}/{len(checks)} quality checks passed.")
        return {"ok": True, "spec": spec, "flights": flights, "hotels": hotels, "weather": weather,
                "budget": budget, "plan": plan, "negotiated": negotiated, "suggestions": suggestions,
                "itinerary": itinerary, "checks": checks, "summary": summary, "disclaimer": DISCLAIMER,
                "latency_s": latency, "llm_calls": getattr(self.llm, "calls", 0), "trace": self.board.trace(),
                "international": is_international(spec["origin"], spec["destination"])}

    def run(self, text: str) -> dict:
        spec, _ = self.parse_request(text)
        return self.plan(spec)

    # ---------------- Step 7: critic ----------------
    def critic(self, spec, plan, weather, itin) -> list[dict]:
        acts = [(d, x) for d in itin["days"] for x in d["items"] if x.get("kind") == "activity"]
        rainy = {d["date"] for d in weather["days"] if d["rainy"]}
        outdoor_rain = [x["activity"] for d, x in acts if d["date"] in rainy and x["setting"] == "outdoor"]
        names = [x["activity"] for _, x in acts]
        checks = [
            ("Total cost within budget", plan["cost"]["total"] <= spec["budget_inr"],
             f"INR {plan['cost']['total']:,} vs budget INR {spec['budget_inr']:,}"),
            ("Return flight after outbound", plan["return"]["date"] > plan["outbound"]["date"],
             f"{plan['outbound']['date']} -> {plan['return']['date']}"),
            ("Hotel nights match trip", plan["hotel"]["nights"] == spec["nights"], f"{plan['hotel']['nights']} nights"),
            ("Enough rooms for group", plan["hotel"]["rooms"] * 2 >= spec["travellers"],
             f"{plan['hotel']['rooms']} room(s) for {spec['travellers']}"),
            ("Itinerary grounded in knowledge base", itin["grounding"]["score"] == 1.0,
             f"{itin['grounding']['grounded']}/{itin['grounding']['total']} grounded"),
            ("No outdoor activity on rainy days", not outdoor_rain,
             "none" if not outdoor_rain else f"{len(outdoor_rain)}: {outdoor_rain[:3]}"),
            ("No repeated attractions", len(names) == len(set(names)), f"{len(names)} activities"),
            ("Entry fees within activity allowance", itin["activity_cost_pp"] <= itin["activity_allowance_pp"],
             f"INR {itin['activity_cost_pp']:,} of {itin['activity_allowance_pp']:,} pp"),
        ]
        out = [{"check": c, "passed": bool(p), "detail": d} for c, p, d in checks]
        failed = [c["check"] for c in out if not c["passed"]]
        self.board.post("Critic", "Orchestrator", "feedback" if failed else "result",
                        "All checks passed." if not failed else "Issues: " + ", ".join(failed))
        return out

    # ---------------- Step 8: respond ----------------
    def summarise(self, spec, plan, weather, itin, budget) -> str:
        facts = {"destination": spec["destination"], "dates": f"{spec['start_date']} to {spec['end_date']}",
                 "travellers": spec["travellers"], "outbound": f"{plan['outbound']['airline']} {plan['outbound']['depart']}",
                 "return": f"{plan['return']['airline']} {plan['return']['depart']}",
                 "hotel": f"{plan['hotel']['name']} ({plan['hotel']['stars']}*, {plan['hotel']['area']}, rated {plan['hotel']['rating']})",
                 "weather": f"avg high {weather['avg_high']} C, {weather['rainy_days']} rainy day(s)",
                 "total_inr": plan["cost"]["total"], "budget_inr": spec["budget_inr"],
                 "top_tip": (itin.get("local_tips") or [""])[0]}
        if self.llm is not None and self.llm.enabled:
            try:
                return self.llm.complete(prompts.SUMMARY_SYSTEM, str(facts))
            except LLMError:
                pass
        status = (f"leaving INR {spec['budget_inr'] - plan['cost']['total']:,} of your budget unused"
                  if plan["within_budget"] else f"which is INR {plan['cost']['total'] - spec['budget_inr']:,} over budget")
        return (f"Your {spec['nights']}-night trip to {spec['destination']} for {spec['travellers']} is planned for "
                f"{spec['start_date']:%d %b} to {spec['end_date']:%d %b %Y}. Fly out on {facts['outbound']} and back on "
                f"{facts['return']}, staying at {facts['hotel']}. Expect {facts['weather']}. The full trip comes to about "
                f"INR {plan['cost']['total']:,} including food, local travel, activities and a 5% buffer, {status}. "
                f"Top tip: {facts['top_tip']} Prices are indicative and must be confirmed at booking.")

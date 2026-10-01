"""
Itinerary Agent - Retrieval-Augmented Generation.

1. RETRIEVE  attractions + local tips for the destination (TF-IDF over the KB),
             using the traveller's interests and weather as the query.
2. AUGMENT   a prompt with trip facts, per-day weather and the retrieved context.
3. GENERATE  a JSON itinerary with the LLM (if a key is configured).
4. VALIDATE  every activity against the KB; drop anything not grounded.
             If the LLM fails or is poorly grounded, a deterministic,
             weather-aware scheduler builds the plan from the same context.
"""

import json
from datetime import timedelta

from agents.base import Agent
from agents.budget_agent import ACTIVITY_SHARE, daily_allowance
from agents.guardrails import grounding_check
from core import prompts
from core.llm import LLMError
from core.rag import format_context, get_retriever

SLOTS = ["Morning", "Afternoon", "Evening"]

# Travel zones for spread-out destinations, so one day stays in one part of town.
ZONES = {
    "Goa": {"Calangute": "North", "Candolim": "North", "Vagator": "North", "Anjuna": "North", "Pilerne": "North",
            "Panaji": "Central", "Old Goa": "Central", "Ponda": "Central", "Mollem": "Central",
            "Palolem": "South", "Benaulim": "South"},
    "Kochi": {"Fort Kochi": "Fort Kochi", "Mattancherry": "Fort Kochi", "Jew Town": "Fort Kochi",
              "Ernakulam": "Mainland", "Thevara": "Mainland", "Vypin": "Vypin", "Alappuzha": "Alappuzha"},
    "Mumbai": {"Colaba": "South", "Fort": "South", "Kala Ghoda": "South", "Churchgate": "South",
               "Elephanta Island": "South", "Worli": "Central", "Dharavi": "Central", "Bandra": "Central",
               "Borivali": "North"},
    "Delhi": {"Old Delhi": "Old Delhi", "Central Delhi": "Central", "Janpath": "Central", "INA": "South",
              "Nizamuddin": "Central", "Mehrauli": "South", "South Delhi": "South", "Kalkaji": "South",
              "Yamuna Bank": "East"},
    "Varanasi": {"Sarnath": "Sarnath", "Ramnagar": "Ramnagar"},
    "Jaipur": {"Amer": "Amer", "Aravalli Hills": "Amer", "Sanganer": "South", "Tonk Road": "South"},
}


def zone(a) -> str:
    return ZONES.get(a["city"], {}).get(a["area"], "Centre")


def _hour(t: str) -> int:
    return int(t[:2])


def available_slots(day_idx: int, n_days: int, arrive: str, depart: str) -> list[str]:
    if day_idx == 0:
        if "+1" in arrive:
            return []
        h = _hour(arrive)
        return ["Afternoon", "Evening"] if h < 11 else (["Evening"] if h < 16 else [])
    if day_idx == n_days - 1:
        h = _hour(depart)
        return ["Morning", "Afternoon"] if h >= 20 else (["Morning"] if h >= 15 else [])
    return list(SLOTS)


def is_open(a, d) -> bool:
    return d.month in a.get("open_months", range(1, 13)) and d.strftime("%a") in a.get(
        "open_days", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])


def score(a, slot, wx, interests, style, day_zones) -> float:
    s = 1.0 + 2.0 * len(set(a["tags"]) & set(interests))
    s += 1.5 if a["best_time"].lower() == slot.lower() else (0.5 if a["best_time"] == "any" else -1.5)
    if wx.get("rainy") and a["setting"] == "indoor":
        s += 1.0
    if wx.get("showers") and a["setting"] == "outdoor" and slot == "Afternoon":
        s -= 1.5
    if wx.get("hot") and a["setting"] == "outdoor" and slot == "Afternoon":
        s -= 2.5
    if style == "Budget" and a["cost_inr"] > 1500:
        s -= 1.0
    if day_zones:
        s += 1.5 if zone(a) in day_zones else -2.0
    return s


class ItineraryAgent(Agent):
    name = "Itinerary Agent"
    role = "Build a day-by-day plan grounded in the destination knowledge base."
    tools = ["RAG retriever (TF-IDF)", "LLM generator", "grounding validator"]

    def run(self, spec, plan, weather) -> dict:
        city = spec["destination"]
        rag = get_retriever()
        query = " ".join(spec["interests"] or ["sightseeing"]) + f" things to do in {city}"
        if weather["rainy_days"]:
            query += " indoor museum rainy day"
        hits = rag.retrieve(query, city=city, k=14, section="Attractions")
        tips_hits = [(c, 0.0) for c in rag.chunks if c.city == city and c.section in
                     ("Food", "Getting Around", "Travel Tips", "Safety and Entry", "Best Time to Visit")]
        self.say("RAG Retriever", "tool_call",
                 f"retrieve('{query[:60]}...', city={city}) -> {len(hits)} attraction chunks + {len(tips_hits)} tip chunks")
        allowance = int(daily_allowance(spec) * ACTIVITY_SHARE * (spec["nights"] + 1))
        kb = {a["name"].lower(): a for a in rag.attractions(city)}

        itinerary, mode = None, "rule-based"
        if self.llm is not None and self.llm.enabled:
            try:
                itinerary = self._llm_plan(spec, plan, weather, hits + tips_hits, allowance, kb)
                mode = f"LLM ({self.llm.provider})"
            except (LLMError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                self.say("Orchestrator", "warning", f"LLM itinerary failed ({str(e)[:80]}); using rule-based scheduler.")
        if itinerary is None:
            itinerary = self._rule_plan(spec, plan, weather, [c.meta for c, _ in hits] + list(kb.values()), allowance)
            itinerary["local_tips"] = self._tips(rag, city)

        g = grounding_check(itinerary, set(kb))
        itinerary.update(mode=mode, grounding=g, activity_allowance_pp=allowance,
                         retrieved=[{"source": c.chunk_id, "score": round(s, 3), "text": c.text[:140]} for c, s in hits])
        self.say("Orchestrator", "result",
                 f"{len(itinerary['days'])}-day itinerary via {mode}; {g['grounded']}/{g['total']} activities grounded "
                 f"in knowledge base; est. entry fees INR {itinerary['activity_cost_pp']:,} pp.")
        self.board.put("itinerary", itinerary)
        return itinerary

    # ------------------------------------------------------------------ #
    def _rule_plan(self, spec, plan, weather, candidates, allowance) -> dict:
        seen, pool = set(), []
        for a in candidates:
            if a["name"] not in seen:
                seen.add(a["name"])
                pool.append(a)
        used, spent, days = set(), 0, []
        n_days = spec["nights"] + 1
        for i in range(n_days):
            wx = weather["days"][i] if i < len(weather["days"]) else {}
            day_date = spec["start_date"] + timedelta(days=i)
            date = day_date.isoformat()
            slots = available_slots(i, n_days, plan["outbound"]["arrive"], plan["return"]["depart"])
            items, areas, filled = [], set(), set()
            for slot in slots:
                if slot in filled:
                    continue
                best, best_s = None, -1.0
                for a in pool:
                    if a["name"] in used or spent + a["cost_inr"] > allowance or not is_open(a, day_date):
                        continue
                    if wx.get("rainy") and a["setting"] == "outdoor":
                        continue  # hard rule: no outdoor plans on heavy-rain days
                    if a["hours"] >= 5 and not (slot == "Morning" and "Afternoon" in slots):
                        continue
                    s = score(a, slot, wx, spec["interests"], spec["style"], areas)
                    if s > best_s:
                        best, best_s = a, s
                if best is None and wx.get("rainy") and slot != "Evening":
                    items.append({"time": slot, "kind": "free", "activity": "Rain buffer - cafe, spa or rest at hotel",
                                  "note": "Heavy rain expected; keep this slot flexible."})
                if best:
                    used.add(best["name"])
                    spent += best["cost_inr"]
                    areas.add(zone(best))
                    span = "Morning-Afternoon" if best["hours"] >= 5 else slot
                    if best["hours"] >= 5:
                        filled.add("Afternoon")
                    items.append({"time": span, "activity": best["name"], "source": best["source"],
                                  "area": best["area"], "setting": best["setting"], "hours": best["hours"],
                                  "cost_inr": best["cost_inr"], "note": best["description"], "kind": "activity"})
                filled.add(slot)
            if i == 0:
                items.insert(0, {"time": "Arrival", "kind": "free",
                                 "activity": f"Land {plan['outbound']['arrive']}, check in at {plan['hotel']['name']}",
                                 "note": f"Hotel area: {plan['hotel']['area']}"})
            if i == n_days - 1:
                items.append({"time": "Departure", "kind": "free",
                              "activity": f"Check out, fly {plan['return']['flight_no']} at {plan['return']['depart']}",
                              "note": "Reach the airport 2 h before domestic, 3 h before international departures."})
            if not [x for x in items if x["kind"] == "activity"] and 0 < i < n_days - 1:
                items.append({"time": "Flexible", "kind": "free", "activity": "Free day - rest or explore local food",
                              "note": "See local food tips."})
            theme = self._theme(items, wx)
            days.append({"day": i + 1, "date": date, "theme": theme, "weather": wx.get("advice", ""), "items": items})
        return {"days": days, "activity_cost_pp": spent, "assumptions": [
            "Opening days and fees are approximate; check before visiting.",
            "Travel time between areas is not optimised to the minute."]}

    @staticmethod
    def _theme(items, wx):
        acts = [x for x in items if x["kind"] == "activity"]
        if not acts:
            if items and items[0]["time"] == "Arrival":
                return "Travel and settle in"
            return "Travel day" if items and items[-1]["time"] == "Departure" else "Free / rain-buffer day"
        settings = {x["setting"] for x in acts}
        if wx.get("rainy") and settings == {"indoor"}:
            return "Rainy-day indoor highlights"
        return " & ".join(dict.fromkeys(x["area"] for x in acts))

    @staticmethod
    def _tips(rag, city):
        tips = []
        for sec in ("Travel Tips", "Getting Around", "Food", "Safety and Entry"):
            txt = rag.section(city, sec)
            if txt:
                tips.append(txt.split(". ")[0].rstrip(".") + ".")
        return tips

    def _llm_plan(self, spec, plan, weather, hits, allowance, kb) -> dict:
        wx_lines = "\n".join(f"Day {i + 1} {d['date']}: {d['condition']}, high {d['high_c']} C, "
                             f"{'RAINY' if d['rainy'] else ''}{' HOT' if d['hot'] else ''}".strip()
                             for i, d in enumerate(weather["days"]))
        user = prompts.ITINERARY_USER.format(
            nights=spec["nights"], city=spec["destination"], travellers=spec["travellers"], style=spec["style"],
            start=spec["start_date"], end=spec["end_date"], interests=", ".join(spec["interests"]) or "general",
            hotel_area=plan["hotel"]["area"], weather=wx_lines, activity_allowance=allowance,
            context=format_context(hits))
        self.say("LLM", "tool_call", f"generate itinerary ({len(user)} chars prompt, {len(hits)} context chunks)")
        raw = self.llm.complete_json(prompts.ITINERARY_SYSTEM, user)
        days, dropped, spent = [], [], 0
        for d in raw["days"]:
            items = []
            for it in d.get("items", []):
                name = str(it.get("activity", "")).strip()
                a = kb.get(name.lower()) or next((v for k, v in kb.items() if k in name.lower() or name.lower() in k), None)
                i_day = int(d.get("day", len(days) + 1)) - 1
                wx_d = weather["days"][i_day] if 0 <= i_day < len(weather["days"]) else {}
                if not a:
                    dropped.append(name)
                    continue
                if not is_open(a, spec["start_date"] + timedelta(days=i_day)) or \
                        (wx_d.get("rainy") and a["setting"] == "outdoor"):
                    dropped.append(f"{name} (closed or rainy)")
                    continue
                spent += a["cost_inr"]
                items.append({"time": it.get("time", ""), "activity": a["name"], "source": a["source"],
                              "area": a["area"], "setting": a["setting"], "hours": a["hours"],
                              "cost_inr": a["cost_inr"], "note": it.get("note") or a["description"], "kind": "activity"})
            i = int(d.get("day", len(days) + 1)) - 1
            wx = weather["days"][i] if 0 <= i < len(weather["days"]) else {}
            days.append({"day": i + 1, "date": d.get("date") or (spec["start_date"] + timedelta(days=i)).isoformat(),
                         "theme": d.get("theme", ""), "weather": wx.get("advice", ""), "items": items})
        total = sum(len(d["items"]) for d in days) + len(dropped)
        if not days or (total and len(dropped) / total > 0.3):
            raise LLMError(f"poorly grounded output ({len(dropped)} of {total} items not in KB)")
        if dropped:
            self.say("Orchestrator", "warning", f"Removed {len(dropped)} ungrounded item(s): {dropped}")
        return {"days": days, "activity_cost_pp": spent, "local_tips": raw.get("local_tips", [])[:5],
                "llm_dropped": dropped, "llm_items_proposed": total,
                "assumptions": raw.get("assumptions", []) + ([f"Removed ungrounded: {dropped}"] if dropped else [])}

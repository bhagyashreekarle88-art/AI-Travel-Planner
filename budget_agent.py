"""
Budget Agent - constrained optimisation.

Decision variables : outbound flight f1, return flight f2, hotel h
Objective          : maximise U = 0.50*Q(h) + 0.25*C(f1) + 0.25*C(f2) - lambda * cost/budget
Constraint         : flights + hotel + daily spend + 5% contingency <= budget
lambda (price sensitivity) depends on travel style.

The search space is small (~7 x 7 x 10 = 490 plans) so an exhaustive search
finds the exact optimum in milliseconds; for larger inventories the same
objective works with greedy pruning or an ILP solver.
If nothing is feasible the agent negotiates: it asks the Hotel Agent to widen
its search, then the Flight Agent to scan flexible dates, and finally returns
concrete suggestions (fewer nights, cheaper style) with the gap in rupees.
"""

from itertools import product

from agents.base import Agent
from core.data import CITIES, DAILY_SPEND

PRICE_SENSITIVITY = {"Budget": 0.9, "Moderate": 0.45, "Luxury": 0.15}
CONTINGENCY = 0.05
ACTIVITY_SHARE = 0.35  # share of daily spend reserved for entry fees / experiences


def spend_days(spec) -> int:
    """Full days in between plus two half travel days = number of nights (min 1)."""
    return max(spec["nights"], 1)


def daily_allowance(spec) -> int:
    return int(DAILY_SPEND[spec["style"]] * CITIES[spec["destination"]]["cost_index"])


def cost_of(spec, f_out, f_ret, hotel) -> dict:
    n, days = spec["travellers"], spend_days(spec)
    daily = daily_allowance(spec) * n * days
    flights = (f_out["price_pp"] + f_ret["price_pp"]) * n
    stay = hotel["total_price"]
    subtotal = flights + stay + daily
    contingency = int(subtotal * CONTINGENCY)
    return {"flights": flights, "hotel": stay,
            "food_local_transport": int(daily * (1 - ACTIVITY_SHARE)),
            "activities": int(daily * ACTIVITY_SHARE),
            "contingency": contingency, "total": subtotal + contingency}


def utility(spec, f_out, f_ret, hotel, total) -> float:
    lam = PRICE_SENSITIVITY[spec["style"]]
    return round(0.5 * hotel["quality"] + 0.25 * f_out["convenience"] + 0.25 * f_ret["convenience"]
                 - lam * total / spec["budget_inr"], 4)


def _plan(spec, combo, label):
    f1, f2, h = combo
    c = cost_of(spec, f1, f2, h)
    return {"label": label, "outbound": f1, "return": f2, "hotel": h, "cost": c,
            "utility": utility(spec, f1, f2, h, c["total"]),
            "within_budget": c["total"] <= spec["budget_inr"],
            "budget_left": spec["budget_inr"] - c["total"]}


class BudgetAgent(Agent):
    name = "Budget Agent"
    role = "Choose the best flight + hotel combination that fits the total budget."
    tools = ["exhaustive constrained optimiser"]

    def optimise(self, spec, flights, hotels) -> dict:
        combos = list(product(flights["outbound"], flights["return"], hotels))
        self.say("Orchestrator", "info", f"Evaluating {len(combos)} candidate plans against budget INR {spec['budget_inr']:,}.")
        scored = []
        for f1, f2, h in combos:
            c = cost_of(spec, f1, f2, h)
            scored.append((utility(spec, f1, f2, h, c["total"]), c["total"], (f1, f2, h)))
        feasible = [s for s in scored if s[1] <= spec["budget_inr"]]
        cheapest = min(scored, key=lambda s: s[1])
        result = {"evaluated": len(combos), "feasible_count": len(feasible),
                  "allowance_pp_day": daily_allowance(spec), "status": "ok"}
        if not feasible:
            result.update(status="over_budget", cheapest=_plan(spec, cheapest[2], "Cheapest possible"),
                          gap=cheapest[1] - spec["budget_inr"])
            self.say("Hotel Agent", "feedback",
                     f"No plan fits. Cheapest is INR {cheapest[1]:,} (over by INR {result['gap']:,}). "
                     "Please widen hotel search.")
            return result
        best = max(feasible, key=lambda s: s[0])
        value = min(feasible, key=lambda s: s[1])
        comfort = max(feasible, key=lambda s: s[0] + PRICE_SENSITIVITY[spec["style"]] * s[1] / spec["budget_inr"])
        plans = [_plan(spec, best[2], "Recommended (best value)")]
        for s, label in [(value, "Lowest cost"), (comfort, "Most comfortable within budget")]:
            if s[2] != best[2]:
                plans.append(_plan(spec, s[2], label))
        result["plans"] = plans
        b = plans[0]
        self.say("Orchestrator", "result",
                 f"{len(feasible)} of {len(combos)} plans fit. Recommended total INR {b['cost']['total']:,} "
                 f"(INR {b['budget_left']:,} left): {b['outbound']['airline']} + {b['hotel']['name']}.")
        self.board.put("budget", result)
        return result

    def suggestions(self, spec, result, date_options) -> list[str]:
        tips, gap = [], result["gap"]
        per_night = result["cheapest"]["hotel"]["price_per_night"] * result["cheapest"]["hotel"]["rooms"] \
            + daily_allowance(spec) * spec["travellers"]
        cut = -(-gap // max(per_night, 1))
        if cut <= spec["nights"] - 1:
            tips.append(f"Shorten the trip by {cut} night(s) - saves about INR {cut * per_night:,}.")
        elif spec["nights"] > 1:
            tips.append("Shortening the trip alone cannot close the gap - consider a closer or cheaper destination.")
        if spec["style"] != "Budget":
            lower = "Moderate" if spec["style"] == "Luxury" else "Budget"
            saving = (DAILY_SPEND[spec["style"]] - DAILY_SPEND[lower]) * CITIES[spec["destination"]]["cost_index"] \
                * spec["travellers"] * spend_days(spec)
            tips.append(f"Switch daily spending to '{lower}' - saves about INR {int(saving):,}.")
        cur = result["cheapest"]["outbound"]["price_pp"] + result["cheapest"]["return"]["price_pp"]
        if date_options and (cur - date_options[0]["round_trip_pp"]) * spec["travellers"] >= 500 * spec["travellers"]:
            d = date_options[0]
            tips.append(f"Travel {d['start']} to {d['end']} instead - flights about INR "
                        f"{(cur - d['round_trip_pp']) * spec['travellers']:,} cheaper for the group.")
        tips.append(f"Or raise the budget to at least INR {result['cheapest']['cost']['total']:,}.")
        self.say("Orchestrator", "result", "Budget negotiation suggestions: " + " | ".join(tips))
        return tips

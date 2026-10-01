"""
AI Travel Planner with Autonomous Agents - Streamlit front end.
Run locally:  streamlit run app.py
"""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from agents.orchestrator import Orchestrator
from core.data import INTERESTS, destinations, origins
from core.export import plan_to_markdown
from core.llm import DEFAULT_MODELS, OFFLINE, PROVIDERS, LLMClient

st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")

STYLES = ["Budget", "Moderate", "Luxury"]
DEFAULTS = {"origin": "Mumbai", "destination": "Goa", "start_date": date.today() + timedelta(days=30),
            "nights": 4, "travellers": 2, "budget_inr": 60000, "style": "Moderate", "interests": ["beach", "food"]}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:  # no secrets.toml present
        return ""


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ AI engine")
    provider = st.selectbox("LLM provider", PROVIDERS,
                            index=PROVIDERS.index(secret("LLM_PROVIDER")) if secret("LLM_PROVIDER") in PROVIDERS else 0,
                            help="Offline mode needs no key. Gemini and Groq have free tiers.")
    api_key, model = "", ""
    if provider != OFFLINE:
        key_name = {"Google Gemini": "GEMINI_API_KEY", "Groq": "GROQ_API_KEY", "OpenAI": "OPENAI_API_KEY"}[provider]
        api_key = st.text_input(f"{provider} API key", value=secret(key_name), type="password",
                                help=f"Or set {key_name} in .streamlit/secrets.toml")
        model = st.text_input("Model", value=DEFAULT_MODELS[provider])
        if not api_key:
            st.info("No key entered - the planner will run in offline mode.")
    live_weather = st.toggle("Live weather (Open-Meteo)", value=True)
    st.divider()
    st.markdown("**Agents in this system**")
    st.markdown("🧭 Orchestrator · ✈️ Flight · 🏨 Hotel · 🌦️ Weather · 💰 Budget · 🗺️ Itinerary · ✅ Critic")
    st.caption("TEC 406 - Foundations of Generative AI and Agentic AI · Capstone Project 6")


def make_orchestrator() -> Orchestrator:
    llm = LLMClient(provider, api_key, model) if provider != OFFLINE else None
    return Orchestrator(llm=llm, allow_network=live_weather)


def understand_request():
    text = st.session_state.get("free_text", "").strip()
    if not text:
        return
    spec, warnings = make_orchestrator().parse_request(text)
    filled = []
    limits = {"nights": (1, 21), "travellers": (1, 9), "budget_inr": (3000, 2_000_000)}
    for k in ["origin", "destination", "nights", "travellers", "budget_inr", "style"]:
        v = spec.get(k)
        if not v:
            continue
        if k in limits:
            v = min(max(int(v), limits[k][0]), limits[k][1])
        if k == "origin" and v not in origins() or k == "destination" and v not in destinations() \
                or k == "style" and v not in STYLES:
            continue
        st.session_state[k] = v
        filled.append(k)
    if spec.get("start_date"):
        try:
            st.session_state["start_date"] = date.fromisoformat(str(spec["start_date"]))
            filled.append("start_date")
        except ValueError:
            pass
    if spec.get("interests"):
        st.session_state["interests"] = [i for i in spec["interests"] if i in INTERESTS]
        filled.append("interests")
    st.session_state["parse_note"] = (filled, spec.get("assumptions", []), warnings)


# ------------------------------------------------------------------ header + input
st.title("✈️ AI Travel Planner")
st.caption("Multi-agent planner: flights, hotels, weather, budget optimisation and a RAG-grounded itinerary.")

with st.container(border=True):
    st.text_area("Describe your trip in your own words (optional)", key="free_text", height=80,
                 placeholder="e.g. Me and my wife want a relaxed beach trip from Mumbai, 4 nights from 12 Dec, "
                             "budget 60k, we love seafood and some history")
    st.button("🧭 Understand my request", on_click=understand_request)
    if note := st.session_state.get("parse_note"):
        filled, assumptions, warnings = note
        st.success(f"Filled in: {', '.join(filled) or 'nothing recognised'}. Check the form below before planning.")
        for a in assumptions:
            st.caption(f"Assumption: {a}")
        for w in warnings:
            st.warning(w)

    c1, c2, c3, c4 = st.columns(4)
    c1.selectbox("From", origins(), key="origin")
    c2.selectbox("To", destinations(), key="destination")
    c3.date_input("Start date", key="start_date", min_value=date.today(),
                  max_value=date.today() + timedelta(days=330))
    c4.number_input("Nights", 1, 21, key="nights")
    c5, c6, c7, c8 = st.columns(4)
    c5.number_input("Travellers", 1, 9, key="travellers")
    c6.number_input("Total budget (INR)", 3000, 2_000_000, step=5000, key="budget_inr")
    c7.selectbox("Travel style", STYLES, key="style")
    c8.multiselect("Interests", INTERESTS, key="interests")
    go = st.button("🚀 Plan my trip", type="primary")

if go:
    spec = {k: st.session_state[k] for k in DEFAULTS}
    with st.spinner("Agents are collaborating on your plan..."):
        st.session_state["result"] = make_orchestrator().plan(spec)

r = st.session_state.get("result")
if not r:
    st.info("Fill in the trip details and press **Plan my trip**.")
    st.stop()
if not r["ok"]:
    for e in r["errors"]:
        st.error(e)
    st.stop()

# ------------------------------------------------------------------ results
s, p, w, it = r["spec"], r["plan"], r["weather"], r["itinerary"]
cost = p["cost"]
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total trip cost", f"₹{cost['total']:,}")
m2.metric("Budget left", f"₹{p['budget_left']:,}", delta="within budget" if p["within_budget"] else "over budget",
          delta_color="normal" if p["within_budget"] else "inverse")
m3.metric("Flights", f"₹{cost['flights']:,}")
m4.metric("Hotel", f"₹{cost['hotel']:,}", help=p["hotel"]["name"])
m5.metric("Avg high", f"{w['avg_high']} °C", help=f"{w['rainy_days']} rainy day(s)")

if not p["within_budget"]:
    st.error("No combination fits this budget, even after the agents negotiated. Here is the cheapest plan and how to close the gap:")
    for t in r["suggestions"]:
        st.markdown(f"- {t}")
elif r["negotiated"]:
    st.warning("The Budget Agent asked the Hotel Agent to widen the search to fit your budget.")
if r["international"]:
    st.info("International trip - check passport validity, visa and entry rules with the official embassy website.")

st.markdown(f"> {r['summary']}")

tabs = st.tabs(["🗺️ Itinerary", "✈️ Flights", "🏨 Hotels", "🌦️ Weather", "💰 Budget", "🤖 Agent trace", "✅ Quality checks"])

with tabs[0]:
    st.caption(f"Generated by: {it['mode']} · {it['grounding']['grounded']}/{it['grounding']['total']} activities "
               f"verified against the knowledge base · est. entry fees ₹{it['activity_cost_pp']:,} per person")
    for d in it["days"]:
        with st.expander(f"Day {d['day']} · {d['date']} · {d['theme']}", expanded=d["day"] <= 2):
            if d.get("weather"):
                st.caption(f"🌦️ {d['weather']}")
            for x in d["items"]:
                icon = "🏛️" if x.get("setting") == "indoor" else ("🌳" if x.get("setting") == "outdoor" else "🧳")
                fee = f" · ₹{x['cost_inr']:,}" if x.get("cost_inr") else ""
                st.markdown(f"**{x['time']}** {icon} {x['activity']}{fee}  \n<small>{x.get('note', '')}</small>",
                            unsafe_allow_html=True)
    if it.get("local_tips"):
        st.subheader("Local tips")
        for t in it["local_tips"]:
            st.markdown(f"- {t}")
    st.download_button("⬇️ Download plan (Markdown)", plan_to_markdown(r), file_name=f"trip_{s['destination'].lower()}.md")

with tabs[1]:
    cols = ["airline", "flight_no", "depart", "arrive", "duration_min", "stops", "price_pp", "convenience"]
    for title, key, chosen in [("Outbound", "outbound", p["outbound"]["id"]), ("Return", "return", p["return"]["id"])]:
        st.subheader(f"{title} · {r['flights'][key][0]['date']}")
        df = pd.DataFrame(r["flights"][key])[cols + ["id"]]
        df.insert(0, "pick", df["id"].eq(chosen).map({True: "⭐", False: ""}))
        st.dataframe(df.drop(columns="id"), hide_index=True)

with tabs[2]:
    df = pd.DataFrame(r["hotels"])[["id", "name", "stars", "rating", "reviews", "area", "price_per_night",
                                    "total_price", "quality", "free_cancellation"]]
    df.insert(0, "pick", df["id"].eq(p["hotel"]["id"]).map({True: "⭐", False: ""}))
    st.dataframe(df.drop(columns="id"), hide_index=True)

with tabs[3]:
    st.caption(f"Source: {w['source']}")
    wdf = pd.DataFrame(w["days"])
    st.line_chart(wdf.set_index("date")[["high_c", "low_c"]])
    st.dataframe(wdf[["date", "condition", "high_c", "low_c", "rain_mm", "advice"]], hide_index=True)

with tabs[4]:
    bdf = pd.DataFrame({"Item": ["Flights", "Hotel", "Food & local", "Activities", "Contingency"],
                        "INR": [cost["flights"], cost["hotel"], cost["food_local_transport"],
                                cost["activities"], cost["contingency"]]})
    st.bar_chart(bdf.set_index("Item"))
    b = r["budget"]
    st.caption(f"The Budget Agent evaluated {b['evaluated']} flight + hotel combinations; "
               f"{b['feasible_count']} fit the budget. Daily allowance ₹{b['allowance_pp_day']:,} per person.")
    if b["status"] == "ok":
        alt = pd.DataFrame([{"plan": x["label"], "total": x["cost"]["total"], "hotel": x["hotel"]["name"],
                             "stars": x["hotel"]["stars"], "outbound": x["outbound"]["airline"],
                             "return": x["return"]["airline"], "utility": x["utility"]} for x in b["plans"]])
        st.dataframe(alt, hide_index=True)

with tabs[5]:
    st.caption("Every message exchanged between agents and tools during this run.")
    st.dataframe(pd.DataFrame(r["trace"]), hide_index=True, height=420)
    st.caption(f"Total latency {r['latency_s']} s · LLM calls {r['llm_calls']}")

with tabs[6]:
    st.dataframe(pd.DataFrame(r["checks"]).replace({True: "✅", False: "❌"}), hide_index=True)

st.divider()
st.caption(r["disclaimer"])

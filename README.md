# ✈️ AI Travel Planner with Autonomous Agents

Capstone / Applied AI Mini Project · **TEC 406 – Foundations of Generative AI and Agentic AI** · Project 6 (Tourism)

A multi-agent system that turns a plain-English trip request into a complete, budget-checked travel plan. It
finds flights, finds hotels, checks the weather, optimises the budget, and writes a day-by-day itinerary
grounded in a destination knowledge base (RAG).

## Agents

| Agent | Job | Tools |
|---|---|---|
| 🧭 Orchestrator | Understands the request, delegates, resolves conflicts | LLM parser / rule parser, guardrails |
| ✈️ Flight Agent | Outbound + return options, convenience score, flexible-date scan | Flight inventory API |
| 🏨 Hotel Agent | Hotels by travel style, quality score, widens search on request | Hotel inventory API |
| 🌦️ Weather Agent | Daily outlook, flags rain/heat | Open-Meteo forecast + archive APIs, climate normals |
| 💰 Budget Agent | Picks the best flight + hotel combination within budget; negotiates if over | Constrained optimiser |
| 🗺️ Itinerary Agent | Day-wise plan, weather-aware, grounded in the knowledge base | RAG retriever, LLM, validator |
| ✅ Critic | Checks 8 constraints before the plan is shown | Rule checks |

Flight, Hotel and Weather agents run **in parallel**; the Budget Agent then optimises, and if no plan fits it asks
the Hotel Agent to widen the search and the Flight Agent to scan ±2 days, then proposes trade-offs.

## Project structure

```
app.py                     Streamlit front end
agents/                    orchestrator + 5 specialist agents, guardrails, shared blackboard
core/                      LLM client, RAG retriever, prompts, simulated inventory, reference data
data/knowledge_base/       10 destination guides (the RAG corpus)
evaluation/evaluate.py     metrics harness (20 scenarios + baseline comparison)
tests/test_agents.py       unit tests
```

## Run locally

```bash
git clone https://github.com/<your-username>/ai-travel-planner.git
cd ai-travel-planner
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app works with **no API key** (offline rule-based mode). To enable the LLM, pick a provider in the sidebar
and paste a key, or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill it in.
Google Gemini and Groq both offer free tiers.

## Deploy on Streamlit Community Cloud

1. Push this folder to a **public GitHub repository** (do not commit `secrets.toml`; `.gitignore` already excludes it).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**.
3. Choose the repository, branch `main`, main file `app.py`, then **Deploy**.
4. Optional: in **App settings → Secrets**, paste the contents of `secrets.toml.example` with your real key.

## Evaluate

```bash
python -m evaluation.evaluate                          # offline, deterministic
python -m evaluation.evaluate --llm gemini --key KEY   # adds an LLM-as-judge score
python -m pytest -q                                    # unit tests
```

## Honest limitations

* Flight and hotel results are **simulated** (deterministic mock APIs that respond to distance, season and
  booking window); real booking APIs need partner access. Replace `core/inventory.py` to go live.
* Weather is real when Open-Meteo is reachable; otherwise monthly climate averages are used.
* Visa and entry rules change; the app always points users to official sources.

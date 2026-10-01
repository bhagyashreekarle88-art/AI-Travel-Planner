"""
Weather Agent - real API (Open-Meteo, free, no key) with graceful fallbacks.

Strategy (planning & reasoning):
  trip starts within 15 days  -> live daily FORECAST
  trip further out            -> same dates LAST YEAR from the historical archive
                                 (a real-data seasonal reference)
  network unavailable         -> monthly CLIMATE NORMALS from the local dataset
Each day is then tagged (rainy / hot / pleasant) so the Itinerary Agent can
move outdoor activities indoors or to cooler hours.
"""

from calendar import monthrange
from datetime import date, timedelta

import requests

from agents.base import Agent
from core.data import CITIES, CLIMATE_NORMALS

WMO = {0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Fog",
       51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle", 61: "Light rain", 63: "Rain",
       65: "Heavy rain", 80: "Rain showers", 81: "Rain showers", 82: "Violent showers",
       95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm"}
DAILY = "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code"


def _tag(day: dict) -> dict:
    prob = day.get("rain_prob")
    rainy = prob >= 0.70 if prob is not None else day["rain_mm"] >= 8
    showers = not rainy and (prob >= 0.35 if prob is not None else day["rain_mm"] >= 1)
    hot = day["high_c"] >= 36
    day["rainy"], day["showers"], day["hot"] = rainy, showers, hot
    if rainy:
        day["advice"] = "Rain likely - favour indoor sights, carry an umbrella."
    elif showers and hot:
        day["advice"] = "Hot with possible showers - outdoor sights early, indoor in the afternoon."
    elif showers:
        day["advice"] = "Passing showers possible - keep afternoons flexible and carry an umbrella."
    elif hot:
        day["advice"] = "Very hot - do outdoor sights early morning or after 5 pm."
    elif day["high_c"] <= 15:
        day["advice"] = "Cool - pack a warm layer for mornings and evenings."
    else:
        day["advice"] = "Good weather for outdoor sightseeing."
    return day


class WeatherAgent(Agent):
    name = "Weather Agent"
    role = "Get the weather outlook for each trip day and flag rain or heat risk."
    tools = ["Open-Meteo forecast API", "Open-Meteo archive API", "climate normals"]

    def __init__(self, board, llm=None, allow_network: bool = True, timeout: int = 8):
        super().__init__(board, llm)
        self.allow_network = allow_network
        self.timeout = timeout

    def run(self, spec: dict, today: date | None = None) -> dict:
        today = today or date.today()
        city = spec["destination"]
        start, end = spec["start_date"], spec["end_date"]
        days, source = None, None
        if self.allow_network:
            try:
                if (start - today).days <= 14 and (end - today).days <= 15:
                    days, source = self._fetch("https://api.open-meteo.com/v1/forecast", city, start, end), \
                        "Open-Meteo live forecast"
                else:
                    ly_s, ly_e = self._last_year(start), self._last_year(end)
                    days = self._fetch("https://archive-api.open-meteo.com/v1/archive", city, ly_s, ly_e)
                    for i, d in enumerate(days):
                        d["date"] = (start + timedelta(days=i)).isoformat()
                    source = f"Open-Meteo historical (same dates {ly_s.year}) - seasonal reference"
            except Exception as e:  # network blocked, timeout, API change
                self.say("Orchestrator", "warning", f"Weather API unavailable ({type(e).__name__}); using climate normals.")
                days = None
        if days is None:
            days, source = self._normals(city, start, end), "Climate normals (monthly averages)"
        days = [_tag(d) for d in days]
        summary = {
            "source": source, "days": days,
            "avg_high": round(sum(d["high_c"] for d in days) / len(days), 1),
            "rainy_days": sum(d["rainy"] for d in days),
            "hot_days": sum(d["hot"] for d in days),
        }
        self.say("Orchestrator", "result",
                 f"{city}: avg high {summary['avg_high']} C, {summary['rainy_days']} rainy and "
                 f"{summary['hot_days']} very hot day(s) of {len(days)}. Source: {source}.")
        self.board.put("weather", summary)
        return summary

    # ----------------------------------------------------------------------
    @staticmethod
    def _last_year(d: date) -> date:
        try:
            return d.replace(year=d.year - 1)
        except ValueError:  # 29 Feb
            return d.replace(year=d.year - 1, day=28)

    def _fetch(self, url, city, start, end) -> list[dict]:
        c = CITIES[city]
        self.say("Open-Meteo", "tool_call", f"GET {url.split('//')[1].split('/')[0]} {city} {start}..{end}")
        r = requests.get(url, timeout=self.timeout, params={
            "latitude": c["lat"], "longitude": c["lon"], "daily": DAILY, "timezone": "auto",
            "start_date": start.isoformat(), "end_date": end.isoformat()})
        r.raise_for_status()
        dd = r.json()["daily"]
        return [{"date": dd["time"][i],
                 "high_c": round(dd["temperature_2m_max"][i] or 0, 1),
                 "low_c": round(dd["temperature_2m_min"][i] or 0, 1),
                 "rain_mm": round(dd["precipitation_sum"][i] or 0, 1),
                 "condition": WMO.get(dd["weather_code"][i], "Mixed")}
                for i in range(len(dd["time"]))]

    def _normals(self, city, start, end) -> list[dict]:
        out, d = [], start
        while d <= end:
            hi, lo, rd = CLIMATE_NORMALS[city][d.month]
            p = rd / monthrange(d.year, d.month)[1]
            out.append({"date": d.isoformat(), "high_c": hi, "low_c": lo,
                        "rain_mm": round(12 * p, 1), "rain_prob": round(p, 2),
                        "condition": "Frequent heavy rain" if p >= 0.70 else ("Showers likely" if p >= 0.35 else "Mostly dry")})
            d += timedelta(days=1)
        return out

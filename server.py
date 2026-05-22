"""
Dota 2 Oracle — Backend API Server
Handles all AI + web search requests for the Mini App
"""

import os
import json
import re
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CURRENT_PATCH = "7.41c"

# Trusted data sources
SOURCES = "dota2protracker.com, dotabuff.com, stratz.com, liquipedia.net/dota2"

# ── Setup ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow Mini App to call this server

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Helpers ───────────────────────────────────────────────────────────────────

def search_and_analyze(search_query: str, analysis_prompt: str, max_tokens: int = 800) -> dict:
    """
    Step 1: Use Sonnet with web search to get live data from DotaBuff/ProTracker
    Step 2: Use Haiku to structure the data into clean JSON
    """

    # Step 1 — Live web search
    logger.info(f"Searching: {search_query[:80]}...")
    try:
        search_resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            system=f"""You are a Dota 2 data analyst. 
Search ONLY these trusted sources: {SOURCES}
Find current patch {CURRENT_PATCH} data and summarize the key statistics in under 200 words.
Focus only on win rates, pick rates, tier rankings, and gameplay facts.""",
            messages=[{"role": "user", "content": search_query}]
        )
        live_data = " ".join(
            b.text for b in search_resp.content if b.type == "text"
        ).strip()
        logger.info(f"Search complete, got {len(live_data)} chars")
    except Exception as e:
        logger.error(f"Search error: {e}")
        live_data = f"No live data available. Use general patch {CURRENT_PATCH} knowledge."

    # Step 2 — Structure into JSON
    full_prompt = f"""Live data from {SOURCES}:
{live_data}

{analysis_prompt}

Respond ONLY with valid JSON, no markdown, no extra text."""

    haiku_resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        system="You are a Dota 2 expert. Respond only with valid JSON. No markdown. No extra text.",
        messages=[{"role": "user", "content": full_prompt}]
    )

    raw = " ".join(b.text for b in haiku_resp.content if b.type == "text")
    m = re.search(r'\{[\s\S]*\}', raw)
    if not m:
        raise ValueError("No JSON in response")
    return json.loads(m.group())


def fast_analyze(prompt: str, max_tokens: int = 700) -> dict:
    """Fast Haiku-only analysis for simple requests (no web search needed)"""
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        system="You are a Dota 2 expert. Respond only with valid JSON. No markdown. No extra text.",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = " ".join(b.text for b in resp.content if b.type == "text")
    m = re.search(r'\{[\s\S]*\}', raw)
    if not m:
        raise ValueError("No JSON in response")
    return json.loads(m.group())


def error_response(msg: str, status: int = 500):
    return jsonify({"error": msg}), status


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "patch": CURRENT_PATCH})


@app.route("/draft", methods=["POST"])
def draft():
    """Last pick analyzer with live meta data"""
    try:
        body = request.get_json()
        allies = body.get("allies", {})
        enemies = body.get("enemies", {})
        last_pos = body.get("last_pos", 1)
        expected5 = body.get("expected5", "")
        bracket = body.get("bracket", "Legend")

        pos_labels = ["Carry", "Mid", "Off Lane", "Support", "Hard Support"]
        all_heroes = list(allies.values()) + list(enemies.values())
        if expected5:
            all_heroes.append(expected5)

        search_query = (
            f"Search {SOURCES} for Dota 2 patch {CURRENT_PATCH}: "
            f"current win rates and meta strength for these heroes: {', '.join(all_heroes)}. "
            f"Also find the strongest Position {last_pos} ({pos_labels[last_pos-1]}) heroes in current meta. "
            f"Include tier rankings and recent buff/nerf info."
        )

        ally_lines = "\n".join(
            f"  Pos {i} {pos_labels[int(i)-1]}: {h}" if int(i) != last_pos
            else f"  Pos {i} {pos_labels[int(i)-1]}: [LAST PICK]"
            for i, h in allies.items()
        ) + f"\n  Pos {last_pos} {pos_labels[last_pos-1]}: [LAST PICK]"

        enemy_lines = "\n".join(f"  Enemy {i}: {h}" for i, h in enemies.items())
        enemy_lines += f"\n  Enemy 5: {expected5}" if expected5 else "\n  Enemy 5: UNKNOWN (also last picking)"

        important = (
            f"Enemy expected last pick: {expected5}. Pick handles both their lineup AND {expected5}."
            if expected5 else
            "Enemy last pick unknown. Favor heroes hard to counter-pick."
        )

        analysis_prompt = f"""Based on the live data above, give a last pick recommendation for this Dota 2 draft.

My team:
{ally_lines}

Enemy team (positions unknown):
{enemy_lines}

Last pick position: {last_pos} ({pos_labels[last_pos-1]})
Skill bracket: {bracket}
{important}

Priority order:
1. META STRENGTH in patch {CURRENT_PATCH} based on live DotaBuff/ProTracker data
2. SYNERGY with allied heroes
3. COUNTER enemy lineup

JSON format:
{{"pick1":{{"hero":"Name","tags":["S-tier meta","synergy X","counters Y"],"reason":"2-3 sentences leading with current win rate or pro pick rate from live data."}},"pick2":{{"hero":"Name","tags":["A-tier meta","synergy X"],"reason":"2-3 sentences."}},"pick3":{{"hero":"Name","tags":["A-tier meta","safe pick"],"reason":"2-3 sentences."}},"threat":"One sentence on biggest enemy threat based on current meta.","item_hint":"Top 3 core items for pick1 in current patch."}}"""

        data = search_and_analyze(search_query, analysis_prompt)
        return jsonify(data)

    except Exception as e:
        logger.error(f"Draft error: {e}")
        return error_response(str(e))


@app.route("/tierlist", methods=["POST"])
def tierlist():
    """Live tier list from DotaBuff & ProTracker"""
    try:
        body = request.get_json()
        bracket = body.get("bracket", "Legend")

        search_query = (
            f"Search {SOURCES} for Dota 2 patch {CURRENT_PATCH} tier list. "
            f"Find the current S-tier and A-tier heroes for each position: "
            f"Carry (pos 1), Mid (pos 2), Offlane (pos 3), Soft Support (pos 4), Hard Support (pos 5). "
            f"Based on current win rates and pro pick rates for {bracket} bracket. "
            f"Include actual percentages where available."
        )

        analysis_prompt = f"""Based on the live DotaBuff/ProTracker data above, create a tier list for patch {CURRENT_PATCH}.
Bracket: {bracket}

JSON format (include real heroes based on live data, not guesses):
{{"carry":{{"S":["hero1","hero2"],"A":["hero3","hero4","hero5"]}},"mid":{{"S":["hero1","hero2"],"A":["hero3","hero4","hero5"]}},"offlane":{{"S":["hero1","hero2"],"A":["hero3","hero4","hero5"]}},"support":{{"S":["hero1","hero2"],"A":["hero3","hero4","hero5"]}},"hard_support":{{"S":["hero1","hero2"],"A":["hero3","hero4","hero5"]}},"source":"DotaBuff/ProTracker patch {CURRENT_PATCH}","updated":"{CURRENT_PATCH}"}}"""

        data = search_and_analyze(search_query, analysis_prompt, max_tokens=900)
        return jsonify(data)

    except Exception as e:
        logger.error(f"Tierlist error: {e}")
        return error_response(str(e))


@app.route("/itembuild", methods=["POST"])
def itembuild():
    """Live item build from high-MMR DotaBuff data"""
    try:
        body = request.get_json()
        hero = body.get("hero", "")
        if not hero:
            return error_response("Hero name required", 400)

        search_query = (
            f"Search {SOURCES} for best item build for {hero} in Dota 2 patch {CURRENT_PATCH}. "
            f"Find current starting items, early game items, core items, and luxury items "
            f"based on high-MMR player data and win rates."
        )

        analysis_prompt = f"""Based on live {SOURCES} data for {hero} patch {CURRENT_PATCH}:

JSON format:
{{"hero":"{hero}","starting":["item1","item2","item3"],"early":["item1","item2"],"core":["item1","item2","item3"],"luxury":["item1","item2"],"tip":"One sentence tip based on current meta data."}}"""

        data = search_and_analyze(search_query, analysis_prompt)
        return jsonify(data)

    except Exception as e:
        logger.error(f"Itembuild error: {e}")
        return error_response(str(e))


@app.route("/counter", methods=["POST"])
def counter():
    """Live counter picks from DotaBuff matchup data"""
    try:
        body = request.get_json()
        hero = body.get("hero", "")
        if not hero:
            return error_response("Hero name required", 400)

        search_query = (
            f"Search {SOURCES} for best counter picks against {hero} in Dota 2 patch {CURRENT_PATCH}. "
            f"Find heroes with highest win rate against {hero} based on current matchup statistics. "
            f"Include both hard counters and soft counters with win rate data."
        )

        analysis_prompt = f"""Based on live {SOURCES} matchup data for countering {hero} in patch {CURRENT_PATCH}:

JSON format (3 hard counters, 3 soft counters based on actual win rate data):
{{"hero":"{hero}","hard":[{{"name":"Hero","reason":"Why based on matchup stats."}}],"soft":[{{"name":"Hero","reason":"Why based on matchup stats."}}],"avoid":["hero1","hero2"],"tip":"One tip for playing against {hero} in current meta."}}"""

        data = search_and_analyze(search_query, analysis_prompt)
        return jsonify(data)

    except Exception as e:
        logger.error(f"Counter error: {e}")
        return error_response(str(e))


@app.route("/guide", methods=["POST"])
def guide():
    """Hero guide with current meta context"""
    try:
        body = request.get_json()
        hero = body.get("hero", "")
        if not hero:
            return error_response("Hero name required", 400)

        # Guide uses fast Haiku (less data-dependent)
        data = fast_analyze(
            f"""Dota 2 hero guide for {hero} in patch {CURRENT_PATCH}.
JSON: {{"hero":"{hero}","role":"primary role in current meta","playstyle":"2 sentences on how to play.","strengths":["s1","s2","s3"],"weaknesses":["w1","w2"],"tip":"one pro tip for patch {CURRENT_PATCH}."}}"""
        )
        return jsonify(data)

    except Exception as e:
        logger.error(f"Guide error: {e}")
        return error_response(str(e))


@app.route("/patch", methods=["POST"])
def patch():
    """Live patch summary from official sources"""
    try:
        search_query = (
            f"Search dota2.com and {SOURCES} for Dota 2 patch {CURRENT_PATCH} full patch notes. "
            f"Find all hero buffs, hero nerfs, item changes, and how the meta shifted. "
            f"Include specific numbers where available."
        )

        analysis_prompt = f"""Based on live patch {CURRENT_PATCH} data from official sources and {SOURCES}:

JSON format:
{{"patch":"{CURRENT_PATCH}","headline":"One sentence summary of patch theme.","buffed":["hero1","hero2","hero3","hero4"],"nerfed":["hero1","hero2","hero3","hero4"],"item_changes":"Key item changes in one sentence.","meta_shift":"How the meta changed in 2 sentences based on pro play data."}}"""

        data = search_and_analyze(search_query, analysis_prompt)
        return jsonify(data)

    except Exception as e:
        logger.error(f"Patch error: {e}")
        return error_response(str(e))


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 Dota 2 Oracle API starting on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)

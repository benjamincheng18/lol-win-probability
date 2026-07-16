import os
import json
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("RIOT_API_KEY")

REGION_PLATFORM = "kr"   # league-v4, summoner-v4 use platform routing
REGION_ROUTING = "asia"  # match-v5 uses regional routing
RAW_DIR = "data/raw"
MANIFEST_PATH = "data/processed/manifest.csv"


def get_seed_puuids(queue="RANKED_SOLO_5x5"):
    puuids = []
    for tier in ["challengerleagues", "grandmasterleagues"]:
        url = f"https://{REGION_PLATFORM}.api.riotgames.com/lol/league/v4/{tier}/by-queue/{queue}"
        data = safe_request(url)
        if data is None:
            print(f"Skipping {tier} — request failed")
            continue
        puuids.extend(entry["puuid"] for entry in data["entries"])
    return set(puuids)


def get_match_ids_for_puuid(puuid, count=20, queue=420):
    """Fetch recent ranked solo match IDs for a given puuid. Returns [] on failure."""
    url = f"https://{REGION_ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {"queue": queue, "count": count}
    result = safe_request(url, params=params)
    return result if result is not None else []


def safe_request(url, params=None, max_retries=5):
    """Rate-limit-aware GET request with exponential backoff on 429."""
    headers = {"X-Riot-Token": API_KEY}

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params)
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            time.sleep(2 ** attempt)  # backoff even on network errors
            continue

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            wait = int(response.headers.get("Retry-After", 2 ** attempt))
            print(f"Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
        elif response.status_code in (403, 401):
            print(f"Auth error {response.status_code} — check API key")
            return None  # don't retry, key is bad, retrying won't help
        elif response.status_code == 404:
            return None  # match/resource doesn't exist, don't retry
        else:
            print(f"Unexpected status {response.status_code}, retrying...")
            time.sleep(2 ** attempt)

    print(f"Max retries exceeded for {url}")
    return None

        


def fetch_and_cache_match(match_id):
    """Fetch match detail + timeline, cache both as raw JSON. Skip if already cached or on failure."""
    detail_path = f"{RAW_DIR}/{match_id}_detail.json"
    timeline_path = f"{RAW_DIR}/{match_id}_timeline.json"

    if os.path.exists(detail_path) and os.path.exists(timeline_path):
        return

    match_url = f"https://{REGION_ROUTING}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    match_detail = safe_request(match_url)
    if match_detail is None:
        print(f"Skipping {match_id} — match detail fetch failed")
        return

    timeline_url = f"https://{REGION_ROUTING}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    timeline = safe_request(timeline_url)
    if timeline is None:
        print(f"Skipping {match_id} — timeline fetch failed")
        return

    with open(detail_path, "w") as f:
        json.dump(match_detail, f)
    with open(timeline_path, "w") as f:
        json.dump(timeline, f)


def snowball_match_ids(seed_puuids, target_count=5000, matches_per_player=20):
    """
    Expand from seed players to target_count unique match IDs by:
    1. Pulling match IDs for each seed puuid
    2. Pulling match detail to discover new puuids (other 9 participants)
    3. Repeating with newly discovered puuids until target_count reached
    Returns: set of match IDs
    """
    # TODO: implement the snowball loop
    # Hint: use a queue/set of "puuids to process" and a set of "already processed puuids" to avoid re-crawling the same player
    # Hint: use a set (not list) for match_ids to naturally dedupe
    # Hint: this is the function where checkpointing matters most — consider saving progress every N matches
    pass


def build_manifest(match_ids):
    """
    Given a list of match IDs (already cached via fetch_and_cache_match),
    build a flat manifest dataframe: match_id, patch, queue_id, duration, winner, timestamp
    Save to MANIFEST_PATH, appending/checkpointing as it goes.
    """
    # TODO: for each match_id, load its cached detail JSON, extract the manifest fields
    # Hint: relevant fields live under detail["info"] — gameVersion (patch), queueId, gameDuration, and
    #        teams' "win" field under info["teams"] to get the winner
    pass


if __name__ == "__main__":
    seed_puuids = get_seed_puuids()
    match_ids = snowball_match_ids(seed_puuids, target_count=5000)
    for mid in match_ids:
        fetch_and_cache_match(mid)
        time.sleep(1.2)  # crude pacing under 100 req/2min; refine once safe_request has real backoff
    manifest = build_manifest(match_ids)
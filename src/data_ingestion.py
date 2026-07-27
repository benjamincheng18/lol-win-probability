import os
import json
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from collections import deque

load_dotenv()
API_KEY = os.getenv("RIOT_API_KEY")

REGION_PLATFORM = "kr"   # league-v4, summoner-v4 use platform routing
REGION_ROUTING = "asia"  # match-v5 uses regional routing
RAW_DIR = "data/raw"
MANIFEST_PATH = "data/processed/manifest.csv"
CHECKPOINT_PATH = "data/processed/match_ids_checkpoint.json"


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
    url = f"https://{REGION_ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    params = {"queue": queue, "count": count}
    result = safe_request(url, params=params)
    return result if result is not None else []


def safe_request(url, params=None, max_retries=5):
    headers = {"X-Riot-Token": API_KEY}

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            time.sleep(2 ** attempt)  # backoff even on network errors
            continue

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            wait = min(int(response.headers.get("Retry-After", 2 ** attempt)), 120)
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
    detail_path = f"{RAW_DIR}/{match_id}_detail.json"
    timeline_path = f"{RAW_DIR}/{match_id}_timeline.json"

    if os.path.exists(detail_path) and os.path.exists(timeline_path):
        with open(detail_path) as f:
            return json.load(f)

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
    return match_detail    


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return set(), set()
    with open(CHECKPOINT_PATH) as f:
        state = json.load(f)
    return set(state["match_ids"]), set(state["processed_puuids"])


def snowball_match_ids(seed_puuids, target_count=5000, matches_per_player=20,
                       checkpoint_every=100, match_ids=None, processed_puuids=None):
    to_process = deque(seed_puuids)
    processed_puuids = processed_puuids or set()
    match_ids = match_ids or set()

    while len(match_ids) < target_count and to_process:
        puuid = to_process.popleft()
        if puuid in processed_puuids:
            continue
        processed_puuids.add(puuid)

        new_ids = get_match_ids_for_puuid(puuid, count=matches_per_player)

        for match_id in new_ids:
            if match_id in match_ids:
                continue
            match_ids.add(match_id)

            detail = fetch_and_cache_match(match_id)
            if detail is None:
                continue  # fetch failed, can't discover new puuids from it

            for participant_puuid in detail["metadata"]["participants"]:
                if participant_puuid not in processed_puuids:
                    to_process.append(participant_puuid)

            if len(match_ids) % checkpoint_every == 0:
                with open(CHECKPOINT_PATH, "w") as f:
                    json.dump({
                        "match_ids": list(match_ids),
                        "processed_puuids": list(processed_puuids),
                    }, f)
                print(f"Checkpoint: {len(match_ids)} matches collected")

            if len(match_ids) >= target_count:
                break
    return match_ids


def build_manifest(match_ids):
    rows = []

    for match_id in match_ids:
        detail_path = f"{RAW_DIR}/{match_id}_detail.json"
        try:
            with open(detail_path) as f:
                detail = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Skipping {match_id} — {type(e).__name__}")
            continue

        info = detail["info"]
        winner = next((t["teamId"] for t in info["teams"] if t["win"]), None)

        rows.append({
            "match_id": match_id,
            "patch": info["gameVersion"],
            "queue_id": info["queueId"],
            "duration": info["gameDuration"],
            "winner_team": winner,
        })

    manifest = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"Manifest built: {len(manifest)} matches")
    return manifest


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)

    match_ids, processed_puuids = load_checkpoint()
    if match_ids:
        print(f"Resuming from checkpoint: {len(match_ids)} matches, {len(processed_puuids)} puuids processed")

    seeds = get_seed_puuids()
    match_ids = snowball_match_ids(
        seeds,
        target_count=5000,
        match_ids=match_ids,
        processed_puuids=processed_puuids,
    )

    build_manifest(match_ids)
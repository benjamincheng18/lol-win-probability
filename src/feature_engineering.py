import os
import json
import pandas as pd

RAW_DIR = "data/raw"
MANIFEST_PATH = "data/processed/manifest.csv"
FEATURES_PATH = "data/processed/features.csv"

TEAM_100_IDS = [str(i) for i in range(1, 6)]    # participants 1-5
TEAM_200_IDS = [str(i) for i in range(6, 11)]   # participants 6-10


def compute_frame_diffs(participant_frames):
    def team_sum(ids, key):
        return sum(participant_frames[i][key] for i in ids)

    return {
        "gold_diff": team_sum(TEAM_100_IDS, "totalGold") - team_sum(TEAM_200_IDS, "totalGold"),
        "xp_diff": team_sum(TEAM_100_IDS, "xp") - team_sum(TEAM_200_IDS, "xp"),
        "level_diff": team_sum(TEAM_100_IDS, "level") - team_sum(TEAM_200_IDS, "level"),
        "cs_diff": (
            team_sum(TEAM_100_IDS, "minionsKilled") + team_sum(TEAM_100_IDS, "jungleMinionsKilled")
            - team_sum(TEAM_200_IDS, "minionsKilled") - team_sum(TEAM_200_IDS, "jungleMinionsKilled")
        ),
    }


MONSTER_TO_COUNTER = {
    "DRAGON": "dragon_diff",
    "BARON_NASHOR": "baron_diff",
    "RIFTHERALD": "herald_diff",
    "HORDE": "grub_diff",
}


def update_objective_counts(events, counts):
    """
    Given one frame's events list and the running counts dict, return updated counts.
    Tracks towers, dragons, barons, heralds, grubs as team100-minus-team200 running diffs.
    """
    for e in events:
        if e["type"] == "ELITE_MONSTER_KILL":
            counter = MONSTER_TO_COUNTER.get(e["monsterType"])
            if counter is None:
                continue                      # unknown monster type, skip
            counts[counter] += 1 if e["killerTeamId"] == 100 else -1

        elif e["type"] == "BUILDING_KILL":
            if e.get("buildingType") != "TOWER_BUILDING":
                continue                      # skip inhibitors
            counts["tower_diff"] += -1 if e["teamId"] == 100 else 1
    return counts


def process_match(match_id, winner_team):
    """Process one match's timeline into a list of per-frame feature rows."""
    timeline_path = f"{RAW_DIR}/{match_id}_timeline.json"
    try:
        with open(timeline_path) as f:
            timeline = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Skipping {match_id} — {type(e).__name__}")
        return []

    counts = {
        "tower_diff": 0,
        "dragon_diff": 0,
        "baron_diff": 0,
        "herald_diff": 0,
        "grub_diff": 0,
    }

    rows = []
    won = 1 if winner_team == 100 else 0

    for frame in timeline["info"]["frames"]:
        update_objective_counts(frame["events"], counts)
        diffs = compute_frame_diffs(frame["participantFrames"])

        row = {
            "match_id": match_id,
            "minute": frame["timestamp"] // 60000,
            **diffs,
            **counts,
            "won": won,
        }
        rows.append(row)
    return rows


def build_feature_table():
    manifest = pd.read_csv(MANIFEST_PATH)

    all_rows = []
    for row in manifest.itertuples(index=False):
        all_rows.extend(process_match(row.match_id, row.winner_team))

    features = pd.DataFrame(all_rows)
    features = features.sort_values(["match_id", "minute"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    features.to_csv(FEATURES_PATH, index=False)
    print(f"Feature table built: {len(features)} rows from {features['match_id'].nunique()} matches")
    return features


if __name__ == "__main__":
    build_feature_table()
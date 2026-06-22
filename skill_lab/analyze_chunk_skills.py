import json
from collections import Counter


CHUNK_SIZE = 100


def analyze_episode(skill_ids, chunk_size=100):
    total = 0
    pure = 0
    cross = 0
    patterns = Counter()

    max_start = len(skill_ids) - chunk_size + 1
    if max_start < 1:
        return {"total": 0, "pure": 0, "cross": 0, "patterns": {}}

    for t in range(max_start):
        window = skill_ids[t : t + chunk_size]
        uniq = sorted(set(window))
        total += 1

        if len(uniq) == 1:
            pure += 1
        else:
            cross += 1

        patterns[tuple(uniq)] += 1

    return {
        "total": total,
        "pure": pure,
        "cross": cross,
        "patterns": dict(patterns),
    }


if __name__ == "__main__":
    path = "/home/zjq/lerobot/skill_lab/episode_skill_ids_grab_cube.json"

    with open(path, "r") as f:
        data = json.load(f)

    all_total = 0
    all_pure = 0
    all_cross = 0
    all_patterns = Counter()

    for episode_key, skill_ids in data.items():
        result = analyze_episode(skill_ids, chunk_size=CHUNK_SIZE)

        print(f"\n{episode_key}")
        print(" total chunks:", result["total"])
        print(" pure chunks:", result["pure"])
        print(" cross chunks:", result["cross"])
        print(" patterns:", result["patterns"])

        all_total += result["total"]
        all_pure += result["pure"]
        all_cross += result["cross"]
        all_patterns.update(result["patterns"])

    print("\n=== overall ===")
    print("total chunks:", all_total)
    print("pure chunks:", all_pure)
    print("cross chunks:", all_cross)
    if all_total > 0:
        print("pure ratio:", all_pure / all_total)
        print("cross ratio:", all_cross / all_total)
    print("pattern counts:", dict(all_patterns))
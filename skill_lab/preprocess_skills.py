import json
from skills import SKILL_VOCAB


def build_skill_ids(traj_len, segments):
    skill_ids = [-1] * traj_len

    for seg in segments:
        sid = SKILL_VOCAB[seg["skill"]]
        start = seg["start"]
        end = seg["end"]

        for t in range(start, end + 1):
            if t < 0 or t >= traj_len:
                raise ValueError(f"frame {t} out of range (traj_len={traj_len})")
            if skill_ids[t] != -1:
                raise ValueError(f"frame {t} assigned twice")
            skill_ids[t] = sid

    if any(s == -1 for s in skill_ids):
        missing = [i for i, s in enumerate(skill_ids) if s == -1][:10]
        raise ValueError(f"unlabeled frames exist: {missing}")

    return skill_ids


def build_all_episode_skill_ids(episode_lengths, segments_json_path, output_json_path):
    with open(segments_json_path, "r") as f:
        segments_db = json.load(f)

    output = {}
    for episode_key, segments in segments_db.items():
        if episode_key not in episode_lengths:
            raise KeyError(f"{episode_key} not found in episode_lengths")

        traj_len = episode_lengths[episode_key]
        output[episode_key] = build_skill_ids(traj_len, segments)

    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    episode_lengths = {
        "episode_0": 481,
        "episode_1": 445,
        "episode_2": 432,
        "episode_3": 500,
        "episode_4": 567,
        "episode_5": 379,
        "episode_6": 407,
        "episode_7": 443,
        "episode_8": 380,
        "episode_9": 479,
    }

    segments_json_path = "/home/zjq/lerobot/skill_lab/segments_grab_cube.json"
    output_json_path = "/home/zjq/lerobot/skill_lab/episode_skill_ids_grab_cube.json"

    build_all_episode_skill_ids(
        episode_lengths=episode_lengths,
        segments_json_path=segments_json_path,
        output_json_path=output_json_path,
    )

    print(f"Saved per-frame skill ids to: {output_json_path}")
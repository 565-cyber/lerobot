import pandas as pd


def get_episode_lengths():
    episodes_file = "/home/zjq/.cache/huggingface/lerobot/zjq/record-test3/meta/episodes/chunk-000/file-000.parquet"
    df = pd.read_parquet(episodes_file)

    episode_lengths = {}
    for _, row in df.iterrows():
        ep_idx = int(row["episode_index"])
        length = int(row["length"])
        episode_lengths[f"episode_{ep_idx}"] = length

    return episode_lengths


if __name__ == "__main__":
    episode_lengths = get_episode_lengths()

    for k in list(episode_lengths.keys())[:10]:
        print(k, episode_lengths[k])
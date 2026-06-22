from lerobot.datasets.lerobot_dataset import LeRobotDataset

def main():
    dataset = LeRobotDataset(
        repo_id="zjq/record-test3",
        root="/home/zjq/.cache/huggingface/lerobot/zjq/record-test3",
    )

    print("dataset len =", len(dataset))

    sample = dataset[0]
    print("sample keys =", sample.keys())
    print("episode_index =", sample["episode_index"])
    print("frame_index =", sample["frame_index"])
    print("action shape =", sample["action"].shape)
    print("state shape =", sample["observation.state"].shape)
    print("front image shape =", sample["observation.images.front"].shape)
    print("up image shape =", sample["observation.images.up"].shape)

if __name__ == "__main__":
    main()
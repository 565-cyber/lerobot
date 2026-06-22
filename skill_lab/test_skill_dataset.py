from lerobot.datasets.lerobot_dataset import LeRobotDataset
from skill_dataset_wrapper import SkillAnnotatedDataset

SKILL_LABEL_PATH = "/home/zjq/lerobot/skill_lab/episode_skill_ids_grab_cube.json"

def main():
    base_dataset = LeRobotDataset(
        repo_id="zjq/record-test3",
        root="/home/zjq/.cache/huggingface/lerobot/zjq/record-test3",
    )

    wrapped_dataset = SkillAnnotatedDataset(
        base_dataset=base_dataset,
        skill_label_path=SKILL_LABEL_PATH,
        chunk_size=100,
    )

    print("raw dataset len =", len(base_dataset))
    print("wrapped dataset len =", len(wrapped_dataset))

    sample = wrapped_dataset[0]
    print("sample keys =", sample.keys())
    print("skill_id =", sample["skill_id"])
    print("episode_index =", sample["episode_index"])
    print("frame_index =", sample["frame_index"])
    print("state shape =", sample["observation.state"].shape)
    print("action shape =", sample["action"].shape)
    print("front image shape =", sample["observation.images.front"].shape)
    print("up image shape =", sample["observation.images.up"].shape)

if __name__ == "__main__":
    main()
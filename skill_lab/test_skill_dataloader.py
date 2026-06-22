from torch.utils.data import DataLoader
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

    dataloader = DataLoader(
        wrapped_dataset,
        batch_size=8,
        shuffle=True,
    )

    batch = next(iter(dataloader))

    print("batch keys =", batch.keys())
    print("skill_id shape =", batch["skill_id"].shape)
    print("skill_id =", batch["skill_id"])

    print("state shape =", batch["observation.state"].shape)
    print("action shape =", batch["action"].shape)
    print("front image shape =", batch["observation.images.front"].shape)
    print("up image shape =", batch["observation.images.up"].shape)

    print("episode_index shape =", batch["episode_index"].shape)
    print("frame_index shape =", batch["frame_index"].shape)


if __name__ == "__main__":
    main()
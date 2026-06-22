import json


class SkillAnnotatedDataset:
    def __init__(self, base_dataset, skill_label_path, chunk_size=100):
        self.base_dataset = base_dataset
        self.chunk_size = chunk_size

        with open(skill_label_path, "r") as f:
            self.skill_label_db = json.load(f)

        self.valid_indices = []
        self.valid_skill_ids = []

        for i in range(len(self.base_dataset)):
            sample = self.base_dataset[i]

            episode_index = int(sample["episode_index"])
            frame_index = int(sample["frame_index"])
            episode_key = f"episode_{episode_index}"

            if episode_key not in self.skill_label_db:
                continue

            skill_ids = self.skill_label_db[episode_key]

            if self.is_single_skill_chunk(skill_ids, frame_index, self.chunk_size):
                self.valid_indices.append(i)
                self.valid_skill_ids.append(skill_ids[frame_index])

    def is_single_skill_chunk(self, skill_ids, start, chunk_size):
        if start + chunk_size > len(skill_ids):
            return False

        sid = skill_ids[start]
        return all(x == sid for x in skill_ids[start : start + chunk_size])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        base_idx = self.valid_indices[idx]
        sample = self.base_dataset[base_idx]

        sample = dict(sample)
        sample["skill_id"] = self.valid_skill_ids[idx]

        return sample
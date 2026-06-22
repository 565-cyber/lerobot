SKILL_VOCAB = {
    "approach_cube": 0,
    "grasp_cube": 1,
    "lift_cube": 2,
}

ID2SKILL = {v: k for k, v in SKILL_VOCAB.items()}
NUM_SKILLS = len(SKILL_VOCAB)
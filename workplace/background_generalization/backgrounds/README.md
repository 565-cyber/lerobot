# Backgrounds

Drop background images (`.jpg` / `.png` / `.bmp` / `.webp`) here. They are sampled
deterministically per episode (`episode_index % len(images)`) so every frame of an
episode shares one background (temporal consistency, no flicker).

If the directory is empty (or `--background-root` is not given),
`BackgroundAugmentedDataset` falls back to a deterministic solid color per episode,
so the pipeline is runnable immediately.

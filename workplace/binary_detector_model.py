"""
Binary detector: ResNet18 with single-output head for binary classification.

Each detector answers one question about the scene:
  - Is yellow block on the table?
  - Is blue block on the table?
  - Is the trash can lid open?
"""
import torch
import torch.nn as nn
from torchvision import transforms, models


class BinaryDetector:
    def __init__(
        self,
        ckpt_path: str,
        device: str = None,
        camera_key: str = "observation.images.up",
        vote_window: int = 5,
        confidence_threshold: float = 0.5,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.camera_key = camera_key
        self.vote_window = vote_window
        self.confidence_threshold = confidence_threshold
        self.history: list[bool] = []

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        self.label_name = ckpt.get("label_name", "unknown")

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(device)
        self.model.eval()

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _prepare_image(self, image):
        """
        Support common formats:
        1. torch.Tensor [C, H, W]
        2. torch.Tensor [H, W, C]
        3. torch.Tensor [B, C, H, W]
        4. torch.Tensor [B, H, W, C]
        """
        if not torch.is_tensor(image):
            image = torch.as_tensor(image)

        if image.ndim == 4:
            image = image[0]

        if image.ndim != 3:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        # HWC -> CHW
        if image.shape[-1] == 3:
            image = image.permute(2, 0, 1)

        image = image.float()

        # 0~255 -> 0~1
        if image.max() > 2:
            image = image / 255.0

        x = self.tf(image)
        x = x.unsqueeze(0).to(self.device)

        return x

    @torch.no_grad()
    def predict_once(self, obs: dict) -> tuple:
        """Return (bool, float) — binary prediction and sigmoid confidence."""
        image = obs[self.camera_key]
        x = self._prepare_image(image)

        logit = self.model(x)
        prob = torch.sigmoid(logit).item()

        pred = prob >= self.confidence_threshold
        return pred, prob

    @torch.no_grad()
    def __call__(self, obs: dict) -> tuple:
        """Return (bool, float) after temporal voting."""
        pred, confidence = self.predict_once(obs)

        self.history.append(pred)
        if len(self.history) > self.vote_window:
            self.history.pop(0)

        # Majority vote
        true_count = sum(self.history)
        false_count = len(self.history) - true_count

        if true_count >= false_count:
            voted_pred = True
            voted_confidence = true_count / len(self.history)
        else:
            voted_pred = False
            voted_confidence = false_count / len(self.history)

        return voted_pred, voted_confidence

    def history_vote(self) -> tuple:
        """Return (bool, float) — voted result from current history without adding new frame.
        If history is empty, returns (False, 0.0)."""
        if not self.history:
            return False, 0.0

        true_count = sum(self.history)
        false_count = len(self.history) - true_count

        if true_count >= false_count:
            return True, true_count / len(self.history)
        else:
            return False, false_count / len(self.history)

    def reset(self):
        self.history.clear()

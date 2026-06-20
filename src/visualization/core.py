import sys
import os
import random
import glob as glob_module
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM, EigenCAM, LayerCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

CLASS_NAMES = ['Anger', 'Disgust', 'Fear', 'Sad', 'Happy', 'Surprise', 'Neutral', 'Contempt']

CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "hirescam": HiResCAM,
    "eigencam": EigenCAM,
    "layercam": LayerCAM,
    "scorecam": ScoreCAM,
}


class _CamEvalWrapper(nn.Module):
    """mmcls BaseBackbone.train() does not return self, so model.eval() returns None.
    pytorch_grad_cam stores self.model = model.eval(); wrap IRSE-based models so .eval() works."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)


def strip_module_prefix(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k.removeprefix("module.")] = v
    return new_sd


def load_model(network, weights_path, num_classes, uses_ef_modules=True, device="cpu"):
    if network == "EF":
        from networks.EfficientFace.models.EfficientFace import efficient_face

        model = efficient_face()
        model.fc = nn.Linear(1024, num_classes)

        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        state_dict = strip_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=False)

        target_layers = [model.conv5[0]]
        input_size = 224

    elif network == "MLDG":
        from networks.LDG.models.ModifiedLDG import load_base_mLDG

        inner = load_base_mLDG(
            checkpoint_path=weights_path,
            uses_ef_modules=uses_ef_modules,
            num_classes=num_classes,
        )
        model = _CamEvalWrapper(inner)
        target_layers = [model.inner.body[3]]
        input_size = 112

    else:
        raise ValueError(f"Unknown network: {network!r}. Expected 'EF' or 'MLDG'.")

    model.to(device).eval()
    return model, target_layers, input_size


def get_transform(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_image(image_path, input_size):
    img = Image.open(image_path).convert("RGB")
    transform = get_transform(input_size)
    input_tensor = transform(img).unsqueeze(0)

    rgb_image = img.resize((input_size, input_size))
    rgb_image = np.array(rgb_image, dtype=np.float32) / 255.0

    return input_tensor, rgb_image


def collect_images(input_paths=None, input_dir=None, sample=False,
                   dataset_path=None, sample_split="test", sample_n=1,
                   sample_classes=7, sample_seed=None):
    images = []

    if input_paths:
        images = [(p, None) for p in input_paths]

    elif input_dir:
        exts = ("*.jpg", "*.jpeg", "*.png")
        files = []
        for ext in exts:
            files.extend(glob_module.glob(os.path.join(input_dir, ext)))
        files.sort()
        images = [(f, None) for f in files[:16]]

    elif sample:
        split_dir = os.path.join(dataset_path, sample_split)
        if sample_seed is not None:
            random.seed(sample_seed)

        all_files = []
        for cls_idx in range(sample_classes):
            cls_dir = os.path.join(split_dir, str(cls_idx))
            if not os.path.isdir(cls_dir):
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for fp in glob_module.glob(os.path.join(cls_dir, ext)):
                    all_files.append((fp, cls_idx))

        sample_n = max(1, min(sample_n, 16))
        if len(all_files) > sample_n:
            all_files = random.sample(all_files, sample_n)
        images = all_files

    return images[:16]


def generate_cam(model, target_layers, input_tensor, rgb_image,
                 cam_method="gradcam", target_class=None):
    cam_cls = CAM_METHODS[cam_method]

    with torch.no_grad():
        if target_class is None:
            logits = model(input_tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)
            confidence, predicted_class = probs.max(dim=1)
            predicted_class = predicted_class.item()
            confidence = confidence.item()
            targets = [ClassifierOutputTarget(predicted_class)]
        else:
            targets = [ClassifierOutputTarget(target_class)]
            logits = model(input_tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)
            predicted_class = probs.argmax(dim=1).item()
            confidence = probs[0, predicted_class].item()

    with cam_cls(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    cam_image = show_cam_on_image(rgb_image, grayscale_cam[0], use_rgb=True)
    return cam_image, predicted_class, confidence


def run_cam_pipeline(network, weights_path, num_classes, uses_ef_modules=True,
                     input_paths=None, input_dir=None, sample=False,
                     dataset_path=None, sample_split="test", sample_n=1,
                     sample_classes=7, sample_seed=None, cam_method="gradcam",
                     target_class=None, output_dir="./cam_output", device="cpu"):

    model, target_layers, input_size = load_model(
        network, weights_path, num_classes,
        uses_ef_modules=uses_ef_modules, device=device,
    )

    images = collect_images(
        input_paths=input_paths, input_dir=input_dir, sample=sample,
        dataset_path=dataset_path, sample_split=sample_split,
        sample_n=sample_n, sample_classes=sample_classes,
        sample_seed=sample_seed,
    )

    os.makedirs(output_dir, exist_ok=True)

    results = []
    for image_path, gt_label in images:
        input_tensor, rgb_image = load_image(image_path, input_size)
        input_tensor = input_tensor.to(device)

        cam_image, predicted_class, confidence = generate_cam(
            model, target_layers, input_tensor, rgb_image,
            cam_method=cam_method, target_class=target_class,
        )

        stem = Path(image_path).stem
        out_name = f"{stem}_cam.png"
        out_path = os.path.join(output_dir, out_name)
        Image.fromarray(cam_image).save(out_path)

        results.append({
            "input_path": image_path,
            "output_path": out_path,
            "predicted_class": predicted_class,
            "predicted_name": CLASS_NAMES[predicted_class],
            "confidence": confidence,
            "gt_label": gt_label,
            "gt_name": CLASS_NAMES[gt_label] if gt_label is not None else None,
            "cam_image": cam_image,
        })

    return results

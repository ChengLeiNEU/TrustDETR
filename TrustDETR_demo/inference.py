import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

DEMO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_ROOT.parent / "TrustDETR"


def _ensure_project_on_path():
    root = os.environ.get("TRUSTDETR_ROOT", str(PROJECT_ROOT))
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_val_transforms(config_path):
    _ensure_project_on_path()
    from src.core import YAMLConfig

    cfg = YAMLConfig(config_path, resume="", use_amp=False)
    dataset = cfg.val_dataloader.dataset
    return dataset.transforms, cfg


def _read_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_pair(ir_path, rgb_path, transforms):
    ir = Image.open(ir_path).convert("RGB")
    rgb = Image.open(rgb_path).convert("RGB")
    w, h = ir.size
    target = {
        "boxes": tv_tensors.BoundingBoxes(
            torch.zeros((0, 4)), format="xyxy", canvas_size=(h, w)
        ),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "orig_size": torch.as_tensor([int(w), int(h)]),
        "size": torch.as_tensor([int(w), int(h)]),
    }
    ir_t, rgb_t, target = transforms(ir, rgb, target)
    return ir_t, rgb_t, target["orig_size"]


def load_model(config_path, checkpoint_path, device="cuda"):
    _ensure_project_on_path()
    from src.core import YAMLConfig

    cfg = YAMLConfig(config_path, resume=checkpoint_path, use_amp=False)
    model = cfg.model.to(device)
    postprocessor = cfg.postprocessor.to(device) if hasattr(cfg, "postprocessor") else None
    model.eval()
    if postprocessor is not None:
        postprocessor.eval()
    return model, postprocessor, cfg


def predict(config_path, checkpoint_path, ir_path, rgb_path, score_thr=0.5, device="cuda"):
    transforms, cfg = _load_val_transforms(config_path)
    model, postprocessor, cfg = load_model(config_path, checkpoint_path, device)
    ir_t, rgb_t, orig_size = preprocess_pair(ir_path, rgb_path, transforms)
    ir_t = ir_t.unsqueeze(0).to(device)
    rgb_t = rgb_t.unsqueeze(0).to(device)
    orig_size = orig_size.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(ir_t, rgb_t)
        if postprocessor is not None:
            results = postprocessor(outputs, orig_size)[0]
        else:
            raise RuntimeError("Postprocessor is required for demo inference.")

    keep = results["scores"] >= score_thr
    boxes = results["boxes"][keep].cpu().numpy()
    scores = results["scores"][keep].cpu().numpy()
    labels = results["labels"][keep].cpu().numpy()
    return boxes, scores, labels, cfg


def draw_boxes(image_rgb, boxes, labels, scores, class_names, score_thr=0.5):
    img = image_rgb.copy()
    colors = [
        (0, 255, 0),
        (255, 128, 0),
        (0, 128, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (128, 0, 255),
    ]
    for box, label, score in zip(boxes, labels, scores):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = colors[int(label) % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        name = class_names[int(label)] if int(label) < len(class_names) else f"class_{int(label)}"
        text = f"{name} {score:.2f}"
        cv2.putText(img, text, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def class_names_from_config(cfg):
    if hasattr(cfg, "class_names"):
        return cfg.class_names
    n = getattr(cfg, "num_classes", 3)
    if n == 7:
        return ["ship", "car", "cyclist", "pedestrian", "bus", "drone", "plane"]
    if n == 1:
        return ["person"]
    return ["person", "rider", "crowd"][:n]


def run_demo_pair(config_path, checkpoint_path, ir_path, rgb_path, score_thr=0.5, device="cuda"):
    boxes, scores, labels, cfg = predict(
        config_path, checkpoint_path, ir_path, rgb_path, score_thr=score_thr, device=device
    )
    names = class_names_from_config(cfg)
    ir = _read_image(ir_path)
    rgb = _read_image(rgb_path)
    ir_vis = draw_boxes(ir, boxes, labels, scores, names, score_thr)
    rgb_vis = draw_boxes(rgb, boxes, labels, scores, names, score_thr)
    panel = np.hstack([ir_vis, rgb_vis])
    return Image.fromarray(panel), len(boxes)

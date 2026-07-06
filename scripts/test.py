import os
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import label
from transformers import AutoTokenizer
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete

from datasets.data_paths import load_organ_data
from datasets.text_utils import read_text_file
from utils.transforms import val_transforms
from utils.predict import predictor
from models.multimodal_model import MultiModalSegModel

TEST_ROOT = "./data_preprocessed/test"
CKPT_DIR = "./ckpt"
OUTPUT_FILE = "./ckpt/test_results.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_ORGANS = os.listdir(TEST_ROOT)

def get_single_value(x):
    if isinstance(x, (list, tuple)):
        return x[0]
    return x

def remove_small_components(
    mask,
    min_size=100,
    keep_ratio=1/100,
):
    binary_mask = mask.astype(np.uint8)

    labeled_mask, num_features = label(binary_mask)

    if num_features == 0:
        return binary_mask

    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0

    largest_size = sizes.max()

    cleaned_mask = np.zeros_like(binary_mask)

    for component_id in range(1, num_features + 1):
        component_size = sizes[component_id]

        keep_component = (
            component_size >= min_size and
            component_size >= largest_size * keep_ratio
        )

        if keep_component:
            cleaned_mask[labeled_mask == component_id] = 1

    return cleaned_mask.astype(np.uint8)

def main(OUTPUT_FILE):
    # ==== Build test dataset with organ field ====
    test_data_dicts = []
    for organ in TEST_ORGANS:
        organ_dir = os.path.join(TEST_ROOT, organ)
        img_paths, lbl_paths, txt_paths = load_organ_data(organ_dir)

        for img_path, lbl_path, txt_path in zip(img_paths, lbl_paths, txt_paths):
            test_data_dicts.append({
                "image": img_path,
                "label": lbl_path,
                "text": read_text_file(txt_path),
                "organ": organ,
            })

    test_ds = Dataset(data=test_data_dicts, transform=val_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=6,
    )

    # ==== Model ====
    model = MultiModalSegModel().to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

    ckpt_path = f"{CKPT_DIR}/best_checkpoint.pth"
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ==== Metrics ====
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    post_pred = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5)
    ])
    post_label = AsDiscrete(threshold=0.5)

    dice_by_organ = defaultdict(list)

    with torch.no_grad():
        for test_data in tqdm(test_loader, desc="Testing process"):
            test_inputs = test_data["image"].to(DEVICE)
            test_labels = test_data["label"].to(DEVICE)
            test_texts = test_data["text"]
            organ_name = get_single_value(test_data["organ"])

            prompt_inputs = tokenizer(
                test_texts,
                padding=True,
                truncation=True,
                max_length=48,
                return_tensors="pt"
            ).to(DEVICE)

            test_outputs = sliding_window_inference(
                inputs=test_inputs,
                roi_size=(96, 96, 96),
                sw_batch_size=4,
                predictor=lambda img_patch: predictor(model, img_patch, prompt_inputs),
            )

            # Post-process per sample
            test_outputs = [post_pred(i) for i in decollate_batch(test_outputs)]
            test_labels = [post_label(i) for i in decollate_batch(test_labels)]

            # Post processing
            processed_outputs = []

            for pred in test_outputs:

                pred_np = pred.squeeze(0).cpu().numpy()

                pred_np = remove_small_components(
                    pred_np,
                    min_size=250,
                    keep_ratio=1/120,
                )

                pred_tensor = torch.tensor(pred_np,device=pred.device).unsqueeze(0)

                processed_outputs.append(pred_tensor)

            test_outputs = processed_outputs

            # Dice per case
            dice_metric(y_pred=test_outputs, y=test_labels)
            dice_value = dice_metric.aggregate().item()
            dice_metric.reset()
            dice_by_organ[organ_name].append(dice_value)

    # ==== Overall mean metrics ====
    all_dice_vals = [v for vals in dice_by_organ.values() for v in vals]
    overall_dice = float(np.mean(all_dice_vals)) if len(all_dice_vals) > 0 else float("nan")

    lines = []

    lines.append(f"\nDice Score: {overall_dice:.4f}")

    # ==== Per-organ report ====
    lines.append("\n=== Per-organ Dice ===")
    for organ in TEST_ORGANS:
        vals = dice_by_organ.get(organ, [])
        if len(vals) > 0:
            lines.append(f"{organ}: {np.mean(vals):.4f} ({len(vals)} cases)")
        else:
            lines.append(f"{organ}: no cases")


    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main(OUTPUT_FILE)
import os
import torch
import nibabel as nib
import numpy as np
from transformers import AutoTokenizer
from scipy.ndimage import label
from .transforms import inference_transforms, invert_transforms
from models.multimodal_model import MultiModalSegModel
from monai.inferers import sliding_window_inference

def predictor(model, img_patch, txt_inputs):
    return model(img_patch, txt_inputs)

def post_process_mask(
    mask,
    keep_top_k=None,
    min_size=100,
    keep_ratio=0.1,
):

    binary_mask = (mask > 0).astype(np.uint8)

    labeled_mask, num_features = label(binary_mask)

    print(f"Number of connected components: {num_features}")

    if num_features == 0:
        return binary_mask

    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0

    component_ids = np.arange(1, num_features + 1)

    sorted_indices = np.argsort(sizes[1:])[::-1]
    sorted_component_ids = component_ids[sorted_indices]

    largest_size = sizes[sorted_component_ids[0]]

    print("Component voxel counts (largest -> smallest):")

    for rank, comp_id in enumerate(sorted_component_ids, start=1):
        print(f"{rank:02d}. {sizes[comp_id]} voxels")

    keep_components = []

    for comp_id in sorted_component_ids:

        comp_size = sizes[comp_id]

        if comp_size < min_size:
            continue

        if comp_size < largest_size * keep_ratio:
            continue

        keep_components.append(comp_id)

    if keep_top_k is not None:
        keep_components = keep_components[:keep_top_k]

    print(f"Kept {len(keep_components)} component(s).")

    cleaned_mask = np.isin(
        labeled_mask,
        keep_components
    )

    return cleaned_mask.astype(np.uint8)

def predict(
    path_to_test_image: str,
    text: str,
    path_to_best_checkpoint: str,
    path_to_save_mask: str,
    device: torch.device | None = None,
    roi_size=(96, 96, 96),
    sw_batch_size=4,
    overlap=0.25,
    threshold=0.5,
    num_organs_to_keep=None,

    keep_top_k=None,
    min_size=100,
    keep_ratio=1/120,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiModalSegModel().to(device)
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

    data_of_test = {
        "image": path_to_test_image,
        "text": text,
    }

    data = inference_transforms(data_of_test)

    image = data["image"].unsqueeze(0).to(device)
    prompt_inputs = tokenizer(
        data["text"],
        padding=True,
        truncation=True,
        max_length=48,
        return_tensors="pt",
    ).to(device)

    checkpoint = torch.load(path_to_best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        output_seg = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=lambda img_patch: predictor(model, img_patch, prompt_inputs),
            overlap=overlap,
        )

    output_final = (torch.sigmoid(output_seg) > threshold).float()

    pred_tensor = output_final.squeeze(0).cpu()
    data["pred"] = pred_tensor

    result = invert_transforms(data)
    final_segmentation = result["pred"]

    mask = final_segmentation.squeeze().cpu().numpy()

    mask = post_process_mask(
        mask = mask,
        keep_top_k=keep_top_k,
        min_size=min_size,
        keep_ratio=keep_ratio,
    )

    affine = result["image"].meta["affine"]
    new_img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    nib.save(new_img, path_to_save_mask)

    return mask

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_image_dir = "./img_to_pred"
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)

    test_image_filename = os.listdir(test_image_dir)[0]
    # path_to_test_image = f"{test_image_dir}/{test_image_filename}"
    path_to_test_image = "data/test/Bladder/Image/Image007.nii.gz"

    text = "Locate the Bladder in the CT image."
    # path_to_best_checkpoint = "./ckpt_v0/best_checkpoint.pth"
    path_to_best_checkpoint = "./ckpt_1.2/best_checkpoint.pth"
    path_to_save_mask = f"{output_dir}/predicted_mask.nii.gz"
    
    mask = predict(
        path_to_test_image=path_to_test_image,
        text=text,
        path_to_best_checkpoint=path_to_best_checkpoint,
        path_to_save_mask=path_to_save_mask,
        device = device,
        threshold=0.5,
    
        # keep_top_k=1,
        min_size=250,
        keep_ratio=1/120,
    )
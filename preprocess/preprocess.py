import os
import shutil
from monai.transforms import (
    Compose, LoadImaged, Orientationd, EnsureChannelFirstd,
    Spacingd, ScaleIntensityRanged, SaveImaged
)
from monai.data import Dataset, DataLoader, decollate_batch
from tqdm import tqdm
from datasets.data_paths import load_organ_data

# ROOT_RAW = "./data/train"
# ROOT_PRE = "./data_preprocessed_1.2/train"
# ORGANS = os.listdir(ROOT_RAW)

# ROOT_RAW = "./data/val"
# ROOT_PRE = "./data_preprocessed_1.2/val"
# ORGANS = os.listdir(ROOT_RAW)

ROOT_RAW = "./data/test"
ROOT_PRE = "./data_preprocessed_1.2/test"
ORGANS = os.listdir(ROOT_RAW)

# organ_spacing = {
#     "Seminal_vesicles": (1.2, 1.2, 1.2),
#     "Adrenal_Left": (1.2, 1.2, 1.2),
#     "Adrenal_Right": (1.2, 1.2, 1.2),
#     "Prostate": (1.2, 1.2, 1.2),
#     "Gallbladder": (1.2, 1.2, 1.2),
#     "Esophagus": (1.2, 1.2, 1.2),
#     "Duodenum": (1.2, 1.2, 1.2),
#     "Rectum": (1.2, 1.2, 1.2),
#     "Pancreas": (1.2, 1.2, 1.2),

#     "Kidney_Left": (1.2, 1.2, 1.2),
#     "Kidney_Right": (1.2, 1.2, 1.2),
#     "Spleen": (1.2, 1.2, 1.2),
#     "Bladder": (1.2, 1.2, 1.2),
#     "Stomach": (1.2, 1.2, 1.2),
#     "Colon": (1.2, 1.2, 1.2),
#     "Intestine": (1.2, 1.2, 1.2),

#     "Liver": (1.2, 1.2, 1.2),
#     "Head_of_femur_Left": (1.2, 1.2, 1.2),
#     "Head_of_femur_Right": (1.2, 1.2, 1.2),
# }

# DEFAULT_SPACING = (1.2, 1.2, 1.2)

def build_pre_transforms(pixdim_value):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=pixdim_value,
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
    ])

for organ in ORGANS:
    print(f"--- Processing: {organ} ---")

    pixdim_value = (1.2, 1.2, 1.2)

    output_img_dir = os.path.join(ROOT_PRE, organ, "Image")
    output_lbl_dir = os.path.join(ROOT_PRE, organ, "Label")
    output_txt_dir = os.path.join(ROOT_PRE, organ, "Text")

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)

    img_paths, label_paths, txt_paths = load_organ_data(os.path.join(ROOT_RAW, organ))

    print("Copying text files...")
    for txt in txt_paths:
        file_name = os.path.basename(txt)
        shutil.copy(txt, os.path.join(output_txt_dir, file_name))
    print("Text files copied.")

    data_dicts = [{"image": i, "label": l} for i, l in zip(img_paths, label_paths)]
    
    pre_transforms = build_pre_transforms(pixdim_value)

    ds = Dataset(data=data_dicts, transform=pre_transforms)
    loader = DataLoader(ds, batch_size=1, num_workers=4)

    img_saver = SaveImaged(
        keys=["image"], 
        output_dir=output_img_dir, 
        output_postfix="", 
        output_ext=".nii.gz", 
        resample=False, 
        separate_folder=False
    )
    lbl_saver = SaveImaged(
        keys=["label"], 
        output_dir=output_lbl_dir, 
        output_postfix="", 
        output_ext=".nii.gz", 
        resample=False, 
        separate_folder=False
    )

    for batch in tqdm(loader, desc="Processing Images"):
        batch_list = decollate_batch(batch)
        for item in batch_list:
            img_saver(item)
            lbl_saver(item)

print("Done !")
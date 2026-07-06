import os

def load_organ_data(base_folder):
    img_folder = f"{base_folder}/Image"
    label_folder = f"{base_folder}/Label"
    txt_folder = f"{base_folder}/Text"

    img_paths = sorted([f"{img_folder}/{p}" for p in os.listdir(img_folder)])
    label_paths = sorted([f"{label_folder}/{p}" for p in os.listdir(label_folder)])
    txt_paths = sorted([f"{txt_folder}/{p}" for p in os.listdir(txt_folder)])

    return img_paths, label_paths, txt_paths


def build_dataset_paths(root_dir, organs):
    all_imgs, all_labels, all_txts = [], [], []

    for organ in organs:
        base = f"{root_dir}/{organ}"
        imgs, labels, txts = load_organ_data(base)

        all_imgs += imgs
        all_labels += labels
        all_txts += txts

    return all_imgs, all_labels, all_txts
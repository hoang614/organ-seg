from pathlib import Path

def read_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def get_organ_name(path):
    parts = Path(path).parts

    for split in ["train", "val", "test"]:
        if split in parts:
            idx = parts.index(split)
            return parts[idx + 1]

    return None
from pathlib import Path
import csv
from datetime import datetime


def get_run_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def init_history_file(path):
    path = Path(path)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "epoch", "train_loss", "val_loss", "val_dice", "lr"])


def append_history(path, run_id, epoch, train_loss, val_loss, val_dice, lr):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([run_id, epoch, train_loss, val_loss, val_dice, lr])
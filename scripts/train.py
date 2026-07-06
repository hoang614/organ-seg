import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoTokenizer
from monai.data import DataLoader, Dataset, decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, Activations
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from datasets.data_paths import build_dataset_paths, load_organ_data
from datasets.dataset import build_train_dataset
from trainers.trainer import train_one_epoch
from trainers.optim import build_optimizer, build_scheduler
from utils.freeze import freeze_encoders
from utils.transforms import train_transforms, val_transforms
from utils.predict import predictor
from utils.history import get_run_id, init_history_file, append_history
from losses.loss import get_loss
from models.multimodal_model import MultiModalSegModel
from .early_stopping import EarlyStopping

TRAIN_ROOT = "./data_preprocessed/train"
VAL_ROOT = "./data_preprocessed/val"
CKPT_DIR = "./ckpt"
HISTORY_PATH = f"{CKPT_DIR}/history.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ADDITIONAL_EPOCHS = 20
VAL_INTERVAL = 1
PATIENCE = 8
TRAIN_ORGANS = os.listdir(TRAIN_ROOT)
VAL_ORGANS = os.listdir(VAL_ROOT)

os.makedirs(CKPT_DIR, exist_ok=True)

init_history_file(HISTORY_PATH)

##! ==== DATA ====
train_img_paths, train_label_paths, train_txt_paths = build_dataset_paths(TRAIN_ROOT, TRAIN_ORGANS)
val_img_paths, val_label_paths, val_txt_paths = build_dataset_paths(VAL_ROOT, VAL_ORGANS)

train_ds = build_train_dataset(train_img_paths, train_label_paths, train_txt_paths, train_transforms)
train_loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

val_ds = build_train_dataset(val_img_paths, val_label_paths, val_txt_paths, val_transforms)
val_loader = DataLoader(
    val_ds,
    batch_size=1,
    num_workers=6,
)

##! ==== MODEL ====
model = MultiModalSegModel().to(DEVICE)
freeze_encoders(model)

optimizer = build_optimizer(model)
scheduler = build_scheduler(optimizer)
loss_fn = get_loss()
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

dice_metric = DiceMetric(include_background=False, reduction="mean")
post_pred = Compose([
    Activations(sigmoid=True), 
    AsDiscrete(threshold=0.5)
])
post_label = AsDiscrete(threshold=0.5)
early_stopper = EarlyStopping(patience=PATIENCE)

best_val_loss = float('inf')
best_train_loss = float('inf')
best_dice = -1.0
start_epoch = 0

latest_path = f"{CKPT_DIR}/latest_checkpoint.pth"
best_path = f"{CKPT_DIR}/best_checkpoint.pth"

if os.path.exists(latest_path):
    ckpt_latest = torch.load(latest_path, map_location=DEVICE)
    
    model.load_state_dict(ckpt_latest["model_state_dict"])
    optimizer.load_state_dict(ckpt_latest["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt_latest["scheduler_state_dict"])
    # optimizer = build_optimizer(model, lr=1.6e-5) 
    # scheduler = build_scheduler(optimizer)
    
    start_epoch = ckpt_latest["epoch"]
    
    print(f">>> Resumed from latest checkpoint at epoch {start_epoch}")
    
    if os.path.exists(best_path):
        ckpt_best = torch.load(best_path, map_location=DEVICE)
        best_dice = ckpt_best.get("dice", best_dice)
        best_val_loss = ckpt_best.get("val_loss", best_val_loss)
        best_train_loss = ckpt_best.get("train_loss", best_train_loss)
        
        print(
            f">>> Historical Record: "
            f"Best Dice {best_dice:.4f} | "
            f"Best Val Loss {best_val_loss:.4f} | "
            f"Best Train Loss {best_train_loss:.4f}"
        )
else:
    print(">>> No checkpoint found, starting training from scratch.")
   
final_epoch = start_epoch + ADDITIONAL_EPOCHS

for epoch in tqdm(range(start_epoch, final_epoch), desc="Training Progress"):

    ##! TRAINING PHASE
    model.train()
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, tokenizer, DEVICE)
    
    ##! VALIDATION PHASE
    current_dice = 0.0
    val_loss = float('inf')
    if (epoch + 1) % VAL_INTERVAL == 0:
        model.eval()
        with torch.no_grad():
            val_loss_total = 0.0
            num_batches = 0
            
            for val_data in tqdm(val_loader, desc="Validation Progress"):
                val_inputs = val_data["image"].to(DEVICE)
                val_labels = val_data["label"].to(DEVICE)
                val_texts = val_data["text"]
                
                prompt_inputs = tokenizer(
                                    val_texts,
                                    padding=True,
                                    truncation=True,
                                    max_length=48,
                                    return_tensors="pt"
                                ).to(DEVICE)
                
                val_outouts = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=(96, 96, 96),
                    sw_batch_size=4,
                    predictor=lambda x: predictor(model, x, prompt_inputs)
                )
                
                loss = loss_fn(val_outouts, val_labels)
                val_loss_total += loss.item()
                num_batches += 1
                
                # val_outouts = post_pred(val_outouts)
                # val_labels = post_label(val_labels)
                
                val_outouts = [post_pred(i) for i in decollate_batch(val_outouts)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                
                dice_metric(y_pred=val_outouts, y=val_labels)
            
            val_loss = val_loss_total / num_batches
            current_dice = dice_metric.aggregate().item()
            dice_metric.reset()

    ##! CHECKPOINTING & LOGGING
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Result: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {current_dice:.4f} | LR: {current_lr:.2e}")

    RUN_ID = get_run_id()
    # Append to history file
    append_history(HISTORY_PATH, RUN_ID, epoch + 1, train_loss, val_loss, current_dice, current_lr)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss

    if train_loss < best_train_loss:
        best_train_loss = train_loss
    
    if current_dice > best_dice:
        best_dice = current_dice
        
        best_ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "dice": best_dice,
            "train_loss": train_loss
        }
        
        torch.save(best_ckpt, best_path)
        print(f"Best checkpoint updated !")
        
    latest_ckpt = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
        "dice": current_dice,
        "train_loss": train_loss,
    }
    
    torch.save(latest_ckpt, latest_path)
    
    early_stopper(current_dice)
    if early_stopper.early_stop:
        print(f"Early stopping triggered at epoch {epoch + 1} with best dice {best_dice:.4f}")
        break
    
    # break

print(f"Training completed. Best Dice: {best_dice:.4f}")
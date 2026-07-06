import torch.optim as optim

def build_optimizer(model, lr=1e-4):
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(params, lr=lr, weight_decay=1e-5)


def build_scheduler(optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = 'min',
        factor=0.5,
        patience=5,
        threshold=1e-4,
        min_lr=1e-6
    )
class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_dice = None
        self.early_stop = False

    def __call__(self, current_dice):
        if self.best_dice is None:
            self.best_dice = current_dice
        elif current_dice < self.best_dice + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_dice = current_dice
            self.counter = 0
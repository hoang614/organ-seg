from monai.losses import DiceCELoss, TverskyLoss, DiceFocalLoss
from monai.transforms import LoadImage, EnsureTyped
import numpy as np

def get_loss():
    # return DiceCELoss(sigmoid=True, squared_pred=False, batch=True, lambda_dice=0.8, lambda_ce=0.2)
    return DiceFocalLoss(
        sigmoid=True,
        gamma=2.0,
        lambda_dice=1.0, 
        lambda_focal=1.0,
        include_background=False 
    )
    
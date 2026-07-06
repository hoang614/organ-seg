from monai.transforms import (
    Compose, LoadImaged, Orientationd, EnsureChannelFirstd, Spacingd, Invertd,
    ScaleIntensityRanged, RandCropByPosNegLabeld, ToTensord, EnsureTyped
)

ROI_SIZE = (96, 96, 96)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    
    EnsureChannelFirstd(keys=["image", "label"]),

    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=ROI_SIZE,
        pos=3,
        neg=1,
        num_samples=4,
    ),

    ToTensord(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"], track_meta=True)
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"], image_only=False),
    EnsureChannelFirstd(keys=["image", "label"]),
    ToTensord(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"], track_meta=True)
])

inference_transforms = Compose([
    LoadImaged(keys=["image"], image_only=True),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(1.2, 1.2, 1.2), mode=("bilinear")),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-175, a_max=250,
        b_min=0.0, b_max=1.0,
        clip=True
    ),
    EnsureTyped(keys=["image"], track_meta=True)
])

invert_transforms = Compose([
    Invertd(
        keys="pred",
        transform=inference_transforms,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        nearest_interp=True,
        to_tensor=True
    )
])
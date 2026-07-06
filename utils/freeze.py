def freeze_encoders(model):
    for p in model.image_encoder.swin_vit.parameters():
        p.requires_grad = False

    for p in model.text_encoder.text_model.parameters():
        p.requires_grad = False
from models.DSSM.model import DSSMDetectionModel


def freeze_streaming_layers(model, k=".m1"):
    for name, param in model.named_parameters():
        if k in name:
            param.requires_grad = False
    print(f"Freeze streaming block...")


def unfreeze_streaming_layers(model, k=".m1"):
    for name, param in model.named_parameters():
        if k in name:
            param.requires_grad = True
    print(f"Unfreeze streaming block...")


def build_DSSM_Detector(args):
    model = DSSMDetectionModel(args)
    freeze_streaming_layers(model, k=".m1")
    return model

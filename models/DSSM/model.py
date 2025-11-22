from util.misc.yaml import yaml_model_load
from copy import deepcopy
import torch
import torch.nn as nn
import math
import contextlib
from models.DSSM.modules import FConv, Conv, C2f, C3k2, Concat, A2C2f, Detect, SPPF, v10Detect, SST
from util.misc.loss import v8DetectionLoss


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def parse_model(d, ch):
    import ast
    legacy = True
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            print(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # Conv.default_act = nn.SiLU()
        FConv.default_act = eval(act)

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n

        if m in {
            FConv,
            Conv,
            C2f,
            SPPF,
            C3k2,
            A2C2f,
            SST
        }:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in {
                C2f,
                C3k2,
                A2C2f,
                SST
            }:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":
                    args.append(True)
                    args.append(1.5)
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, v10Detect}:
            args.append([ch[x] for x in f])
            if m in {Detect}:
                m.legacy = legacy
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class BaseModel(nn.Module):
    def forward(self, x, state, feat=None):
        if isinstance(x, dict):
            return self.loss(x, state, feat)
        return self.predict(x, state)

    def predict(self, x, state):
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:
            if m.f != -1:  # 如果不从前一层传递
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if m.type.split(".")[-1] in ["SST", ]:
                x, state, gates, feat = m(x, state)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
        return x, state, feat, gates

    def loss(self, batch, prev_state=None, prev_feat=None, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds, state, feat, gates = self.forward(batch["img"], prev_state) if preds is None else (preds, prev_state, prev_feat, None)
        if batch["boxes"].numel() != 0:
            return self.criterion(preds, batch, state, prev_state, feat, prev_feat, gates), state, feat
        else:
            return (0., 0.), state, feat

    def load(self, weights):
        def intersect_dicts(da, db, exclude=()):
            return {k: v for k, v in da.items() if
                    k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

        model = weights["ema"] if isinstance(weights, dict) else weights
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load

        freeze_layer_names = ['dfl']
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:
                v.requires_grad = True

    def finetune(self, weights):
        weights_state_dict = weights
        model_state_dict = self.model.state_dict()
        for k, v in model_state_dict.items():
            if k in weights_state_dict:
                if weights_state_dict[k].shape == v.shape:
                    model_state_dict[k] = weights_state_dict[k]
                    print(f"Load module {k}")
        self.model.load_state_dict(model_state_dict)
        freeze_layer_names = ['dfl']
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:
                v.requires_grad = True


class DSSMDetectionModel(BaseModel):
    def __init__(self, args, ch=3):
        super().__init__()
        self.model_name = "SMVM"
        self.args = args
        self.yaml = yaml_model_load(args.model_cfg)
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch)
        self.names = self.yaml["names"] if self.yaml.get("names") is not None else {i: f"{i}" for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace

            def _forward(x, state):
                if self.end2end:  # 训练=False
                    return self.forward(x, state)["one2many"]
                return self.forward(x, state)

            output = _forward(torch.zeros(1, ch, s, s), None)  # forward
            m.stride = torch.tensor([s / x.shape[-2] for x in output[0]])
            # m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])

        initialize_weights(self)

        freeze_layer_names = ['.dfl']
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:
                v.requires_grad = True

    def init_criterion(self):
        return v8DetectionLoss(self)



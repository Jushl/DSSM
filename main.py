import argparse
import torch
import random
import warnings
import numpy as np
import torch.multiprocessing
from engine import Detection
from models import build_model
from util.optim.ema import ModelEMA
from dataset import build_TDSDataset
from dataset import TwoStageDecoupledStreamingDataloader as TDSDataloader
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")


def setup_seed(seed: int, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available() and deterministic:
        torch.backends.cudnn.deterministic = True


def main(args):
    device = torch.device(args.device)
    setup_seed(args.seed)

    dataset_train = build_TDSDataset(mode='train', args=args)
    dataset_val = build_TDSDataset(mode='val', args=args)

    dataloader_train = TDSDataloader(mode='train', dataset=dataset_train, args=args)
    dataloader_val = TDSDataloader(mode='val', dataset=dataset_val, args=args)

    model = build_model(args).to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    ema = ModelEMA(model)

    det = Detection(model,
                    scaler,
                    ema,
                    dataloader_train,
                    dataloader_val,
                    device,
                    args)

    if args.test_only:
        det.val()
    else:
        det.train()


def get_args_parser():
    parser = argparse.ArgumentParser('DSSM Detector', add_help=False)

    # 训练参数
    parser.add_argument('--batch_size_train', default=16, type=int,  help='每次训练读取数据个数')
    parser.add_argument('--batch_size_train_temporal', default=16, type=int,  help='每次训练读取数据个数')
    parser.add_argument('--batch_size_val', default=1, type=int,  help='每次训练读取数据个数')
    parser.add_argument('--epochs', default=72, type=int, help='训练代数')
    parser.add_argument('--epoch', default=0, type=int, help='当前训练代数')
    parser.add_argument('--tepochs', default=8, type=int, help='时空训练代数')

    parser.add_argument('--test_only', default=False, help='是否只验证，默认不验证，直接训练')
    parser.add_argument('--resume', type=str, help='从中断点处继续训练',
                        default="",
                        )
    parser.add_argument('--finetune', type=str, help='对模型进行微调',
                        default=r''
                        )
    # 数据集
    parser.add_argument('--dataset_path', type=str, help='数据集的root路径',
                        default=r"D:\publicData\EMRS-DSSM-Frame"
                        )
    parser.add_argument('--num_workers', default=16, help='线程数')
    parser.add_argument('--ttransforms', default=False, type=bool, help='马赛克数据增强')
    parser.add_argument('--imgsz', default=640, type=int, help='预处理时和马赛克增强时的scale大小')

    # 模型
    parser.add_argument('--model_cfg', default='cfg/dssm-s.yaml', type=str, help='DSSM.yaml路径')

    # 训练设备
    parser.add_argument('--output_dir', default='outputs/dssm/', help='输出路径，包括权重，训练日志')
    parser.add_argument('--device', default='cuda', help='默认使用GPU训练')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--checkpoint_freq', default=10, type=int, help='权重保存频率')

    # 优化器
    parser.add_argument('--amp', default=True, help="混合精度训练,默认使用混合精度训练")
    parser.add_argument('--optimizer', default='auto', help="自动选择优化器")

    #
    parser.add_argument('--half', default=False, help="半精度浮点型")
    parser.add_argument('--nbs', default=64, help="normal batch size")
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--momentum', default=0.937, type=float)
    parser.add_argument('--lr0', default=0.01, type=float)
    parser.add_argument('--lrf', default=0.01, type=float)
    parser.add_argument('--cos_lr', default=False, type=bool)

    parser.add_argument('--box', default=7.5, type=float)
    parser.add_argument('--cls', default=0.5, type=float)
    parser.add_argument('--dfl', default=1.5, type=float)
    parser.add_argument('--stl', default=1.0, type=float)

    parser.add_argument('--conf', default=0.001, type=float)
    parser.add_argument('--iou', default=0.6, type=float)
    parser.add_argument('--max_det', default=300, type=int)

    parser.add_argument('--warmup_epochs', default=3.0, type=float)
    parser.add_argument('--warmup_momentum', default=0.8, type=float)
    parser.add_argument('--warmup_bias_lr', default=0.0, type=float)

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DSSM training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

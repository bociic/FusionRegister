import argparse
import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np

import datasets
import utils
from models import DenoisingDiffusion
from datasets.common import YCbCr2RGB

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def dict2namespace(config):
    ns = argparse.Namespace()
    for k, v in config.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Latent-Retinex Diffusion Models — Testing')
    parser.add_argument("--config", default="unsupervised.yml", type=str,
                        help="Path to the config file under ./configs")
    parser.add_argument("--ckpt", default='ckpt/checkpoint.pth.tar', type=str,
                        help="Checkpoint path to load (training saved file)")
    parser.add_argument("--image_folder", default="Test_results/", type=str,
                        help="Folder to save fused RGB outputs (same style as training)")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    config.device = device
    print(f"[Test] Using device: {device}")
    return args, config

@torch.no_grad()
def run_test(ddm, loader, step, image_folder, device):
    save_dir = image_folder
    os.makedirs(save_dir, exist_ok=True)

    ddm.model.eval()
    print(f"[Test] Saving to: {save_dir}")

    for i, (x, y, vi_cb, vi_cr) in enumerate(loader):
        b, _, img_h, img_w = x.shape

        img_h_64 = int(64 * np.ceil(img_h / 64.0))
        img_w_64 = int(64 * np.ceil(img_w / 64.0))

        x_pad = F.pad(x, (0, img_w_64 - img_w, 0, img_h_64 - img_h), mode='reflect')

        out = ddm.model(x_pad.to(device))
        vi_cb = vi_cb.to(device)
        pred_x = out["pred_x"][:, :, :img_h, :img_w]

        pred_rgb = YCbCr2RGB(pred_x, vi_cb.to(device), vi_cr.to(device))

        for k in range(b):
            name = y[k] if isinstance(y[k], str) else str(y[k])
            utils.logging.save_image(pred_rgb[k:k+1], os.path.join(save_dir, f"{name}"))

        if (i + 1) % 10 == 0:
            print(f"[Test] Processed {i+1} batches")

def main():
    args, config = parse_args_and_config()

    DATASET = datasets.__dict__[config.data.type](config)
    ddm = DenoisingDiffusion(args, config)
    ddm.model.to(config.device)

    ckpt = utils.logging.load_checkpoint(args.ckpt, None)
    ddm.model.load_state_dict(ckpt["state_dict"], strict=True)
    ddm.start_epoch = ckpt.get("epoch", 0)
    ddm.step = ckpt.get("step", 0)

    test_loader = DATASET.get_test_loaders(
        parse_patches=True,
        phase='test'
    )

    os.makedirs(args.image_folder, exist_ok=True)

    run_test(ddm, test_loader, ddm.step, args.image_folder, config.device)
    print("[Test] Done.")

if __name__ == "__main__":
    main()

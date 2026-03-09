import argparse
from torch.utils.data import DataLoader
import utils
from data.data_RGB import get_test_data
from models.FusionRegister import *
from tqdm import tqdm
from models.layers import *
import time

parser = argparse.ArgumentParser(description='EVA')
parser.add_argument('--input_dir', default='/data/Disk_p1/congcong/MISCFilter/dataset/test/llvip/ir/', type=str, help='Directory of validation images')
parser.add_argument('--output_dir', default='/data/Disk_p1/congcong/MISCFilter/dataset/test/llvip/hclfusion+/', type=str, help='Directory of validation images')
parser.add_argument('--weights', default="/data/Disk_p1/congcong/MISCFilter/checkpoints/hclfusion/model_best.pth", type=str, help='Path to weights')
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
result_dir = args.output_dir
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
model_restoration = myNet(inference=False)
utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
# model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# test_dataset = get_validation_data(args.input_dir)
test_dataset = get_test_data(args.input_dir)
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,  drop_last=False, pin_memory=True)

psnr_val_rgb = []
ssim_val_rgb = []
psnr = 0

import yaml
def dict2namespace(config):
    ns = argparse.Namespace()
    for k, v in config.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns
def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Latent-Retinex Diffusion Models — Testing')
    parser.add_argument("--config", default="./models/HCLFuse/configs/unsupervised.yml", type=str,
                        help="Path to the config file under ./configs")
    parser.add_argument("--ckpt", default='./models/HCLFuse/ckpt/checkpoint.pth.tar', type=str,
                        help="Checkpoint path to load (training saved file)")
    parser.add_argument("--image_folder", default="Test_results/", type=str,
                        help="Folder to save fused RGB outputs (same style as training)")
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    config.device = device
    print(f"[Test] Using device: {device}")
    return args, config

from models.HCLFuse.models import DenoisingDiffusion
import models.HCLFuse.utils.logging as logging
utils.mkdir(result_dir)
args_diff, config = parse_args_and_config()
fu = DenoisingDiffusion(args_diff, config)
fu.model.to(config.device)
ckpt = logging.load_checkpoint(args_diff.ckpt, None)
fu.model.load_state_dict(ckpt["state_dict"], strict=True)
fu.start_epoch = ckpt.get("epoch", 0)
fu.step = ckpt.get("step", 0)
fu.model.eval()


from PIL import Image
with torch.no_grad():
    psnr_list = []
    ssim_list = []
    t = []
    tt=[]
    ttt = []
    ab_t =[]
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        with torch.no_grad():

            ir = data_test[0].cuda()
            vi = data_test[1].cuda()
            path_gt = data_test[2][0]
            original_h, original_w = ir.shape[2], ir.shape[3]

            pad_right, pad_bottom = 0, 0
            if original_w % 2 != 0:
                pad_right = 1
            if original_h % 2 != 0:
                pad_bottom = 1

            if pad_right > 0 or pad_bottom > 0:
                ir = F.pad(ir, (0, pad_right, 0, pad_bottom), mode='reflect')  # 'reflect' is a common padding mode
                vi = F.pad(vi, (0, pad_right, 0, pad_bottom), mode='reflect')

            print(os.path.join(result_dir, path_gt.split('/')[0]))
            y, cb, cr = RGB2YCrCb(vi)
            cb = torch.unsqueeze(cb, 0)
            cr = torch.unsqueeze(cr, 0)
            a = time.time()

            b, _, img_h, img_w = y.shape
            img_h_64 = int(64 * np.ceil(img_h / 64.0))
            img_w_64 = int(64 * np.ceil(img_w / 64.0))
            x_pad = F.pad(torch.cat([ir, y], 1), (0, img_w_64 - img_w, 0, img_h_64 - img_h), mode='reflect')
            with torch.no_grad():
                out = fu.model(x_pad)
                input_ = out["pred_x"][:, :, :img_h, :img_w]
                input_ = YCbCr2RGB(input_, cb, cr)

            o = (input_ - torch.min(input_)) / (torch.max(input_) - torch.min(input_))
            o = np.transpose((torch.squeeze(o, 0).cpu().detach().numpy() * 255.),
                                    axes=(1, 2, 0)).astype(np.float32)
            im = Image.fromarray(np.uint8(o))
            im.convert('RGB').save(os.path.join("/data/Disk_p1/congcong/MISCFilter/dataset/test/llvip/hclfusion/", path_gt.split('/')[0]), format='PNG')

            c = time.time()
            # restored, restored_inter, mask, flow, bias, nobias, att_fea = model_restoration(input_, vi, ir)
            restored, restored_inter, mask, flow, bias, restored, att_fea = model_restoration(input_, vi, ir)
            b = time.time()
            ttt.append(c-a)
            tt.append(b-c)
            t.append(b-a)
            # ab_t.append(mrbt)
            # Restore to original size if dimensions were adjusted
            if pad_right > 0 or pad_bottom > 0:
                restored = restored[:, :, :original_h, :original_w]

            restored = (restored- torch.min(restored)) / (torch.max(restored) - torch.min(restored))
            restored = np.transpose((torch.squeeze(restored, 0).cpu().detach().numpy() * 255.),
                                   axes=(1, 2, 0)).astype(np.float32)
            im = Image.fromarray(np.uint8(restored))
            im.convert('RGB').save(os.path.join(result_dir, path_gt.split('/')[0]), format='PNG')

    print("fusion: mean:%s " % (np.mean(ttt)))
    print("reg: mean:%s " % (np.mean(tt)))
    print("all: mean:%s " % (np.mean(t)))
    print("mrb: mean:%s " % (np.mean(ab_t)))
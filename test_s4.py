import argparse
from torch.utils.data import DataLoader
import utils
from data.data_RGB import get_test_data
from models.FusionRegister import *
from tqdm import tqdm
from models.S4Fusion import *
import time

parser = argparse.ArgumentParser(description='EVA')

parser.add_argument('--input_dir', default='/data/Disk_p1/congcong/llvip/ir/', type=str, help='Directory of validation images')
parser.add_argument('--output_dir', default='./results/llvip/S4Fusion+/', type=str, help='Directory of validation images')
parser.add_argument('--weights', default=r"/data/Disk_p1/congcong/MISCFilter/checkpoints/s4fusion/model_latest.pth", type=str, help='Path to weights')
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

utils.mkdir(result_dir)


fu = MambaNet().cuda()
pretrained_state_dict = torch.load("/data/Disk_p1/congcong/S4Fusion/model/model.pkl", map_location='cuda:0')
fu.load_state_dict(pretrained_state_dict['model'])
for para in fu.parameters():
    para.requires_grad = False
fu.eval()


from PIL import Image
with torch.no_grad():
    psnr_list = []
    ssim_list = []
    t = []
    tt=[]
    ttt = []
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
            left_1_pad, right_1_pad, left_1, right_1 = compute_pad(y.squeeze(0).shape[1])
            left_2_pad, right_2_pad, left_2, right_2 = compute_pad(vi.squeeze(0).shape[2])
            vi_s4, ir_s4 = F.pad(y, (left_2_pad, right_2_pad, left_1_pad, right_1_pad), "constant", 0.), \
                           F.pad(ir, (left_2_pad, right_2_pad, left_1_pad, right_1_pad), "constant", 0.)
            input_ = fu(ir_s4, vi_s4)
            input_ = input_[:, :, left_1:right_1, left_2: right_2]
            input_ = YCbCr2RGB(input_, cb, cr)

            f = (input_ - torch.min(input_)) / (torch.max(input_) - torch.min(input_))
            f = np.transpose((torch.squeeze(f, 0).cpu().detach().numpy() * 255.),
                             axes=(1, 2, 0)).astype(np.float32)
            im = Image.fromarray(np.uint8(f))
            im.convert('RGB').save(
                os.path.join("/data/Disk_p1/congcong/MISCFilter/results/llvip/S4Fusion/", path_gt.split('/')[0]),
                format='PNG')

            c = time.time()
            restored, restored_inter, mask, flow, bias, nobias, att_fea = model_restoration(input_, vi, ir)
            b = time.time()
            ttt.append(c-a)
            tt.append(b-c)
            t.append(b-a)
            # Restore to original size if dimensions were adjusted
            if pad_right > 0 or pad_bottom > 0:
                restored = restored[:, :, :original_h, :original_w]

            restored = (restored[0] - torch.min(restored[0])) / (torch.max(restored[0]) - torch.min(restored[0]))
            restored = np.transpose((torch.squeeze(restored, 0).cpu().detach().numpy() * 255.),
                                   axes=(1, 2, 0)).astype(np.float32)
            im = Image.fromarray(np.uint8(restored))
            im.convert('RGB').save(os.path.join(result_dir, path_gt.split('/')[0]), format='PNG')

    print("fusion: mean:%s " % (np.mean(ttt)))
    print("reg: mean:%s " % (np.mean(tt)))
    print("all: mean:%s " % (np.mean(t)))
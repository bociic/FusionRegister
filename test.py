import argparse
from torch.utils.data import DataLoader
import utils
from data.data_RGB import get_test_data
from models.FusionRegister import *
from tqdm import tqdm
from models.layers import *
import time

parser = argparse.ArgumentParser(description='FusionRegister')
parser.add_argument('--input_dir', default='/path/to/infrared/image/folder/', type=str, help='Directory of validation images')
parser.add_argument('--output_dir', default='./register_results/', type=str, help='Directory of Register Results')
parser.add_argument('--fusion_dir', default='./fusion_results/', type=str, help='Directory of Original Fusion Results')
parser.add_argument('--register_weights', default="/path/to/FusionRegister/weight/", type=str, help='Path to weights')
parser.add_argument('--fusion_weights', default="/path/to/Fusion-net/weight/", type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

result_dir = args.output_dir
fusion_dir = args.fusion_dir
fusion_weights = args.fusion_weights

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
model = FusionRegister(inference=False)
utils.load_checkpoint(model, args.register_weights)
print("===>Testing using weights: ", args.weights)
model.cuda()
model.eval()

test_dataset = get_test_data(args.input_dir)
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,  drop_last=False, pin_memory=True)

psnr_val_rgb = []
ssim_val_rgb = []
psnr = 0

utils.mkdir(result_dir)
utils.mkdir(fusion_dir)
fu = fusionnet().cuda()
pretrained_state_dict = torch.load(fusion_weights)
fu.load_state_dict(pretrained_state_dict, strict=True)
fu.eval()

for para in fu.parameters():
    para.requires_grad = False
from PIL import Image
with torch.no_grad():
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

            y, cb, cr = RGB2YCrCb(vi)
            cb = torch.unsqueeze(cb, 0)
            cr = torch.unsqueeze(cr, 0)
            a = time.time()
            input_ = fu(torch.cat([ir, y], 1))
            input_ = YCbCr2RGB(input_, cb, cr)

            o = (input_ - torch.min(input_)) / (torch.max(input_) - torch.min(input_))
            o = np.transpose((torch.squeeze(o, 0).cpu().detach().numpy() * 255.),
                                    axes=(1, 2, 0)).astype(np.float32)
            im = Image.fromarray(np.uint8(o))
            im.convert('RGB').save(os.path.join(fusion_dir, path_gt.split('/')[0]), format='PNG')


            _, restored_inter, mask, flow, bias, restored, att_fea, mrbt = model(input_, vi, ir)

            if pad_right > 0 or pad_bottom > 0:
                restored = restored[:, :, :original_h, :original_w]

            restored = (restored- torch.min(restored)) / (torch.max(restored) - torch.min(restored))
            restored = np.transpose((torch.squeeze(restored, 0).cpu().detach().numpy() * 255.),
                                   axes=(1, 2, 0)).astype(np.float32)
            im = Image.fromarray(np.uint8(restored))
            im.convert('RGB').save(os.path.join(result_dir, path_gt.split('/')[0]), format='PNG')

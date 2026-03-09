import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
torch.backends.cudnn.benchmark = True
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import utils
from data.data_RGB import get_training_data, get_training_1
from models.FusionRegister import *
from loss import losses
from warmup_scheduler import GradualWarmupScheduler
import kornia
import argparse
from torch.cuda.amp import GradScaler as GradScaler

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
start_epoch = 1
scaler = GradScaler()
parser = argparse.ArgumentParser(description='FusionRegister')

parser.add_argument('--train_dir', default='/path/to/infrared/image/folder/', type=str, help='Directory of train infrared images')
parser.add_argument('--model_save_dir', default='./checkpoints/', type=str, help='Path to save weights')
parser.add_argument('--fusion_weights', default="/path/to/Fusion-net/weight/", type=str, help='Path to weights')
parser.add_argument('--pretrained_weights', default="/path/to/pretrained/registration/weight/", type=str, help='Path to weights')
parser.add_argument('--num_epochs', default=2000, type=int, help='num_epochs')
parser.add_argument('--batch_size', default=14, type=int, help='batch_size')
parser.add_argument('--print_epochs', default=1, type=int, help='val_epochs')
args = parser.parse_args()

model_dir = os.path.join(args.model_save_dir)
utils.mkdir(model_dir)
log_dir =  os.path.join(args.model_save_dir, 'log.txt')

train_dir = args.train_dir

num_epochs = args.num_epochs
batch_size = args.batch_size

start_lr = 2e-4
end_lr = 1e-6

######### Model ###########
model = FusionRegister()

def get_parameter_number(net):
    total_num = sum(np.prod(p.size()) for p in net.parameters())
    trainable_num = sum(np.prod(p.size()) for p in net.parameters() if p.requires_grad)

    return total_num, trainable_num

# print number of model
total_num, trainable_num = get_parameter_number(model)
print('Total: ', total_num)
print('Trainable: ', trainable_num)
with open(log_dir,"a+") as f:
    f.write('Total: {}\n'.format(total_num))
    f.write('Trainable: {}\n'.format(trainable_num))

model.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
optimizer = optim.Adam(model.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-8)

######### Scheduler ###########
warmup_epochs = 5
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs-warmup_epochs, eta_min=end_lr)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

RESUME = False

Pretrain = False

model_pre_dir = args.pretrained_weights
######### Pretrain ###########
if Pretrain:
    utils.load_checkpoint(model, model_pre_dir)
    print('------------------------------------------------------------------------------')
    print("==> Retrain Training with: " + model_pre_dir)
    print('------------------------------------------------------------------------------')

######### Resume ###########
if RESUME:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids)>1:
    model = nn.DataParallel(model, device_ids=device_ids)


######### Load the Fusion Net ###########
fu = fusionnet().cuda()
fusion_weights = args.fusion_weights
pretrained_state_dict = torch.load(fusion_weights)
fu.load_state_dict(pretrained_state_dict, strict=True)
fu.eval()
for para in fu.parameters():
    para.requires_grad = False

######### Loss ###########
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()
criterion_fft = losses.fftLoss()
criterion_grad = losses.Gradloss()
######### DataLoaders ###########

train_dataset = get_training_data(train_dir)
train_1 = get_training_1(train_dir)
lens = 200
train_dataset_1, _ = random_split(train_1, [lens, len(train_1)-lens])
dataset = train_dataset + train_dataset_1

print('dataset num:',len(dataset))
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, num_epochs + 1))
print('===> Loading datasets')
with open(log_dir,"a+") as f:
    f.write('===> Start Epoch {} End Epoch {} \n'.format(start_epoch, num_epochs + 1))
    f.write('===> Loading datasets\n')
min_loss = 9999
best_psnr = 0
best_epoch = 0
iter = 0
for epoch in range(start_epoch, num_epochs + 1):
    print('epoch', epoch)
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    iter = 0
    model.train()
    if epoch%10 == 0 and epoch>5:
        lens = 150
        train_dataset_1, _ = random_split(train_1, [lens, len(train_1) - lens])
        dataset = train_dataset + train_dataset_1
        print('new dataset num:', len(dataset))
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False,
                                  pin_memory=True)
    for i, data in enumerate(train_loader, 0):
        # zero_grad
        for param in model.parameters():
            param.grad = None

        ir = data[0].cuda()
        vi = data[1].cuda()
        cls = data[3].cuda()
        y, cb, cr = RGB2YCrCb(vi)
        y = y.cuda()
        cb = cb.cuda()
        cr = cr.cuda()

        vi_ori = y[:, :, 2:258, 2:258].cuda()
        ir_ori = ir[:, :, 2:258, 2:258].cuda()

        vi_input, ir_input, vi_ori, ir_ori, cb, cr, m = random_light_transform(y, ir, vi_ori, ir_ori, cb[:, :, 2:258, 2:258], cr[:, :, 2:258, 2:258], cls)
        ir_input = ir_input.cuda()
        vi_input = vi_input.cuda()
        with torch.no_grad():

            ######### Get the Original Fused Image ###########
            target_ = fu(torch.cat([ir_ori, vi_ori], 1))
            target_ = YCbCr2RGB(target_, cb, cr)


            input_ = fu(torch.cat([ir_input, vi_input], 1))
            input_ = YCbCr2RGB(input_, cb, cr)
            target = kornia.geometry.transform.build_pyramid(target_, 3)
            i_ = kornia.geometry.transform.build_pyramid(ir_ori, 3)
            y_ = kornia.geometry.transform.build_pyramid(YCbCr2RGB(vi_ori, cb, cr), 3)

        restored, restored_inter,  mask, flow, bias, nobias,att_fea = model(input_, YCbCr2RGB(vi_input, cb, cr), ir_input)
        m = ma(mask[1],0.05,10)
        m = filter_and_connect_mask(m)
        m_2 = F.avg_pool2d(m, kernel_size=3, stride=2, padding=1)

        loss_fft = criterion_fft(restored[0], target[0])

        loss_char = criterion_char(restored[0], target[0])
        loss_grad = criterion_grad(target[0]*m, restored[0]*m)

        loss_edge = criterion_edge(restored_inter[0], target[0]) + criterion_edge(restored_inter[1], target[1])

        loss_char_inter = criterion_char(restored_inter[0], target[0]) + criterion_char(restored_inter[1], target[1])

        loss = 1*loss_char + 10*loss_char_inter + 0.1 * loss_fft + 10*loss_edge +1* loss_grad
        loss.backward()
        optimizer.step()

        epoch_loss +=loss.item()
        iter += 1

        print('loss/fft_loss', loss_fft, iter)
        print('loss/char_loss', loss_char, iter)
        print('loss/edge_loss', loss_edge, iter)
        print('loss/inter_loss', loss_char_inter, iter)
        print('loss/grad_loss', loss_grad, iter)
        print('loss/iter_loss', loss, iter)
    if epoch % args.print_epochs == 0:
        print('loss/epoch_loss', epoch_loss, epoch)
    scheduler.step()
    epoch_loss = epoch_loss / len(train_loader)
    if epoch_loss < min_loss:
        torch.save(model.state_dict(), os.path.join(model_dir, "model_min.pth"))
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(log_dir,"a+") as f:
        f.write("------------------------------------------------------------------\n")
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f} \n".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
        f.write("------------------------------------------------------------------\n")

    torch.save({
        'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()

                }, os.path.join(model_dir,"model_latest.pth"))

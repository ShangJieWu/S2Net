import os
from metrics import dice_score
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from datetime import datetime
from models.MyNet import MyNet
from data_loader import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options_cod import opt
torch.cuda.set_device(1)
def WiouWbceLoss(input: object, target: object) -> object:
    weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
    wbce = F.binary_cross_entropy_with_logits(input, target, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    input = torch.sigmoid(input)
    inter = ((input * target) * weit).sum(dim=(2, 3))
    union = ((input + target) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

cudnn.benchmark = True
image_root = opt.rgb_root
gt_root = opt.gt_root
test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
save_path = opt.save_path

logging.basicConfig(filename=save_path + 'MyNet.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("MyNet")

model = MyNet()
num_parms = 0

for p in model.parameters():
    num_parms += p.numel()
logging.info("Total Parameters (For Reference): {}".format(num_parms))
print("Total Parameters (For Reference): {}".format(num_parms))

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=352)
test_loader = test_dataset(test_image_root, test_gt_root, testsize=352)
total_step = len(train_loader)

logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, save_path,
        opt.decay_epoch))

# set loss function
step = 0
writer = SummaryWriter(save_path + 'summary')
best_dice = 0
best_epoch = 0
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.cuda()
    model.train()
    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            gts = gts.cuda()

            y1, y2, y3, y4, c = model(images)
            y1 = F.interpolate(y1, size=352, mode='bilinear', align_corners=False)
            y2 = F.interpolate(y2, size=352, mode='bilinear', align_corners=False)
            y3 = F.interpolate(y3, size=352, mode='bilinear', align_corners=False)
            y4 = F.interpolate(y4, size=352, mode='bilinear', align_corners=False)
            c = F.interpolate(c, size=352, mode='bilinear', align_corners=False)
            bce_iou_res = WiouWbceLoss(y1, gts)
            bce_iou_r4 = WiouWbceLoss(y2, gts)
            bce_iou_r3 = WiouWbceLoss(y3, gts)
            bce_iou_r2 = WiouWbceLoss(y4, gts)
            bce_iou_c = WiouWbceLoss(c, gts)
            bce_iou_deep_supervision = bce_iou_res + bce_iou_r3 + bce_iou_r4 + bce_iou_r2 + bce_iou_c
            loss = bce_iou_deep_supervision
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}||loss:{:4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             optimizer.state_dict()['param_groups'][0]['lr'], loss.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  loss:{:4f} , mem_use:{:.0f}MB'.
                        format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'], loss.data,memory_used))
                writer.add_scalar('Loss', loss.data, global_step=step)
        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'S2Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        raise

# test function
def val(test_loader, model):
    global best_Sm, best_epoch
    model.eval()
    with torch.no_grad():
        Dice_sum = 0
        for i in tqdm(range(test_loader.size)):
            image, gt, name = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res, y2, y3, y4, _ = model(image)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            Dice_sum += dice_score(gt, res)
        Dice = Dice_sum / test_loader.size
        return Dice

if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        dice=val(test_loader, model)
        writer.add_scalar('dice', torch.tensor(dice), global_step=epoch)
        print('Epoch: {} Sm: {} ####  bestDice: {} bestEpoch: {}'.format(epoch, dice, best_dice, best_epoch))
        if epoch == 1:
            best_dice = dice
        else:
            if dice > best_dice:
                best_dice = dice
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'S2Net_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} Sm:{} bestEpoch:{} bestSm:{}'.format(epoch, dice, best_epoch, best_dice))
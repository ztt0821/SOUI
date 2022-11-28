import torch
import os
import torch.nn as nn
import numpy as np
import assess
from model import Model
from train import tensor2im
from data import  TEST_DATASET,  TEST_REAL_DATASET,  TEST_CT_DATASET
from dataset2 import get_dataset

from PIL import Image
from option import args
def test():
    # Real20_dataset = TEST_CT_DATASET(args.Real20Path)
    # Real20_loader = torch.utils.data.DataLoader(dataset=Real20_dataset,batch_size=1,shuffle=False,num_workers=0,)
    model = Model()
    model.cuda()

    # params = model.state_dict()  # 获得模型的原始状态以及参数。
    # for k, v in params.items():
    #     print(k)
    # print("_________________________________")

    model = nn.DataParallel(model)
    # if not os.path.exists('train_new/'):
    #     os.mkdir('train_new/')
    # if not os.path.exists('train_new_add/'):
    #     os.mkdir('train_new_add/')

    train_dataset, val_dataset = get_dataset('/home/ted/ztt/dataset/denoise')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

    check = torch.load('checkpoints2new/94_PL_netG.pth',map_location='cuda:0')
    # for k, v in check.items():
    #     print(k)
    # print("*****************************************")
    model.load_state_dict(check)

    # for k, v in state_dict.items():
    #     name = k[7:]
    model.eval()
    with torch.no_grad():
        # real20 test
        ssim, psnr, num = 0, 0, len(val_loader)
        for i, sample_batched in enumerate(val_loader):
            imgA = sample_batched['input']
            imgB = sample_batched['gt1']

            lx1, lx2, feature2, feature1, output_r, output_t, output_grad = model(imgA)
            t_output = tensor2im(output_t)
            t_imgB = tensor2im(imgB)
            # t_grad = tensor2im(output_grad)
            # Image.fromarray(tensor2im(imgA).astype(np.uint8)).save(
            #     'real110_testing2new/' + str(epoch) + '/' + str(i) + str(epoch) + str(args.local_rank) + 'input_a.png')
            # Image.fromarray(t_imgB.astype(np.uint8)).save(
            #     'real110_testing2new/' + str(epoch) + '/' + str(i) + str(epoch) + str(args.local_rank) + 'input_t.png')
            # Image.fromarray(t_output.astype(np.uint8)).save(
            #     'real110_testing2new/' + str(epoch) + '/' + str(i) + str(epoch) + str(args.local_rank) + 'output_t.png')
            # Image.fromarray(t_grad.astype(np.uint8)).save(
            #     'real110_testing2new/' + str(epoch) + '/' + str(i) + str(epoch) + str(
            #         args.local_rank) + 'output_grad.png')
            assess_dict = assess.quality_assess(t_output, t_imgB)
            # print("ssim:", assess_dict['SSIM'], "psnr", assess_dict['PSNR'])
            # logger.info("ssim:", assess_dict['SSIM'], "psnr", assess_dict['PSNR'])
            ssim += assess_dict['SSIM']
            psnr += assess_dict['PSNR']
        # print("************************")
        print('all:ssim:{0}\tpsnr:{1}'.format(ssim/num, psnr/num))

if __name__ == '__main__':
    test()
import torch
import os
import torch.nn as nn
import numpy as np
import assess
from model import Model
from train import tensor2im
from data import  TEST_DATASET,  TEST_REAL_DATASET,  TEST_CT_DATASET

from PIL import Image
from option import args
def test():
    Real20_dataset = TEST_CT_DATASET(args.Real20Path)
    Real20_loader = torch.utils.data.DataLoader(dataset=Real20_dataset,batch_size=1,shuffle=False,num_workers=0,)
    model = Model()
    model.cuda()

    # params = model.state_dict()  # 获得模型的原始状态以及参数。
    # for k, v in params.items():
    #     print(k)
    # print("_________________________________")

    model = nn.DataParallel(model)
    if not os.path.exists('real20_test2_88_new/'):
        os.mkdir('real20_test2_88_new/')
    check = torch.load(args.test_model,map_location='cuda:0')
    # for k, v in check.items():
    #     print(k)
    # print("*****************************************")
    model.load_state_dict(check)

    # for k, v in state_dict.items():
    #     name = k[7:]
    model.eval()
    with torch.no_grad():
        # real20 test
        ssim, psnr, num = 0, 0, 0
        for i, sample_batched in enumerate(Real20_loader):
            imgA = sample_batched[0]
            # imgB = sample_batched[1]
            lx1, lx2, feature2, feature1, output_r, output_t, output_grad = model(imgA)
            t_output = tensor2im(output_t)
            grad_output = tensor2im(output_grad)
            # t_imgB = tensor2im(imgB)
            Image.fromarray(tensor2im(imgA).astype(np.uint8)).save('real20_test2_88_new/' +'/'+ str(i) +'input_a.png')
            # Image.fromarray(t_imgB.astype(np.uint8)).save('real20_test/' +'/'+ str(i) + 'input_t.png')
            Image.fromarray(t_output.astype(np.uint8)).save('real20_test2_88_new/' +'/' + str(i) + 'output_t.png')
            Image.fromarray(grad_output.astype(np.uint8)).save('real20_test2_88_new/' + '/' + str(i) + 'output_grad.png')
            # assess_dict = assess.quality_assess(t_output, t_imgB)
            # ssim += assess_dict['SSIM']
            # psnr += assess_dict['PSNR']
            num += 1
            print(num)
        # print('<real20>\tssim:{0}\tpsnr:{1}\tnum:{2}'.format(ssim / num, psnr / num, num))

if __name__ == '__main__':
    test()
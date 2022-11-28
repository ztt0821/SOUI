import torch
import os
import torch.nn as nn
import numpy as np
import assess
from model import Model
import torchvision.transforms.functional as TF
from train import tensor2im
from data import  TEST_DATASET,  TEST_REAL_DATASET,  TEST_CT_DATASET, TEST_CT_DATASET_EVAL

from PIL import Image
from option import args
from torch.utils.data import Dataset
import glob
import cv2
class val_datasett(Dataset):
    def __init__(self, root_dir, img_size=128):
        self.root_dir = root_dir
        self.dirs = glob.glob(os.path.join(self.root_dir, 'save_clean', '*.png'))
        self.dir_noise = glob.glob(os.path.join(self.root_dir, 'save_noisy', '*.png'))


        self.img_size = img_size


    def __len__(self):

        return len(self.dirs)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        this_dir = self.dirs[idx]
        img_gt = cv2.imread(this_dir)

        this_dir_n = self.dir_noise[idx]
        img_input = cv2.imread(this_dir_n)
        input = TF.to_tensor(img_input)
        gt1 = TF.to_tensor(img_gt)
        image_name = self.dirs[idx].split('/')[-1]
        data = {
                'input': input,
                'gt1': gt1,
            'image_name': image_name
            }

        return data

def test():
    # Real20_dataset = TEST_CT_DATASET_EVAL(args.Real20Path)
    # Real20_loader = torch.utils.data.DataLoader(dataset=Real20_dataset,batch_size=1,shuffle=False,num_workers=0,)
    model = Model()
    model.cuda()

    # params = model.state_dict()  # 获得模型的原始状态以及参数。
    # for k, v in params.items():
    #     print(k)
    # print("_________________________________")

    model = nn.DataParallel(model)
    if not os.path.exists('contspsrs_ct/'):
        os.mkdir('contspsrs_ct/')

    val_dataset = val_datasett('/home/ted/ztt/dataset/denoise')
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
            img_name = sample_batched['image_name'][0]
            imgB = sample_batched['gt1']
            print(img_name)
            # print("imga", imgA.shape)
            # imgB = sample_batched[1]
            lx1, lx2, feature2, feature1, output_r, output_t, output_grad = model(imgA)
            t_output = tensor2im(output_t)
            grad_output = tensor2im(output_grad)
            alli = output_t + output_grad
            allii = tensor2im(alli)

            t_imgB = tensor2im(imgB)
            # Image.fromarray(tensor2im(imgA).astype(np.uint8)).save('real20_test2_88_new/' +'/'+ str(i) +'input_a.png')
            # # Image.fromarray(t_imgB.astype(np.uint8)).save('real20_test/' +'/'+ str(i) + 'input_t.png')
            # Image.fromarray(t_output.astype(np.uint8)).save('real20_test2_88_new/' +'/' + str(i) + 'output_t.png')
            # Image.fromarray(grad_output.astype(np.uint8)).save('real20_test2_88_new/' + '/' + str(i) + 'output_grad.png')

            # Image.fromarray(t_output.astype(np.uint8)).save('contspsrs_ct/' + img_name)

            # Image.fromarray(allii.astype(np.uint8)).save('train_new_add/' + img_name)
            assess_dict = assess.quality_assess(t_output, t_imgB)
            ssim += assess_dict['SSIM']
            psnr += assess_dict['PSNR']
            # num += 1
            # print(num)
        print('<real20>\tssim:{0}\tpsnr:{1}\tnum:{2}'.format(ssim / num, psnr / num, num))

if __name__ == '__main__':
    test()

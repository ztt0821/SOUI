import cv2
import os
import glob
import random

save_clean = "/home/ted/ztt/dataset/denoise/save_clean/"
save_noisy = "/home/ted/ztt/dataset/denoise/save_noisy/"
if not os.path.exists(save_clean):
    os.mkdir(save_clean)
if not os.path.exists(save_noisy):
    os.mkdir(save_noisy)


noise_path = "/home/ted/ztt/dataset/denoise/"
ct_path = "/home/ted/ztt/dataset/denoise/"

dirs = glob.glob(os.path.join(ct_path, 'test_small', '*.png'))
dir_noise = glob.glob(os.path.join(noise_path, 'self_ul_new_test2_bright', '*.png'))
lennoise = len(dir_noise)
list = list(range(0, lennoise))
# self.list.extend(self.list)
random.shuffle(list)
for i in range(len(dirs)):
    clean = cv2.imread(dirs[i])
    clean_p = clean[116: 116+128, 116:116+128, :]
    noise = cv2.imread(dir_noise[list[i]])
    noise_p = noise[35: 35+128, 35:35+128, :]
    noise_img = clean_p + noise_p
    img_name = dirs[i].split('/')[-1]
    cv2.imwrite(save_clean+img_name, clean_p)
    cv2.imwrite(save_noisy+img_name, noise_img)

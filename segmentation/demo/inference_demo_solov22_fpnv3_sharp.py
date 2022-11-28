from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import os


config_file = './configs/solov2/solov22_light_us_448_r101_fpnv3_4gpu_2x_trainnewsharp.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = './work_dirs/solov22_light_us_release_r101_fpnv3_4gpu_2x_newdata_trainnewsharp/epoch_24.pth'
path = '/public/ttzhang9/dataset/us2/test_new_sharp'
save_path = './vis_solov22_fpnv3_trainsharp_75'
if not os.path.exists(save_path):
    os.makedirs(save_path)

img_list = os.listdir(path)
class_name = ['1_abnormal', '2_abnormal', '3_abnormal']

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
for img in img_list:
    img_name = os.path.join(path, img)
    result = inference_detector(model, img_name)


    show_result_ins(img_name, result, class_name, score_thr=0.75, out_file=os.path.join(save_path, img))

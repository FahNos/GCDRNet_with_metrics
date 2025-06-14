import torch
import torch.nn.functional as F
import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from utils import convert_state_dict
from data.preprocess.crop_merge_image import stride_integral

os.sys.path.append('./models/UNeXt')
from models.UNeXt.unext import UNext_full_resolution_padding, UNext_full_resolution_padding_L_py_L

def save_model_in_both_precisions(model, model_name, save_dir):
    """
    Lưu mô hình ở cả hai độ chính xác fp32 và fp16
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Lưu mô hình fp32 (float32)
    fp32_path = os.path.join(save_dir, f"{model_name}_fp32.pt")
    model_fp32 = model.float()
    torch.save(model_fp32.state_dict(), fp32_path)
    print(f"Đã lưu mô hình fp32: {fp32_path}")
    
    # Lưu mô hình fp16 (half precision)
    fp16_path = os.path.join(save_dir, f"{model_name}_fp16.pt")
    model_fp16 = model.half()
    torch.save(model_fp16.state_dict(), fp16_path)
    print(f"Đã lưu mô hình fp16: {fp16_path}")
    
    return fp32_path, fp16_path

def load_model_from_pt(model_class, model_path, precision='fp32', **model_kwargs):
   
    model = model_class(**model_kwargs).cuda()
    state_dict = torch.load(model_path, map_location='cuda:0')
    model.load_state_dict(state_dict)
    
    if precision == 'fp16':
        model = model.half()
    else:
        model = model.float()
    
    model.eval()
    return model

def test_model1_model2(model1, model2, path_list, in_folder, sav_folder, use_fp16=False):
    for im_path in tqdm(path_list):
        in_name = im_path.split('_')[-1].split('.')[0]
        
        im_org = cv2.imread(im_path)
        im_org, padding_h, padding_w = stride_integral(im_org)
        h, w = im_org.shape[:2]
        im = cv2.resize(im_org, (512, 512))
        im = im_org
        
        with torch.no_grad():
            im = torch.from_numpy(im.transpose(2, 0, 1) / 255).unsqueeze(0)
            im_org = torch.from_numpy(im_org.transpose(2, 0, 1) / 255).unsqueeze(0)
            
            if use_fp16:
                im = im.half().cuda()
                im_org = im_org.half().cuda()
            else:
                im = im.float().cuda()
                im_org = im_org.float().cuda()
            
            shadow = model1(im)
            shadow = F.interpolate(shadow, (h, w))
            
            model1_im = torch.clamp(im_org / shadow, 0, 1)
            pred, _, _, _ = model2(torch.cat((im_org, model1_im), 1))
            
            pred = pred[0].permute(1, 2, 0).data.cpu().numpy()
            pred = (pred * 255).astype(np.uint8)
            pred = pred[padding_h:, padding_w:]
        
        cv2.imwrite(im_path.replace(in_folder, sav_folder), pred)

if __name__ == '__main__':
    # Đường dẫn các file checkpoint pkl gốc
    model1_pkl_path = 'pretrained_model/gcnet.pkl'
    model2_pkl_path = 'pretrained_model/drnet.pkl'
    
    # Thư mục lưu các mô hình .pt
    pt_models_dir = 'checkpoints/pt_models'
    
    print("Đang convert và lưu các mô hình...")
    
    # Load và convert model1 (gcnet)
    print("Đang xử lý model1 (gcnet)...")
    model1 = UNext_full_resolution_padding(num_classes=3, input_channels=3, img_size=512).cuda()
    state1 = convert_state_dict(torch.load(model1_pkl_path, map_location='cuda:0')['model_state'])
    model1.load_state_dict(state1)
    model1.eval()
    
    # Lưu model1 ở cả hai độ chính xác
    save_model_in_both_precisions(model1, "gcnet_model", pt_models_dir)
    
    # Load và convert model2 (drnet)
    print("Đang xử lý model2 (drnet)...")
    model2 = UNext_full_resolution_padding_L_py_L(num_classes=3, input_channels=6, img_size=512).cuda()
    state2 = convert_state_dict(torch.load(model2_pkl_path, map_location='cuda:0')['model_state'])
    model2.load_state_dict(state2)
    model2.eval()
    
    # Lưu model2 ở cả hai độ chính xác
    save_model_in_both_precisions(model2, "drnet_model", pt_models_dir)
    
    print("Hoàn thành việc convert và lưu mô hình!")
    print(f"Các mô hình đã được lưu trong thư mục: {pt_models_dir}")
    
    
    # Load mô hình fp32
    model1_fp32 = load_model_from_pt(
        UNext_full_resolution_padding,
        os.path.join(pt_models_dir, "gcnet_model_fp32.pt"),
        precision='fp32',
        num_classes=3, input_channels=3, img_size=512
    )
    
    model2_fp32 = load_model_from_pt(
        UNext_full_resolution_padding_L_py_L,
        os.path.join(pt_models_dir, "drnet_model_fp32.pt"),
        precision='fp32',
        num_classes=3, input_channels=6, img_size=512
    )
    
    # Load mô hình fp16
    model1_fp16 = load_model_from_pt(
        UNext_full_resolution_padding,
        os.path.join(pt_models_dir, "gcnet_model_fp16.pt"),
        precision='fp16',
        num_classes=3, input_channels=3, img_size=512
    )
    
    model2_fp16 = load_model_from_pt(
        UNext_full_resolution_padding_L_py_L,
        os.path.join(pt_models_dir, "drnet_model_fp16.pt"),
        precision='fp16',
        num_classes=3, input_channels=6, img_size=512
    )
    
    # Chạy inference với mô hình fp32
    img_folder = './distorted/'
    sav_folder_fp32 = './enhanced_fp32/'
    sav_folder_fp16 = './enhanced_fp16/'
    
    if not os.path.exists(sav_folder_fp32):
        os.mkdir(sav_folder_fp32)
    if not os.path.exists(sav_folder_fp16):
        os.mkdir(sav_folder_fp16)
    
    if os.path.exists(img_folder):
        im_paths = glob.glob(os.path.join(img_folder, '*'))
        
        if len(im_paths) > 0:
            print(f"\nChạy inference với mô hình fp32...")
            test_model1_model2(model1_fp32, model2_fp32, im_paths, img_folder, sav_folder_fp32, use_fp16=False)
            
            print(f"\nChạy inference với mô hình fp16...")
            test_model1_model2(model1_fp16, model2_fp16, im_paths, img_folder, sav_folder_fp16, use_fp16=True)
        else:
            print(f"Không tìm thấy hình ảnh trong thư mục {img_folder}")
    else:
        print(f"Thư mục {img_folder} không tồn tại")
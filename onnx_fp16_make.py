import torch
import torch.nn.functional as F
import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
import onnx
import onnxruntime

from utils import convert_state_dict
from data.preprocess.crop_merge_image import stride_integral
os.sys.path.append('./models/UNeXt')
from models.UNeXt.unext import UNext_full_resolution_padding, UNext_full_resolution_padding_L_py_L

def test_model1_model2_onnx(ort_session1, ort_session2, path_list, in_folder, sav_folder, use_fp16=False):
    input_name1 = ort_session1.get_inputs()[0].name
    input_name2 = ort_session2.get_inputs()[0].name

    dtype = np.float16 if use_fp16 else np.float32
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    for im_path in tqdm(path_list):
        im_org_cv = cv2.imread(im_path)
        if im_org_cv is None:
            print(f"Warning: Could not read image {im_path}. Skipping.")
            continue
        
        im_padded_cv, padding_h, padding_w = stride_integral(im_org_cv)
        h_padded, w_padded = im_padded_cv.shape[:2]

        # Input cho model1
        im_for_model1_np = (im_padded_cv.transpose(2, 0, 1).astype(np.float32) / 255.0).astype(dtype)
        im_for_model1_np = np.expand_dims(im_for_model1_np, axis=0)

        im_org_torch_like_np = (im_padded_cv.transpose(2, 0, 1).astype(np.float32) / 255.0).astype(dtype)
        im_org_torch_like_np = np.expand_dims(im_org_torch_like_np, axis=0)

        # Inference Model 1 với ONNX
        ort_inputs1 = {input_name1: im_for_model1_np}
        shadow_onnx_outputs = ort_session1.run(None, ort_inputs1)
        shadow_onnx = shadow_onnx_outputs[0] 

        shadow_torch = torch.from_numpy(shadow_onnx) 
        shadow_resized_torch = F.interpolate(shadow_torch, (h_padded, w_padded), mode='bilinear', align_corners=False)
        
        im_org_for_calc_torch = torch.from_numpy(im_org_torch_like_np) 
        model1_im_torch = torch.clamp(im_org_for_calc_torch / shadow_resized_torch, 0, 1)
        model1_im_np = model1_im_torch.numpy()

        # Chuẩn bị input cho Model 2
        input_model2_np = np.concatenate((im_org_torch_like_np, model1_im_np), axis=1) # Nối 2 array FP16

        # Inference Model 2 với ONNX
        ort_inputs2 = {input_name2: input_model2_np}
        onnx_outputs2 = ort_session2.run(None, ort_inputs2)
        pred_onnx = onnx_outputs2[0] 

        shadow_processed = shadow_resized_torch[0].permute(1,2,0).data.cpu().numpy() 
        shadow_processed = (shadow_processed.astype(np.float32) * 255).astype(np.uint8) 
        shadow_processed = shadow_processed[padding_h:, padding_w:]

        model1_im_processed = model1_im_torch[0].permute(1,2,0).data.cpu().numpy()
        model1_im_processed = (model1_im_processed.astype(np.float32) * 255).astype(np.uint8)
        model1_im_processed = model1_im_processed[padding_h:, padding_w:]

        pred_processed = np.transpose(pred_onnx[0], (1, 2, 0)) 
        pred_processed = (pred_processed.astype(np.float32) * 255).astype(np.uint8)
        pred_processed = pred_processed[padding_h:, padding_w:]

        save_path = im_path.replace(in_folder, sav_folder)
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(save_path, pred_processed)


if __name__ == '__main__':
    USE_FP16_ONNX = True # 

    model_dir = './pretrained_model/'
    onnx_model_dir = './onnx_models/'
    os.makedirs(onnx_model_dir, exist_ok=True)

    model1_pkl_path = os.path.join(model_dir, 'gcnet.pkl')
    model2_pkl_path = os.path.join(model_dir, 'drnet.pkl')
    
    onnx_suffix = "_fp16.onnx" if USE_FP16_ONNX else ".onnx"
    model1_onnx_path = os.path.join(onnx_model_dir, f'gcnet{onnx_suffix}')
    model2_onnx_path = os.path.join(onnx_model_dir, f'drnet{onnx_suffix}')

    img_folder = './distorted/'
    sav_folder = f'./enhanced_onnx_fp16__/'
    if not os.path.exists(sav_folder):
        os.makedirs(sav_folder)

    im_paths = glob.glob(os.path.join(img_folder, '*'))
    if not im_paths:
        print(f"Không tìm thấy ảnh nào trong {img_folder}. Vui lòng kiểm tra đường dẫn.")
        dummy_img_path = os.path.join(img_folder, "dummy_image.png")
        if not os.path.exists(dummy_img_path):
            os.makedirs(img_folder, exist_ok=True)
            dummy_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(dummy_img_path, dummy_array)
            print(f"Đã tạo ảnh dummy tại {dummy_img_path} để thử nghiệm.")
            im_paths = [dummy_img_path]
    
    temp_img_for_shape = cv2.imread(im_paths[0])
    if temp_img_for_shape is None:
        raise ValueError(f"Không thể đọc ảnh mẫu: {im_paths[0]} để xác định shape cho ONNX export.")
    
    temp_img_padded, _, _ = stride_integral(temp_img_for_shape)
    dummy_h, dummy_w = temp_img_padded.shape[:2]
    print(f"Sử dụng kích thước dummy cho ONNX export: H={dummy_h}, W={dummy_w}")

    # --- CHUYỂN ĐỔI SANG ONNX ---
    # Model 1
    if not os.path.exists(model1_onnx_path):
        print(f"Chuyển đổi Model 1 sang ONNX {'FP16' if USE_FP16_ONNX else 'FP32'}...")
        model1_pt = UNext_full_resolution_padding(num_classes=3, input_channels=3, img_size=512).cpu()
        state1 = convert_state_dict(torch.load(model1_pkl_path, map_location=torch.device('cpu'))['model_state'])
        model1_pt.load_state_dict(state1)
        model1_pt.eval()

        if USE_FP16_ONNX:
            model1_pt = model1_pt.half()

        dummy_input1 = torch.randn(1, 3, dummy_h, dummy_w, device='cpu')
        if USE_FP16_ONNX:
            dummy_input1 = dummy_input1.half()
        
        torch.onnx.export(model1_pt,
                          dummy_input1,
                          model1_onnx_path,
                          input_names=['input_1'],
                          output_names=['output_1'],
                          dynamic_axes={'input_1': {0: 'batch', 2: 'height', 3: 'width'},
                                        'output_1': {0: 'batch', 2: 'height', 3: 'width'}},
                          ) 
        print(f"Model 1 đã được lưu vào {model1_onnx_path}")
        del model1_pt

    # Model 2
    if not os.path.exists(model2_onnx_path):
        print(f"Chuyển đổi Model 2 sang ONNX {'FP16' if USE_FP16_ONNX else 'FP32'}...")
        model2_pt = UNext_full_resolution_padding_L_py_L(num_classes=3, input_channels=6, img_size=512).cpu()
        state2 = convert_state_dict(torch.load(model2_pkl_path, map_location=torch.device('cpu'))['model_state'])
        model2_pt.load_state_dict(state2)
        model2_pt.eval()

        if USE_FP16_ONNX:
            model2_pt = model2_pt.half() 

        dummy_input2 = torch.randn(1, 6, dummy_h, dummy_w, device='cpu')
        if USE_FP16_ONNX:
            dummy_input2 = dummy_input2.half() 
        
        output_names_model2 = ['pred_output', 'aux_output1', 'aux_output2', 'aux_output3']
        dynamic_axes_model2 = {'input_2': {0: 'batch', 2: 'height', 3: 'width'}}
        for name in output_names_model2:
            dynamic_axes_model2[name] = {0: 'batch', 2: 'height', 3: 'width'}


        torch.onnx.export(model2_pt,
                          dummy_input2,
                          model2_onnx_path,
                          input_names=['input_2'],
                          output_names=output_names_model2,
                          dynamic_axes=dynamic_axes_model2,
                          )
        print(f"Model 2 đã được lưu vào {model2_onnx_path}")
        del model2_pt

    # --- CHẠY INFERENCE BẰNG ONNX ---
    print(f"Chạy inference bằng ONNX Runtime ({'FP16' if USE_FP16_ONNX else 'FP32'})...")
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    print("ONNX Runtime graph optimization: DISABLED")
    
    
    providers = ['CPUExecutionProvider'] 
    if torch.cuda.is_available() and onnxruntime.get_device() == 'GPU':
        providers.insert(0, 'CUDAExecutionProvider') # Ưu tiên GPU
        print("Sử dụng CUDAExecutionProvider.")
    else:
        print("Sử dụng CPUExecutionProvider. Lợi ích của FP16 trên CPU có thể hạn chế.")

    
    ort_session1 = onnxruntime.InferenceSession(model1_onnx_path, sess_options=sess_options, providers=providers)
    ort_session2 = onnxruntime.InferenceSession(model2_onnx_path, sess_options=sess_options, providers=providers)
    
    test_model1_model2_onnx(ort_session1, ort_session2, im_paths, img_folder, sav_folder, use_fp16=USE_FP16_ONNX)   
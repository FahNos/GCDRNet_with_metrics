import torch
import torch.nn.functional as F
import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
import onnx
import onnxruntime
import argparse

from utils import convert_state_dict
from data.preprocess.crop_merge_image import stride_integral
os.sys.path.append('./models/UNeXt')
from models.UNeXt.unext import UNext_full_resolution_padding, UNext_full_resolution_padding_L_py_L

def export_to_onnx_and_ort(model, model_name, dummy_input, onnx_path, ort_path, use_fp16=False, 
                          input_names=None, output_names=None, dynamic_axes=None):
  
    print(f"Chuyển đổi {model_name} sang ONNX {'FP16' if use_fp16 else 'FP32'}...")
    
    # Chuyển model sang FP16
    if use_fp16:
        model = model.half()
        dummy_input = dummy_input.half()
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=12,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"ONNX model đã được lưu: {onnx_path}")
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    
    print(f"Tối ưu hóa {model_name}...")
    
    try:
        from onnx import optimizer
        optimized_model = optimizer.optimize(onnx_model)
        onnx.save(optimized_model, onnx_path)
        print(f"Đã tối ưu hóa ONNX model")
        onnx_model = optimized_model
    except Exception as e:
        print(f"Warning: Không thể tối ưu hóa ONNX model: {e}")
        onnx.save(onnx_model, onnx_path)
    
    print(f"Chuyển đổi {model_name} sang ORT format...")
    
    sess_options = onnxruntime.SessionOptions()
    # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = ort_path
    
    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers.insert(0, 'CUDAExecutionProvider')
    
    try:
        session = onnxruntime.InferenceSession(onnx_path, sess_options, providers=providers)
        print(f"ORT model đã được lưu: {ort_path}")
        del session
    except Exception as e:
        print(f"Lỗi khi tạo ORT file: {e}")
        import shutil
        shutil.copy2(onnx_path, ort_path)
        print(f"Fallback: Đã copy ONNX file thành ORT file")

def create_ort_models():
    """
    Tạo file ORT cho cả FP32 và FP16
    """
    model_dir = './pretrained_model/'
    onnx_model_dir = './onnx_models/'
    ort_model_dir = './ort_models/'
    
    os.makedirs(onnx_model_dir, exist_ok=True)
    os.makedirs(ort_model_dir, exist_ok=True)

    model1_pkl_path = os.path.join(model_dir, 'gcnet.pkl')
    model2_pkl_path = os.path.join(model_dir, 'drnet.pkl')
    
    img_folder = './distorted/'
    im_paths = glob.glob(os.path.join(img_folder, '*'))
    
    if not im_paths:
        print(f"Không tìm thấy ảnh nào trong {img_folder}. Tạo ảnh dummy...")
        dummy_img_path = os.path.join(img_folder, "dummy_image.png")
        os.makedirs(img_folder, exist_ok=True)
        dummy_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(dummy_img_path, dummy_array)
        im_paths = [dummy_img_path]
    
    temp_img_for_shape = cv2.imread(im_paths[0])
    if temp_img_for_shape is None:
        raise ValueError(f"Không thể đọc ảnh mẫu: {im_paths[0]}")
    
    temp_img_padded, _, _ = stride_integral(temp_img_for_shape)
    dummy_h, dummy_w = temp_img_padded.shape[:2]
    print(f"Sử dụng kích thước để export: H={dummy_h}, W={dummy_w}")
    
    for use_fp16 in [False, True]:
        precision = "fp16" if use_fp16 else "fp32"
        print(f"\n{'='*50}")
        print(f"Tạo models {precision.upper()}")
        print(f"{'='*50}")
        
        model1_onnx_path = os.path.join(onnx_model_dir, f'gcnet_{precision}.onnx')
        model1_ort_path = os.path.join(ort_model_dir, f'gcnet_{precision}.ort')
        model2_onnx_path = os.path.join(onnx_model_dir, f'drnet_{precision}.onnx')
        model2_ort_path = os.path.join(ort_model_dir, f'drnet_{precision}.ort')
        
        # --- Model 1 ---
        if not os.path.exists(model1_ort_path):
            print(f"\nXử lý Model 1 ({precision.upper()})...")
            model1_pt = UNext_full_resolution_padding(num_classes=3, input_channels=3, img_size=512).cpu()
            state1 = convert_state_dict(torch.load(model1_pkl_path, map_location=torch.device('cpu'))['model_state'])
            model1_pt.load_state_dict(state1)
            model1_pt.eval()
            
            dummy_input1 = torch.randn(1, 3, dummy_h, dummy_w, device='cpu')
            
            dynamic_axes1 = {
                'input_1': {0: 'batch', 2: 'height', 3: 'width'},
                'output_1': {0: 'batch', 2: 'height', 3: 'width'}
            }
            
            export_to_onnx_and_ort(
                model1_pt, f"Model 1 ({precision})", dummy_input1,
                model1_onnx_path, model1_ort_path, use_fp16,
                input_names=['input_1'],
                output_names=['output_1'],
                dynamic_axes=dynamic_axes1
            )
            
            del model1_pt
            torch.cuda.empty_cache()
        else:
            print(f"Model 1 ORT ({precision}) đã tồn tại: {model1_ort_path}")
            
        # --- Model 2 ---
        if not os.path.exists(model2_ort_path):
            print(f"\nXử lý Model 2 ({precision.upper()})...")
            model2_pt = UNext_full_resolution_padding_L_py_L(num_classes=3, input_channels=6, img_size=512).cpu()
            state2 = convert_state_dict(torch.load(model2_pkl_path, map_location=torch.device('cpu'))['model_state'])
            model2_pt.load_state_dict(state2)
            model2_pt.eval()
            
            dummy_input2 = torch.randn(1, 6, dummy_h, dummy_w, device='cpu')
            
            output_names_model2 = ['pred_output', 'aux_output1', 'aux_output2', 'aux_output3']
            dynamic_axes_model2 = {'input_2': {0: 'batch', 2: 'height', 3: 'width'}}
            for name in output_names_model2:
                dynamic_axes_model2[name] = {0: 'batch', 2: 'height', 3: 'width'}
            
            export_to_onnx_and_ort(
                model2_pt, f"Model 2 ({precision})", dummy_input2,
                model2_onnx_path, model2_ort_path, use_fp16,
                input_names=['input_2'],
                output_names=output_names_model2,
                dynamic_axes=dynamic_axes_model2
            )
            
            del model2_pt
            torch.cuda.empty_cache()
        else:
            print(f"Model 2 ORT ({precision}) đã tồn tại: {model2_ort_path}")

def test_model1_model2_ort(ort_session1, ort_session2, path_list, in_folder, sav_folder, use_fp16=False):
    """
    Hàm inference sử dụng ORT models
    """
    input_name1 = ort_session1.get_inputs()[0].name
    input_name2 = ort_session2.get_inputs()[0].name

    dtype = np.float16 if use_fp16 else np.float32
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    print(f"Bắt đầu inference với {'FP16' if use_fp16 else 'FP32'}...")
    
    for im_path in tqdm(path_list, desc="Processing images"):
        im_org_cv = cv2.imread(im_path)
        if im_org_cv is None:
            print(f"Warning: Could not read image {im_path}. Skipping.")
            continue
        
        im_padded_cv, padding_h, padding_w = stride_integral(im_org_cv)
        h_padded, w_padded = im_padded_cv.shape[:2]

        im_for_model1_np = (im_padded_cv.transpose(2, 0, 1).astype(np.float32) / 255.0).astype(dtype)
        im_for_model1_np = np.expand_dims(im_for_model1_np, axis=0)

        im_org_torch_like_np = (im_padded_cv.transpose(2, 0, 1).astype(np.float32) / 255.0).astype(dtype)
        im_org_torch_like_np = np.expand_dims(im_org_torch_like_np, axis=0)

        # Inference Model 1
        ort_inputs1 = {input_name1: im_for_model1_np}
        shadow_onnx_outputs = ort_session1.run(None, ort_inputs1)
        shadow_onnx = shadow_onnx_outputs[0]

        shadow_torch = torch.from_numpy(shadow_onnx)
        shadow_resized_torch = F.interpolate(shadow_torch, (h_padded, w_padded), mode='bilinear', align_corners=False)
        
        im_org_for_calc_torch = torch.from_numpy(im_org_torch_like_np)
        model1_im_torch = torch.clamp(im_org_for_calc_torch / shadow_resized_torch, 0, 1)
        model1_im_np = model1_im_torch.numpy()

        input_model2_np = np.concatenate((im_org_torch_like_np, model1_im_np), axis=1)

        # Inference Model 2
        ort_inputs2 = {input_name2: input_model2_np}
        onnx_outputs2 = ort_session2.run(None, ort_inputs2)
        pred_onnx = onnx_outputs2[0]  

        pred_processed = np.transpose(pred_onnx[0], (1, 2, 0))
        pred_processed = (pred_processed.astype(np.float32) * 255).astype(np.uint8)
        pred_processed = pred_processed[padding_h:, padding_w:]

        save_path = im_path.replace(in_folder, sav_folder)
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(save_path, pred_processed)

def run_inference_ort(use_fp16=False, use_cpu=False):
  
    precision = "fp16" if use_fp16 else "fp32"
    
    ort_model_dir = './ort_models/'
    model1_ort_path = os.path.join(ort_model_dir, f'gcnet_{precision}.ort')
    model2_ort_path = os.path.join(ort_model_dir, f'drnet_{precision}.ort')
    
    if not os.path.exists(model1_ort_path) or not os.path.exists(model2_ort_path):
        print(f"ORT models ({precision}) chưa tồn tại. Vui lòng tạo trước.")
        return
    
    img_folder = './distorted/'
    sav_folder = f'./enhanced_ort_{precision}/'
    os.makedirs(sav_folder, exist_ok=True)
    
    im_paths = glob.glob(os.path.join(img_folder, '*'))
    if not im_paths:
        print(f"Không tìm thấy ảnh trong {img_folder}")
        return
    
    providers = ['CPUExecutionProvider']
    if not use_cpu and torch.cuda.is_available():
        providers.insert(0, 'CUDAExecutionProvider')
        print(f"Sử dụng GPU cho inference ORT {precision.upper()}")
    else:
        print(f"Sử dụng CPU cho inference ORT {precision.upper()}")
    
    print(f"Tải ORT models ({precision.upper()})...")
    ort_session1 = onnxruntime.InferenceSession(model1_ort_path, providers=providers)
    ort_session2 = onnxruntime.InferenceSession(model2_ort_path, providers=providers)
    
    test_model1_model2_ort(ort_session1, ort_session2, im_paths, img_folder, sav_folder, use_fp16)
    
    print(f"Hoàn thành inference ORT {precision.upper()}. Kết quả lưu tại: {sav_folder}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ORT Model Converter and Inference")
    parser.add_argument('--mode', type=str, choices=['convert', 'inference'], required=True,
                        help="Chế độ: 'convert' để tạo ORT models, 'inference' để chạy inference")
    parser.add_argument('--fp16', action='store_true',
                        help="Sử dụng FP16 (chỉ áp dụng cho inference)")
    parser.add_argument('--cpu', action='store_true',
                        help="Sử dụng CPU thay vì GPU (chỉ áp dụng cho inference)")
    
    args = parser.parse_args()
    
    if args.mode == 'convert':
        print("Tạo ORT models cho cả FP32 và FP16...")
        create_ort_models()
        print("\nHoàn thành tạo ORT models!")
        print("Sử dụng --mode inference để chạy inference")
        
    elif args.mode == 'inference':
        print(f"Chạy inference ORT {'FP16' if args.fp16 else 'FP32'}...")
        run_inference_ort(use_fp16=args.fp16, use_cpu=args.cpu)
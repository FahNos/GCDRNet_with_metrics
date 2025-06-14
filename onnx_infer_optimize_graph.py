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

# --- Hàm inference bằng ONNX ---
def test_model1_model2_onnx(ort_session1, ort_session2, path_list, in_folder, sav_folder, use_gpu_for_torch_ops):
    input_name1 = ort_session1.get_inputs()[0].name
    input_name2 = ort_session2.get_inputs()[0].name

    device = torch.device("cuda" if use_gpu_for_torch_ops and torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Sử dụng GPU cho các phép toán PyTorch: {torch.cuda.get_device_name(0)}")
    else:
        print("Sử dụng CPU cho các phép toán PyTorch.")

    for im_path in tqdm(path_list):
        im_org_cv = cv2.imread(im_path)
        if im_org_cv is None:
            print(f"Warning: Could not read image {im_path}. Skipping.")
            continue
        
        im_padded_cv, padding_h, padding_w = stride_integral(im_org_cv)
        h_padded, w_padded = im_padded_cv.shape[:2]

        im_for_model1_np = im_padded_cv.transpose(2, 0, 1).astype(np.float32) / 255.0
        im_for_model1_np = np.expand_dims(im_for_model1_np, axis=0) 

        im_org_torch_like_np = im_padded_cv.transpose(2, 0, 1).astype(np.float32) / 255.0
        im_org_torch_like_np = np.expand_dims(im_org_torch_like_np, axis=0) 

      
        ort_inputs1 = {input_name1: im_for_model1_np}
        shadow_onnx_outputs = ort_session1.run(None, ort_inputs1) 
        shadow_onnx = shadow_onnx_outputs[0] 

        shadow_torch = torch.from_numpy(shadow_onnx).to(device)
        shadow_resized_torch = F.interpolate(shadow_torch, (h_padded, w_padded), mode='bilinear', align_corners=False)
        
    
        im_org_for_calc_torch = torch.from_numpy(im_org_torch_like_np).to(device)
        model1_im_torch = torch.clamp(im_org_for_calc_torch / shadow_resized_torch, 0, 1)
       
        model1_im_np = model1_im_torch.cpu().numpy() 

        input_model2_np = np.concatenate((im_org_torch_like_np, model1_im_np), axis=1) 

        # Inference Model 2 với ONNX
        ort_inputs2 = {input_name2: input_model2_np}
        onnx_outputs2 = ort_session2.run(None, ort_inputs2)
        pred_onnx = onnx_outputs2[0] 
    
        shadow_processed = shadow_resized_torch[0].permute(1,2,0).data.cpu().numpy()
        shadow_processed = (shadow_processed * 255).astype(np.uint8)
        shadow_processed = shadow_processed[padding_h:, padding_w:] 

        model1_im_processed = model1_im_torch[0].permute(1,2,0).data.cpu().numpy()
        model1_im_processed = (model1_im_processed * 255).astype(np.uint8)
        model1_im_processed = model1_im_processed[padding_h:, padding_w:] 

        pred_processed = np.transpose(pred_onnx[0], (1, 2, 0))
        pred_processed = (pred_processed * 255).astype(np.uint8)
        pred_processed = pred_processed[padding_h:, padding_w:] 

        save_path = im_path.replace(in_folder, sav_folder)
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(save_path, pred_processed)


def test_model1_model2_pytorch(model1, model2, path_list, in_folder, sav_folder, use_gpu_for_pytorch):
    device = torch.device("cuda" if use_gpu_for_pytorch and torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Sử dụng GPU cho các mô hình PyTorch: {torch.cuda.get_device_name(0)}")
    else:
        print("Sử dụng CPU cho các mô hình PyTorch.")

    model1.to(device)
    model2.to(device)

    for im_path in tqdm(path_list):
        im_org_cv = cv2.imread(im_path)
        if im_org_cv is None:
            print(f"Warning: Could not read image {im_path}. Skipping.")
            continue

        im_padded_cv, padding_h, padding_w = stride_integral(im_org_cv)
        h_padded, w_padded = im_padded_cv.shape[:2]                 
        im_for_model1_pt = im_padded_cv 
        
        with torch.no_grad():
            im_tensor = torch.from_numpy(im_for_model1_pt.transpose(2,0,1)/255.0).unsqueeze(0).float().to(device)
            im_org_tensor = torch.from_numpy(im_padded_cv.transpose(2,0,1)/255.0).unsqueeze(0).float().to(device)

            shadow = model1(im_tensor) 
            shadow = F.interpolate(shadow, (h_padded, w_padded), mode='bilinear', align_corners=False)
            
            model1_im = torch.clamp(im_org_tensor / shadow, 0, 1)
            pred,_,_,_ = model2(torch.cat((im_org_tensor, model1_im), 1))         
            
            pred_out = pred[0].permute(1,2,0).data.cpu().numpy()
            pred_out = (pred_out*255).astype(np.uint8)
            pred_out = pred_out[padding_h:,padding_w:]

        save_path = im_path.replace(in_folder,sav_folder)
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(save_path,pred_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ONNX INFER")

    parser.add_argument('-ONNX', type=str, required=True,
                        help="True => ONNX, False => PyTorch")
    parser.add_argument('-CPU', type=str, required=True,
                        help="True => CPU, False => GPU") 

    args = parser.parse_args()

    use_onnx = args.ONNX.lower() == 'true' 
    use_cpu = args.CPU.lower() == 'true'  

    model_dir = './pretrained_model' 
    onnx_model_dir = './onnx_models/' 
    os.makedirs(onnx_model_dir, exist_ok=True)
    
    optimized_onnx_model_dir = os.path.join(onnx_model_dir, 'optimized')
    os.makedirs(optimized_onnx_model_dir, exist_ok=True)


    model1_pkl_path = os.path.join(model_dir, 'gcnet.pkl')
    model2_pkl_path = os.path.join(model_dir, 'drnet.pkl')
    
    model1_onnx_path = os.path.join(onnx_model_dir, 'gcnet.onnx')
    model2_onnx_path = os.path.join(onnx_model_dir, 'drnet.onnx')

    model1_optimized_onnx_path = os.path.join(optimized_onnx_model_dir, 'gcnet_optimized.onnx')
    model2_optimized_onnx_path = os.path.join(optimized_onnx_model_dir, 'drnet_optimized.onnx')


    img_folder = './distorted/'
    sav_folder = './enhanced_onnx/' if use_onnx else './enhanced_pytorch/'
    if not os.path.exists(sav_folder):
        os.makedirs(sav_folder)

    im_paths = glob.glob(os.path.join(img_folder, '*.[jJ][pP][gG]')) + \
               glob.glob(os.path.join(img_folder, '*.[jJ][pP][eE][gG]')) + \
               glob.glob(os.path.join(img_folder, '*.[pP][nN][gG]'))
    if not im_paths:
        print(f"Không tìm thấy ảnh nào trong {img_folder}. Vui lòng kiểm tra đường dẫn.")
        dummy_img_path = os.path.join(img_folder, "dummy_image.png")
        if not os.path.exists(dummy_img_path):
            os.makedirs(img_folder, exist_ok=True)
            dummy_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(dummy_img_path, dummy_array)
            print(f"Đã tạo ảnh dummy tại {dummy_img_path} để thử nghiệm.")
            im_paths = [dummy_img_path]
    
    if not im_paths: 
        raise SystemExit("Không có ảnh để xử lý, kể cả ảnh dummy.")

    
    temp_img_for_shape = cv2.imread(im_paths[0])
    if temp_img_for_shape is None:
        raise ValueError(f"Không thể đọc ảnh mẫu: {im_paths[0]} để xác định shape cho ONNX export.")
    
    temp_img_padded, _, _ = stride_integral(temp_img_for_shape)
    dummy_h, dummy_w = temp_img_padded.shape[:2]
    print(f"Sử dụng kích thước dummy cho ONNX export: H={dummy_h}, W={dummy_w}")

    export_device = torch.device('cpu') 
    run_device_pytorch = torch.device("cuda" if not use_cpu and torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model1_onnx_path):
        print("Chuyển đổi Model 1 sang ONNX...")
        model1_pt = UNext_full_resolution_padding(num_classes=3, input_channels=3, img_size=None).to(export_device) # img_size=None cho dynamic
        state1 = convert_state_dict(torch.load(model1_pkl_path, map_location=torch.device('cpu'))['model_state'])
        model1_pt.load_state_dict(state1)
        model1_pt.eval()
        
        dummy_input1 = torch.randn(1, 3, dummy_h, dummy_w, device=export_device)
        
        torch.onnx.export(model1_pt,
                          dummy_input1,
                          model1_onnx_path,
                          input_names=['input_1'], 
                          output_names=['output_1'], 
                          dynamic_axes={'input_1': {0: 'batch_size', 2: 'height', 3: 'width'}, 
                                        'output_1': {0: 'batch_size', 2: 'height', 3: 'width'}},
                          opset_version=11) 
        print(f"Model 1 đã được lưu vào {model1_onnx_path}")
        del model1_pt

    if not os.path.exists(model2_onnx_path):
        print("Chuyển đổi Model 2 sang ONNX...")
        model2_pt = UNext_full_resolution_padding_L_py_L(num_classes=3, input_channels=6, img_size=None).to(export_device) # img_size=None cho dynamic
        state2 = convert_state_dict(torch.load(model2_pkl_path, map_location=torch.device('cpu'))['model_state'])
        model2_pt.load_state_dict(state2)
        model2_pt.eval()
      
        dummy_input2 = torch.randn(1, 6, dummy_h, dummy_w, device=export_device)
        
        output_names_model2 = ['pred_output', 'aux_output1', 'aux_output2', 'aux_output3']

        torch.onnx.export(model2_pt,
                          dummy_input2,
                          model2_onnx_path,
                          input_names=['input_2'],
                          output_names=output_names_model2,
                          dynamic_axes={'input_2': {0: 'batch_size', 2: 'height', 3: 'width'},
                                        'pred_output': {0: 'batch_size', 2: 'height', 3: 'width'}, 
                                        'aux_output1': {0: 'batch_size', 2: 'height', 3: 'width'}, 
                                        'aux_output2': {0: 'batch_size', 2: 'height', 3: 'width'},
                                        'aux_output3': {0: 'batch_size', 2: 'height', 3: 'width'}},
                          opset_version=11)
        print(f"Model 2 đã được lưu vào {model2_onnx_path}")
        del model2_pt 

    # --- CHẠY INFERENCE ---
    if use_onnx:
        print("Chạy inference bằng ONNX Runtime...")
        
        providers = ['CPUExecutionProvider']
        if not use_cpu and onnxruntime.get_device() == 'GPU':             
            if torch.cuda.is_available():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print("ONNX Runtime sẽ cố gắng sử dụng GPU (CUDAExecutionProvider).")
            else:
                print("Warning: GPU được yêu cầu cho ONNX nhưng PyTorch/CUDA không khả dụng hoặc ONNX Runtime không build với CUDA. Sẽ sử dụng CPU.")
        else:
            print("ONNX Runtime sẽ sử dụng CPU (CPUExecutionProvider).")

        # --- Tạo hoặc load Model 1 Session ---
        sess_options1 = onnxruntime.SessionOptions()
        if os.path.exists(model1_optimized_onnx_path):
            print(f"Sử dụng model 1 đã tối ưu: {model1_optimized_onnx_path} với ORT_DISABLE_ALL.")
            sess_options1.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            ort_session1 = onnxruntime.InferenceSession(model1_optimized_onnx_path, sess_options1, providers=providers)
        else:
            print(f"Tối ưu model 1: {model1_onnx_path} và lưu vào {model1_optimized_onnx_path} với ORT_ENABLE_ALL.")
            sess_options1.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options1.optimized_model_filepath = model1_optimized_onnx_path
            ort_session1 = onnxruntime.InferenceSession(model1_onnx_path, sess_options1, providers=providers)
            print(f"Model 1 đã được tối ưu và lưu vào {model1_optimized_onnx_path}")

        # --- Tạo hoặc load Model 2 Session ---
        sess_options2 = onnxruntime.SessionOptions()
        if os.path.exists(model2_optimized_onnx_path):
            print(f"Sử dụng model 2 đã tối ưu: {model2_optimized_onnx_path} với ORT_DISABLE_ALL.")
            sess_options2.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            ort_session2 = onnxruntime.InferenceSession(model2_optimized_onnx_path, sess_options2, providers=providers)
        else:
            print(f"Tối ưu model 2: {model2_onnx_path} và lưu vào {model2_optimized_onnx_path} với ORT_ENABLE_ALL.")
            sess_options2.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options2.optimized_model_filepath = model2_optimized_onnx_path
            ort_session2 = onnxruntime.InferenceSession(model2_onnx_path, sess_options2, providers=providers)
            print(f"Model 2 đã được tối ưu và lưu vào {model2_optimized_onnx_path}")
        
        test_model1_model2_onnx(ort_session1, ort_session2, im_paths, img_folder, sav_folder, not use_cpu)
    else:
        print("Chạy inference bằng PyTorch...")
        model1 = UNext_full_resolution_padding(num_classes=3, input_channels=3, img_size=None).to(run_device_pytorch) # img_size=None cho dynamic
        state = convert_state_dict(torch.load(model1_pkl_path, map_location=torch.device('cpu'))['model_state'])    
        model1.load_state_dict(state)
        model2 = UNext_full_resolution_padding_L_py_L(num_classes=3, input_channels=6, img_size=None).to(run_device_pytorch) # img_size=None cho dynamic
        state = convert_state_dict(torch.load(model2_pkl_path, map_location=torch.device('cpu'))['model_state'])    
        model2.load_state_dict(state)

        model1.eval()
        model2.eval()
        
        test_model1_model2_pytorch(model1, model2, im_paths, img_folder, sav_folder, not use_cpu)
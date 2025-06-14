import os
import glob
import cv2
import numpy as np
import torch # Cần cho F.interpolate và torch.from_numpy
import torch.nn.functional as F # Cần cho F.interpolate
import onnxruntime
from tqdm import tqdm
import argparse
from metrics_pytorch import calculate_metrics_numpy, stride_integral

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available for PyTorch: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs PyTorch sees: {torch.cuda.device_count()}")
    print(f"Current PyTorch GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

print(f"ONNX Runtime version: {onnxruntime.__version__}")
print(f"ONNX Runtime available providers: {onnxruntime.get_available_providers()}")
print(f"ONNX Runtime device: {onnxruntime.get_device()}")


ALLOWED_INPUT_EXTENSIONS_EVAL = ('.png', '.jpg', '.jpeg', '.bmp')
ALLOWED_GT_EXTENSIONS_EVAL = ('.png', '.jpg', '.jpeg', '.bmp')


def evaluate_onnx_models_with_gt(
    ort_session1,
    ort_session2,
    distorted_img_folder: str,
    gt_img_folder: str,
    use_fp16: bool = False, 
    limit_images: int = None,
    max_side_len=None 
):
   
    input_name1 = ort_session1.get_inputs()[0].name
    print(f"Input name for Model 1: '{input_name1}'")
    input_name2 = ort_session2.get_inputs()[0].name
    print(f"Input name for Model 2: '{input_name2}'")

    output_name1 = ort_session1.get_outputs()[0].name
    print(f"Output name for Model 1: '{output_name1}'")
    output_name2 = ort_session2.get_outputs()[0].name
    print(f"Output name for Model 2: '{output_name2}'")

    all_output_names1 = [output.name for output in ort_session1.get_outputs()]
    print(f"All output names for Model 1: {all_output_names1}")
    all_output_names2 = [output.name for output in ort_session2.get_outputs()]
    print(f"All output names for Model 2: {all_output_names2}")

    onnx_input_dtype = np.float16 if use_fp16 else np.float32    
    torch_processing_dtype = torch.float32

    distorted_image_paths_all = glob.glob(os.path.join(distorted_img_folder, '*'))
    distorted_image_paths = []
    for p in distorted_image_paths_all:
        if p.lower().endswith(ALLOWED_INPUT_EXTENSIONS_EVAL):
            distorted_image_paths.append(p)
    distorted_image_paths.sort()

    if limit_images is not None and limit_images > 0:
        distorted_image_paths = distorted_image_paths[:limit_images]

    if not distorted_image_paths:
        print(f"Không tìm thấy ảnh nào có đuôi hợp lệ {ALLOWED_INPUT_EXTENSIONS_EVAL} trong {distorted_img_folder}")
        return 0.0, 0.0

    total_psnr = 0
    total_ssim = 0
    count = 0

    for distorted_path in tqdm(distorted_image_paths, desc="Evaluating ONNX Models"):
        distorted_filename_full = os.path.basename(distorted_path)
        gt_path_found = None

        if "_in." in distorted_filename_full:
            try:
                base_name_part = distorted_filename_full.rsplit('_in.', 1)[0]
                for ext_gt in ALLOWED_GT_EXTENSIONS_EVAL:
                    potential_gt_filename = f"{base_name_part}_gt{ext_gt}"
                    potential_gt_path = os.path.join(gt_img_folder, potential_gt_filename)
                    if os.path.exists(potential_gt_path):
                        gt_path_found = potential_gt_path
                        break
            except IndexError:
                pass 
        
        if gt_path_found is None: 
            potential_gt_path = os.path.join(gt_img_folder, distorted_filename_full)
            if os.path.exists(potential_gt_path):
                gt_path_found = potential_gt_path

        if gt_path_found is None:
            expected_gt_name_pattern = distorted_filename_full
            if "_in." in distorted_filename_full:
                try: base_name_part = distorted_filename_full.rsplit('_in.', 1)[0]; expected_gt_name_pattern = f"{base_name_part}_gt.<ext>"
                except IndexError: pass
            print(f"Cảnh báo: Không tìm thấy GT cho {distorted_path} (đã tìm '{expected_gt_name_pattern}' và tên gốc). Bỏ qua.")
            continue

        im_distorted_cv = cv2.imread(distorted_path)
        im_gt_cv = cv2.imread(gt_path_found)

        if max_side_len is not None and max_side_len > 0:
            h_orig, w_orig = im_distorted_cv.shape[:2]
            if h_orig > max_side_len or w_orig > max_side_len:
                scale = float(max_side_len) / max(h_orig, w_orig)
                new_w = int(w_orig * scale)
                new_h = int(h_orig * scale)
                if new_w > 0 and new_h > 0: 
                    im_distorted_cv = cv2.resize(im_distorted_cv, (new_w, new_h), interpolation=cv2.INTER_AREA) 
                else:
                    print(f"Cảnh báo: Kích thước mới không hợp lệ ({new_h},{new_w}) cho ảnh {distorted_filename_full}. Bỏ qua resize.")
        

        if im_distorted_cv is None:
            print(f"Lỗi đọc ảnh distorted: '{distorted_path}'. Bỏ qua.")
            continue
        if im_gt_cv is None:
            print(f"Lỗi đọc ảnh GT: '{gt_path_found}'. Bỏ qua.")
            continue

        im_padded_cv, padding_h, padding_w = stride_integral(im_distorted_cv)
        h_padded, w_padded = im_padded_cv.shape[:2]
        
        im_for_model1_np = (im_padded_cv.transpose(2, 0, 1).astype(np.float32) / 255.0)
        im_for_model1_np = np.expand_dims(im_for_model1_np, axis=0).astype(onnx_input_dtype)

        im_org_torch_like_np = (im_padded_cv.transpose(2, 0, 1).astype(np.float32) / 255.0)
        im_org_torch_like_np = np.expand_dims(im_org_torch_like_np, axis=0).astype(onnx_input_dtype)

        # Model 1 inference
        ort_inputs1 = {input_name1: im_for_model1_np}
        shadow_onnx_outputs = ort_session1.run(None, ort_inputs1)
        shadow_onnx = shadow_onnx_outputs[0] 
        # shadow_onnx = shadow_onnx_outputs
        # print(f"- shadow shape = {shadow_onnx[0].shape}")

        shadow_torch = torch.from_numpy(shadow_onnx.astype(np.float32)) 
        # shadow_resized_torch = F.interpolate(shadow_torch, (h_padded, w_padded), mode='bilinear', align_corners=False)
        shadow_resized_torch = shadow_torch
        
        im_org_for_calc_torch = torch.from_numpy(im_org_torch_like_np.astype(np.float32))
        model1_im_torch = torch.clamp(im_org_for_calc_torch / shadow_resized_torch, 0, 1) 
        
        model1_im_np = model1_im_torch.numpy().astype(onnx_input_dtype)

        input_model2_np = np.concatenate((im_org_torch_like_np, model1_im_np), axis=1)

        # input_model2_np = input_model2_np.astype(np.float16)

        ort_inputs2 = {input_name2: input_model2_np}
        onnx_outputs2 = ort_session2.run(None, ort_inputs2)
        pred_onnx = onnx_outputs2[0] 

        pred_out_np = np.transpose(pred_onnx[0], (1, 2, 0)) # (H, W, C)
        pred_out_np = (pred_out_np.astype(np.float32) * 255) # Chuyển sang FP32 
        pred_out_np = np.clip(pred_out_np, 0, 255).astype(np.uint8)
        pred_out_np = pred_out_np[padding_h:, padding_w:] 

        if pred_out_np.shape != im_gt_cv.shape:
            pred_out_np = cv2.resize(pred_out_np, (im_gt_cv.shape[1], im_gt_cv.shape[0]), interpolation=cv2.INTER_CUBIC)

        psnr_val, ssim_val = calculate_metrics_numpy(im_gt_cv, pred_out_np)

        total_psnr += psnr_val
        total_ssim += ssim_val
        count += 1

    if count == 0:
        print("Không có ảnh nào được xử lý thành công để tính metric.")
        return 0.0, 0.0

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    return avg_psnr, avg_ssim


def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình ONNX (gcnet và drnet) và tính PSNR/SSIM.")
    parser.add_argument("--gc", required=True, help="Đường dẫn đến file gcnet.onnx (model1)")
    parser.add_argument("--dr", required=True, help="Đường dẫn đến file drnet.onnx (model2)")
    parser.add_argument("--distorted_dir", default="./distorted_eval_data/",
                        help="Thư mục chứa ảnh đầu vào (distorted). Mặc định: ./distorted_eval_data/")
    parser.add_argument("--gt_dir", default="./gt_eval_data/",
                        help="Thư mục chứa ảnh ground truth. Mặc định: ./gt_eval_data/")
    parser.add_argument("--limit", type=int, default=None,
                        help="Giới hạn số lượng ảnh để đánh giá (để test nhanh). Mặc định: không giới hạn.")
    parser.add_argument("--fp16", action='store_true',
                        help="Sử dụng nếu các mô hình ONNX được export ở dạng FP16.")
    parser.add_argument("--cpu", action='store_true',
                        help="Buộc sử dụng CPUExecutionProvider ngay cả khi có CUDA.")

    parser.add_argument("--resize", type=int, default=None,
                        help="Kích thước tối đa cho một chiều của ảnh đầu vào (pixel). Nếu vượt quá, ảnh sẽ được resize giữ tỉ lệ. Mặc định: không resize.")


    args = parser.parse_args()

    if not os.path.exists(args.gc):
        print(f"Lỗi: Không tìm thấy file model ONNX cho gcnet: {args.gc}")
        return
    if not os.path.exists(args.dr):
        print(f"Lỗi: Không tìm thấy file model ONNX cho drnet: {args.dr}")
        return
    if not os.path.isdir(args.distorted_dir):
        print(f"Lỗi: Không tìm thấy thư mục ảnh distorted: {args.distorted_dir}")
        return
    if not os.path.isdir(args.gt_dir):
        print(f"Lỗi: Không tìm thấy thư mục ảnh ground truth: {args.gt_dir}")
        return

    for data_dir_path in [args.distorted_dir, args.gt_dir]:
        is_default_path = (data_dir_path == "./distorted_eval_data/" or data_dir_path == "./gt_eval_data/")
        if is_default_path and not os.path.exists(data_dir_path):
            os.makedirs(data_dir_path, exist_ok=True)
            print(f"Đã tạo thư mục: {data_dir_path}")

        if not glob.glob(os.path.join(data_dir_path, '*.[jp][pn]g')) and \
           not glob.glob(os.path.join(data_dir_path, '*.jpeg')) and \
           not glob.glob(os.path.join(data_dir_path, '*.bmp')):
            print(f"Cảnh báo: Thư mục {data_dir_path} rỗng.")
            if is_default_path and data_dir_path == args.distorted_dir : 
                try:
                    dummy_img_name_in = "dummy_eval_in.png"
                    dummy_img_name_gt = "dummy_eval_gt.png" 

                    dummy_distorted_path = os.path.join(args.distorted_dir, dummy_img_name_in)
                    if not os.path.exists(dummy_distorted_path):
                        dummy_array_dist = np.random.randint(0, 150, (280, 320, 3), dtype=np.uint8)
                        cv2.imwrite(dummy_distorted_path, dummy_array_dist)
                        print(f"Đã tạo ảnh dummy: {dummy_distorted_path}")
                    
                    dummy_gt_path_option1 = os.path.join(args.gt_dir, dummy_img_name_gt)
                    dummy_gt_path_option2 = os.path.join(args.gt_dir, dummy_img_name_in) 

                    if not os.path.exists(dummy_gt_path_option1) and not os.path.exists(dummy_gt_path_option2):
                        dummy_array_gt = np.random.randint(100, 255, (280, 320, 3), dtype=np.uint8)
                        cv2.imwrite(dummy_gt_path_option1, dummy_array_gt)
                        print(f"Đã tạo ảnh dummy GT: {dummy_gt_path_option1}")

                except Exception as e:
                    print(f"Lỗi khi tạo ảnh dummy: {e}")


    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    print(f"Đã bật tối ưu hóa biểu đồ ONNX Runtime: {sess_options.graph_optimization_level}")



    providers = ['CPUExecutionProvider']
    if not args.cpu and torch.cuda.is_available() and onnxruntime.get_device() == 'GPU':
        providers.insert(0, 'CUDAExecutionProvider')
        print("Sử dụng CUDAExecutionProvider cho ONNX Runtime.")
    else:
        if args.cpu:
            print("Buộc sử dụng CPUExecutionProvider cho ONNX Runtime theo yêu cầu.")
        else:
            print("Sử dụng CPUExecutionProvider cho ONNX Runtime (CUDA không khả dụng hoặc không được yêu cầu).")
        if args.fp16:
            print("Cảnh báo: Lợi ích của FP16 trên CPU có thể hạn chế.")


    print(f"Đang tải model ONNX cho gcnet từ: {args.gc}")
    ort_session1 = onnxruntime.InferenceSession(args.gc,  providers=providers)
    print(f"Đang tải model ONNX cho drnet từ: {args.dr}")
    ort_session2 = onnxruntime.InferenceSession(args.dr,  providers=providers)
    
    print(f"\nBắt đầu đánh giá mô hình ONNX ({'FP16' if args.fp16 else 'FP32'}) với ảnh ground truth...")
    print(f"Thư mục ảnh đầu vào: {args.distorted_dir}")
    print(f"Thư mục ảnh ground truth: {args.gt_dir}")
    if args.limit:
        print(f"Giới hạn xử lý: {args.limit} ảnh.")

    if args.resize:
            print(f"Ảnh đầu vào sẽ được giới hạn kích thước tối đa một chiều là: {args.resize} pixels.")
    
    avg_psnr, avg_ssim = evaluate_onnx_models_with_gt(
        ort_session1,
        ort_session2,
        args.distorted_dir,
        args.gt_dir,
        use_fp16=args.fp16,
        limit_images=args.limit,
        max_side_len=args.resize
    )

    print("\n--- Kết quả đánh giá (ONNX models vs Ground Truth) ---")
    print(f"PSNR trung bình: {avg_psnr:.4f} dB")
    print(f"SSIM trung bình: {avg_ssim:.4f}")
    print("-----------------------------------------------------------")

if __name__ == '__main__':
    main()
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import onnx
import onnxruntime 
from onnx_tf.backend import prepare as onnx_tf_prepare
import tensorflow as tf
from tqdm import tqdm
import argparse
import shutil

from metrics_pytorch import calculate_metrics_numpy, stride_integral

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available for PyTorch: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  PyTorch CUDA version: {torch.version.cuda}")
    print(f"  Number of GPUs PyTorch sees: {torch.cuda.device_count()}")
    print(f"  Current PyTorch GPU: {torch.cuda.current_device()}")
    print(f"  GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

print(f"ONNX Runtime version: {onnxruntime.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"  TensorFlow is built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"  TensorFlow GPUs available: {tf.config.list_physical_devices('GPU')}")
print(f"ONNX version: {onnx.__version__}")
# print(f"ONNX-TF version: {onnx_tf.__version__}") # onnx_tf không có __version__ trực tiếp

ALLOWED_INPUT_EXTENSIONS_EVAL = ('.png', '.jpg', '.jpeg', '.bmp')
ALLOWED_GT_EXTENSIONS_EVAL = ('.png', '.jpg', '.jpeg', '.bmp')

def convert_onnx_to_tflite(onnx_path: str, tflite_path: str, use_fp16: bool = False):
    """
    Chuyển đổi mô hình ONNX sang TFLite.
    Nếu use_fp16=True, mô hình TFLite sẽ được lượng tử hóa sang FP16.
    """
    print(f"Bắt đầu chuyển đổi {onnx_path} sang {tflite_path} (FP16: {use_fp16})...")
    try:
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx) 
        print(f"  Đã tải và kiểm tra thành công model ONNX: {onnx_path}")

        saved_model_dir = "temp_saved_model_dir_for_tflite"
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir) 
        os.makedirs(saved_model_dir, exist_ok=True)

        tf_rep = onnx_tf_prepare(model_onnx)
        print(f"  Đã chuẩn bị xong TF representation từ ONNX.")
        tf_rep.export_graph(saved_model_dir)
        print(f"  Đã export TF graph sang SavedModel tại: {saved_model_dir}")

        # Chuyển đổi SavedModel sang TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

        if use_fp16:
            print("  Áp dụng tối ưu hóa FP16 cho TFLite.")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        else:
            print("  Không áp dụng tối ưu hóa FP16 (sử dụng FP32).")

        tflite_model = converter.convert()
        print(f"  Đã chuyển đổi thành công sang TFLite model (in-memory).")

        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Đã lưu mô hình TFLite tại: {tflite_path}")

    except Exception as e:
        print(f"Lỗi trong quá trình chuyển đổi {onnx_path} sang TFLite: {e}")
        if os.path.exists(tflite_path): 
            os.remove(tflite_path)
        raise
    finally:
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)
            print(f"  Đã dọn dẹp thư mục tạm: {saved_model_dir}")


def evaluate_tflite_models_with_gt(
    interpreter1, 
    interpreter2, 
    distorted_img_folder: str,
    gt_img_folder: str,
    limit_images: int = None,
    max_side_len: int = None
):
    
    input_details1 = interpreter1.get_input_details()
    output_details1 = interpreter1.get_output_details()
    interpreter1.allocate_tensors() 

    input_details2 = interpreter2.get_input_details()
    output_details2 = interpreter2.get_output_details()
    interpreter2.allocate_tensors() 

    
    gcnet_input_dtype = input_details1[0]['dtype'] 
    drnet_input_dtype = input_details2[0]['dtype'] 

    print(f"TFLite Model 1 (gcnet) input dtype: {gcnet_input_dtype}")
    print(f"TFLite Model 2 (drnet) input dtype: {drnet_input_dtype}")
    
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

    for distorted_path in tqdm(distorted_image_paths, desc="Evaluating TFLite Models"):
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
        
        # Chuẩn bị input cho model 1 (gcnet - FP32)
        im_for_model1_np_float32 = (im_padded_cv.transpose(2, 0, 1).astype(np.float32) / 255.0)
        im_for_model1_np_float32 = np.expand_dims(im_for_model1_np_float32, axis=0).astype(gcnet_input_dtype) 

        im_org_padded_np_float32 = (im_padded_cv.transpose(2, 0, 1).astype(np.float32) / 255.0)
        im_org_padded_np_float32 = np.expand_dims(im_org_padded_np_float32, axis=0) # [1, C, H, W] float32

        # Model 1 (gcnet) inference
        interpreter1.set_tensor(input_details1[0]['index'], im_for_model1_np_float32)
        interpreter1.invoke()
        shadow_tflite_output = interpreter1.get_tensor(output_details1[0]['index']) 

        # Xử lý output của model 1 (tương tự code ONNX)
        shadow_torch = torch.from_numpy(shadow_tflite_output.astype(np.float32)) 
        shadow_resized_torch = F.interpolate(shadow_torch, (h_padded, w_padded), mode='bilinear', align_corners=False)
        
        im_org_for_calc_torch = torch.from_numpy(im_org_padded_np_float32.astype(np.float32))
        model1_im_torch = torch.clamp(im_org_for_calc_torch / shadow_resized_torch, 0, 1) 
        
        model1_im_np_float32 = model1_im_torch.numpy() 

     
        input_model2_np_float32 = np.concatenate((im_org_padded_np_float32, model1_im_np_float32), axis=1)
        
        input_model2_tflite_input = input_model2_np_float32.astype(drnet_input_dtype)

        # Model 2 (drnet) inference
        interpreter2.set_tensor(input_details2[0]['index'], input_model2_tflite_input)
        interpreter2.invoke()
        pred_tflite_output = interpreter2.get_tensor(output_details2[0]['index']) 

        
        pred_out_np = np.transpose(pred_tflite_output[0], (1, 2, 0)) # (H, W, C)
        pred_out_np = (pred_out_np.astype(np.float32) * 255) 
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
    parser = argparse.ArgumentParser(description="Chuyển đổi ONNX sang TFLite và đánh giá (gcnet, drnet) tính PSNR/SSIM.")
    parser.add_argument("--gc_onnx", required=True, help="Đường dẫn đến file gcnet.onnx (model1)")
    parser.add_argument("--dr_onnx", required=True, help="Đường dẫn đến file drnet.onnx (model2)")
    parser.add_argument("--distorted_dir", default="./distorted_eval_data/",
                        help="Thư mục chứa ảnh đầu vào (distorted). Mặc định: ./distorted_eval_data/")
    parser.add_argument("--gt_dir", default="./gt_eval_data/",
                        help="Thư mục chứa ảnh ground truth. Mặc định: ./gt_eval_data/")
    parser.add_argument("--limit", type=int, default=None,
                        help="Giới hạn số lượng ảnh để đánh giá (để test nhanh). Mặc định: không giới hạn.")
    parser.add_argument("--resize", type=int, default=None,
                        help="Kích thước tối đa cho một chiều của ảnh đầu vào (pixel). Nếu vượt quá, ảnh sẽ được resize giữ tỉ lệ. Mặc định: không resize.")
    parser.add_argument("--skip_conversion", action='store_true',
                        help="Bỏ qua bước chuyển đổi ONNX sang TFLite (nếu file TFLite đã tồn tại).")

    args = parser.parse_args()

    for model_path_key in ["gc_onnx", "dr_onnx"]:
        model_path = getattr(args, model_path_key)
        if not os.path.exists(model_path):
            print(f"Lỗi: Không tìm thấy file model ONNX: {model_path}")
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

    # --- Chuyển đổi ONNX sang TFLite ---
    gcnet_tflite_path = args.gc_onnx.replace(".onnx", "_fp32.tflite")
    drnet_tflite_path = args.dr_onnx.replace(".onnx", "_fp16.tflite")

    if not args.skip_conversion or not os.path.exists(gcnet_tflite_path):
        convert_onnx_to_tflite(args.gc_onnx, gcnet_tflite_path, use_fp16=False) # gcnet -> FP32
    else:
        print(f"Bỏ qua chuyển đổi {args.gc_onnx}, sử dụng file đã có: {gcnet_tflite_path}")

    if not args.skip_conversion or not os.path.exists(drnet_tflite_path):
        convert_onnx_to_tflite(args.dr_onnx, drnet_tflite_path, use_fp16=True)  # drnet -> FP16
    else:
        print(f"Bỏ qua chuyển đổi {args.dr_onnx}, sử dụng file đã có: {drnet_tflite_path}")

    if not os.path.exists(gcnet_tflite_path) or not os.path.exists(drnet_tflite_path):
        print("Lỗi: Một hoặc cả hai file TFLite không được tạo/tìm thấy. Kết thúc chương trình.")
        return

    # --- Tải mô hình TFLite và đánh giá ---
    print(f"\nĐang tải model TFLite cho gcnet từ: {gcnet_tflite_path}")
    interpreter_gcnet = tf.lite.Interpreter(model_path=gcnet_tflite_path)
    
    print(f"Đang tải model TFLite cho drnet từ: {drnet_tflite_path}")
    interpreter_drnet = tf.lite.Interpreter(model_path=drnet_tflite_path)
    
    print(f"\nBắt đầu đánh giá mô hình TFLite (gcnet_fp32, drnet_fp16) với ảnh ground truth...")
    print(f"Thư mục ảnh đầu vào: {args.distorted_dir}")
    print(f"Thư mục ảnh ground truth: {args.gt_dir}")
    if args.limit:
        print(f"Giới hạn xử lý: {args.limit} ảnh.")
    if args.resize:
        print(f"Ảnh đầu vào sẽ được giới hạn kích thước tối đa một chiều là: {args.resize} pixels.")
    
    avg_psnr, avg_ssim = evaluate_tflite_models_with_gt(
        interpreter_gcnet,
        interpreter_drnet,
        args.distorted_dir,
        args.gt_dir,
        limit_images=args.limit,
        max_side_len=args.resize
    )

    print("\n--- Kết quả đánh giá (TFLite models vs Ground Truth) ---")
    print(f"PSNR trung bình: {avg_psnr:.4f} dB")
    print(f"SSIM trung bình: {avg_ssim:.4f}")
    print("-----------------------------------------------------------")

if __name__ == '__main__':
    main()
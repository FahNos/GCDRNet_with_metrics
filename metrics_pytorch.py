import torch
import torch.nn.functional as F
import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio as psnr
import skimage
import argparse 

try:
    from utils import convert_state_dict
except ImportError:
    print("Cảnh báo: không tìm thấy 'utils.convert_state_dict'. Đảm bảo file utils.py có trong PYTHONPATH hoặc cùng thư mục.")
    def convert_state_dict(state): return state

try:
    from data.preprocess.crop_merge_image import stride_integral
except ImportError:
    print("Cảnh báo: không tìm thấy 'data.preprocess.crop_merge_image.stride_integral'.")
    def stride_integral(image_array, stride_val=32): # Thêm stride_val cho phù hợp hơn
        h, w = image_array.shape[:2]
        ph = ((h - 1) // stride_val + 1) * stride_val
        pw = ((w - 1) // stride_val + 1) * stride_val
        padding_h = ph - h
        padding_w = pw - w
        if padding_h > 0 or padding_w > 0:
            image_array = cv2.copyMakeBorder(image_array, 0, padding_h, 0, padding_w, cv2.BORDER_REFLECT)
        return image_array, 0, 0 

if os.path.exists('./models/UNeXt'):
    os.sys.path.append('./models/UNeXt')
    try:
        from models.UNeXt.unext import UNext_full_resolution_padding, UNext_full_resolution_padding_L_py_L
    except ImportError:
        print("Lỗi: Không thể import UNext từ './models/UNeXt'. Vui lòng kiểm tra lại cấu trúc thư mục và file __init__.py.")
        class UNext_full_resolution_padding(torch.nn.Module):
            def __init__(self, *args, **kwargs): super().__init__(); self.fc = torch.nn.Linear(1,1)
            def forward(self, x): return x
        class UNext_full_resolution_padding_L_py_L(torch.nn.Module):
            def __init__(self, *args, **kwargs): super().__init__(); self.fc = torch.nn.Linear(1,1)
            def forward(self, x): return x,x,x,x
else:
    print("Cảnh báo: Thư mục './models/UNeXt' không tồn tại. Không thể import UNext models.")
    class UNext_full_resolution_padding(torch.nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.fc = torch.nn.Linear(1,1)
        def forward(self, x): return x
    class UNext_full_resolution_padding_L_py_L(torch.nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.fc = torch.nn.Linear(1,1)
        def forward(self, x): return x,x,x,x


def calculate_metrics_numpy(img_true_np, img_pred_np):
    if img_true_np.shape != img_pred_np.shape:
        img_pred_np = cv2.resize(img_pred_np, (img_true_np.shape[1], img_true_np.shape[0]),
                                 interpolation=cv2.INTER_CUBIC)

    img_true_np = np.clip(img_true_np, 0, 255).astype(np.uint8)
    img_pred_np = np.clip(img_pred_np, 0, 255).astype(np.uint8)

    if np.array_equal(img_true_np, img_pred_np):
        psnr_val = float('inf')
    elif img_true_np.size == 0 or img_pred_np.size == 0:
        print(f"Cảnh báo: Ảnh rỗng (true_shape={img_true_np.shape}, pred_shape={img_pred_np.shape}). PSNR/SSIM sẽ là 0.")
        return 0.0, 0.0
    else:
        try:
            psnr_val = psnr(img_true_np, img_pred_np, data_range=255)
        except Exception as e_psnr:
            print(f"Lỗi khi tính PSNR: {e_psnr}. Ảnh true shape: {img_true_np.shape}, pred shape: {img_pred_np.shape}. Đặt PSNR = 0.")
            psnr_val = 0.0

    ssim_val = 0.0
    min_dimension_true = min(img_true_np.shape[0], img_true_np.shape[1])
    min_dimension_pred = min(img_pred_np.shape[0], img_pred_np.shape[1])
    
    if min_dimension_true < 3 or min_dimension_pred < 3:
        return psnr_val, 0.0

    current_win_size = min(7, min_dimension_true, min_dimension_pred)
    if current_win_size % 2 == 0:
        current_win_size -= 1
    current_win_size = max(3, current_win_size)

    try:
        if img_true_np.ndim == 3 and img_true_np.shape[2] == 3:
            try:
                ssim_val = structural_similarity(img_true_np, img_pred_np,
                                                 channel_axis=-1,
                                                 data_range=255,
                                                 win_size=current_win_size,
                                                 gaussian_weights=True)
            except TypeError: 
                ssim_val = structural_similarity(img_true_np, img_pred_np,
                                                 multichannel=True,
                                                 data_range=255,
                                                 win_size=current_win_size,
                                                 gaussian_weights=True)
        elif img_true_np.ndim == 2:
            ssim_val = structural_similarity(img_true_np, img_pred_np,
                                             data_range=255,
                                             win_size=current_win_size,
                                             gaussian_weights=True)
        else:
            print(f"Cảnh báo: Định dạng ảnh không được hỗ trợ cho SSIM. Shape: {img_true_np.shape}. SSIM sẽ là 0.")
            ssim_val = 0.0
    except ValueError as ve:
        if "win_size exceeds image extent" in str(ve):
            print(f"Cảnh báo SSIM: Kích thước ảnh (true: {img_true_np.shape[0]}x{img_true_np.shape[1]}, pred: {img_pred_np.shape[0]}x{img_pred_np.shape[1]}) quá nhỏ cho win_size={current_win_size}. Lỗi: {ve}. SSIM sẽ là 0.")
        else:
            print(f"Lỗi ValueError không mong muốn khi tính SSIM (win_size={current_win_size}): {ve}. Ảnh shape: {img_true_np.shape}. SSIM sẽ là 0.")
        ssim_val = 0.0
    except Exception as e:
        print(f"Lỗi không mong muốn khi tính SSIM (win_size={current_win_size}): {e}. Ảnh shape: {img_true_np.shape}. SSIM sẽ là 0.")
        ssim_val = 0.0
    return psnr_val, ssim_val

def evaluate_pytorch_model_with_gt(
    model1, model2,
    distorted_img_folder,
    gt_img_folder,
    limit_images=None,
    device='cuda',
    max_side_len=None 
):
    ALLOWED_INPUT_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')
    distorted_image_paths_all = glob.glob(os.path.join(distorted_img_folder, '*'))

    distorted_image_paths = []
    for p in distorted_image_paths_all:
        if p.lower().endswith(ALLOWED_INPUT_EXTENSIONS):
            distorted_image_paths.append(p)
    distorted_image_paths.sort()

    if limit_images is not None and limit_images > 0:
        distorted_image_paths = distorted_image_paths[:limit_images]

    if not distorted_image_paths:
        print(f"Không tìm thấy ảnh nào có đuôi hợp lệ ({ALLOWED_INPUT_EXTENSIONS}) trong {distorted_img_folder}")
        return 0, 0

    total_psnr = 0
    total_ssim = 0
    count = 0
    model1.eval()
    model2.eval()
    ALLOWED_GT_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

    for distorted_path in tqdm(distorted_image_paths, desc="Evaluating PyTorch Model"):
        distorted_filename_full = os.path.basename(distorted_path)
        gt_path_found = None
        if "_in." in distorted_filename_full:
            try:
                base_name_part = distorted_filename_full.rsplit('_in.', 1)[0]
                for ext_gt in ALLOWED_GT_EXTENSIONS:
                    potential_gt_filename = f"{base_name_part}_gt{ext_gt}"
                    potential_gt_path = os.path.join(gt_img_folder, potential_gt_filename)
                    if os.path.exists(potential_gt_path):
                        gt_path_found = potential_gt_path
                        break
            except IndexError:
                print(f"Cảnh báo: Tên file '{distorted_filename_full}' có '_in.' nhưng không theo định dạng chuẩn. Thử tìm tên gốc.")
        if gt_path_found is None:
            potential_gt_path = os.path.join(gt_img_folder, distorted_filename_full)
            if os.path.exists(potential_gt_path):
                gt_path_found = potential_gt_path
        if gt_path_found is None:
            expected_gt_name_pattern = distorted_filename_full
            if "_in." in distorted_filename_full:
                try:
                    base_name_part = distorted_filename_full.rsplit('_in.', 1)[0]
                    expected_gt_name_pattern = f"{base_name_part}_gt.<ext>"
                except IndexError: pass
            print(f"Cảnh báo: Không tìm thấy GT cho {distorted_path} (đã tìm '{expected_gt_name_pattern}' và tên gốc). Bỏ qua.")
            continue

        im_distorted_cv = cv2.imread(distorted_path)
        if im_distorted_cv is None:
            print(f"Lỗi đọc ảnh distorted: '{distorted_path}'. Bỏ qua.")
            continue

        if max_side_len is not None and max_side_len > 0:
            h_orig, w_orig = im_distorted_cv.shape[:2]
            if h_orig > max_side_len or w_orig > max_side_len:
                scale = float(max_side_len) / max(h_orig, w_orig)
                new_w = int(w_orig * scale)
                new_h = int(h_orig * scale)
                if new_w > 0 and new_h > 0: # Đảm bảo kích thước mới hợp lệ
                    im_distorted_cv = cv2.resize(im_distorted_cv, (new_w, new_h), interpolation=cv2.INTER_AREA) 
                    # print(f"Resized distorted image {distorted_filename_full} from ({h_orig},{w_orig}) to ({new_h},{new_w}) for evaluation.")
                else:
                    print(f"Cảnh báo: Kích thước mới không hợp lệ ({new_h},{new_w}) cho ảnh {distorted_filename_full}. Bỏ qua resize.")

        im_gt_cv = cv2.imread(gt_path_found)
        if im_gt_cv is None:
            print(f"Lỗi đọc ảnh GT: '{gt_path_found}'. Bỏ qua.")
            continue

        im_padded_cv, padding_h_top, padding_w_left = stride_integral(im_distorted_cv) 
        h_padded, w_padded = im_padded_cv.shape[:2]
        
        padding_h_bottom = h_padded - im_distorted_cv.shape[0] - padding_h_top
        padding_w_right = w_padded - im_distorted_cv.shape[1] - padding_w_left


        with torch.no_grad():
            im_tensor_for_m1 = torch.from_numpy(im_padded_cv.transpose(2,0,1)/255.0).unsqueeze(0).float().to(device)
            im_org_tensor = torch.from_numpy(im_padded_cv.transpose(2,0,1)/255.0).unsqueeze(0).float().to(device)

            shadow = model1(im_tensor_for_m1)
            print(f"- shadow shape = {shadow.shape}")
            # shadow = F.interpolate(shadow, (h_padded, w_padded), mode='bilinear', align_corners=False)

            model1_im_tensor = torch.clamp(im_org_tensor / shadow, 0, 1)
            outputs_model2 = model2(torch.cat((im_org_tensor, model1_im_tensor), 1))
            if isinstance(outputs_model2, tuple) and len(outputs_model2) >= 1:
                pred_tensor = outputs_model2[0]
            else:
                pred_tensor = outputs_model2

            pred_out_np = pred_tensor[0].permute(1,2,0).data.cpu().numpy()
            pred_out_np = (pred_out_np * 255)
            pred_out_np = np.clip(pred_out_np, 0, 255).astype(np.uint8)
            
            if padding_h_bottom > 0 and padding_w_right > 0:
                 pred_out_np = pred_out_np[padding_h_top : -padding_h_bottom, padding_w_left : -padding_w_right]
            elif padding_h_bottom > 0:
                 pred_out_np = pred_out_np[padding_h_top : -padding_h_bottom, padding_w_left:]
            elif padding_w_right > 0:
                 pred_out_np = pred_out_np[padding_h_top:, padding_w_left : -padding_w_right]
            else: 
                 pred_out_np = pred_out_np[padding_h_top:, padding_w_left:]


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


def test_model1_model2_original(model1, model2, path_list, in_folder, sav_folder, device_to_use, max_side_len=None): # Thêm max_side_len
    model1.eval()
    model2.eval()
    for im_path in tqdm(path_list, desc="Running original inference for saving"):
        im_org_cv = cv2.imread(im_path)
        if im_org_cv is None:
            print(f"Warning: Could not read image {im_path}. Skipping.")
            continue

        if max_side_len is not None and max_side_len > 0:
            h_orig, w_orig = im_org_cv.shape[:2]
            if h_orig > max_side_len or w_orig > max_side_len:
                scale = float(max_side_len) / max(h_orig, w_orig)
                new_w = int(w_orig * scale)
                new_h = int(h_orig * scale)
                if new_w > 0 and new_h > 0:
                    im_org_cv = cv2.resize(im_org_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    # print(f"Resized image {os.path.basename(im_path)} from ({h_orig},{w_orig}) to ({new_h},{new_w}) for saving.")
                else:
                    print(f"Cảnh báo: Kích thước mới không hợp lệ ({new_h},{new_w}) cho ảnh {os.path.basename(im_path)}. Bỏ qua resize.")

        im_padded_cv, padding_h_top, padding_w_left = stride_integral(im_org_cv)
        h_padded, w_padded = im_padded_cv.shape[:2]
        
        padding_h_bottom = h_padded - im_org_cv.shape[0] - padding_h_top
        padding_w_right = w_padded - im_org_cv.shape[1] - padding_w_left

        with torch.no_grad():
            im_tensor = torch.from_numpy(im_padded_cv.transpose(2,0,1)/255.0).unsqueeze(0).float().to(device_to_use)
            im_org_tensor = torch.from_numpy(im_padded_cv.transpose(2,0,1)/255.0).unsqueeze(0).float().to(device_to_use)

            shadow_pred_tensor = model1(im_tensor)
            shadow_pred_tensor = F.interpolate(shadow_pred_tensor, (h_padded, w_padded), mode='bilinear', align_corners=False)

            model1_im_output_tensor = torch.clamp(im_org_tensor / shadow_pred_tensor, 0, 1)

            outputs_model2 = model2(torch.cat((im_org_tensor, model1_im_output_tensor), 1))
            if isinstance(outputs_model2, tuple) and len(outputs_model2) >= 1:
                final_pred_tensor = outputs_model2[0]
            else:
                final_pred_tensor = outputs_model2

            pred_out_final_np = final_pred_tensor[0].permute(1,2,0).data.cpu().numpy()
            pred_out_final_np = (pred_out_final_np*255)
            pred_out_final_np = np.clip(pred_out_final_np, 0, 255).astype(np.uint8)
            
            if padding_h_bottom > 0 and padding_w_right > 0:
                 pred_out_final_np = pred_out_final_np[padding_h_top : -padding_h_bottom, padding_w_left : -padding_w_right]
            elif padding_h_bottom > 0:
                 pred_out_final_np = pred_out_final_np[padding_h_top : -padding_h_bottom, padding_w_left:]
            elif padding_w_right > 0:
                 pred_out_final_np = pred_out_final_np[padding_h_top:, padding_w_left : -padding_w_right]
            else:
                 pred_out_final_np = pred_out_final_np[padding_h_top:, padding_w_left:]

        save_path = im_path.replace(in_folder, sav_folder)
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(save_path, pred_out_final_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Đánh giá và/hoặc chạy inference với mô hình PyTorch GCDRNet.")
    parser.add_argument("--model1_path", type=str, default='./pretrained_model/gcnet.pkl',
                        help="Đường dẫn đến file checkpoint của model1 (gcnet).")
    parser.add_argument("--model2_path", type=str, default='./pretrained_model/drnet.pkl',
                        help="Đường dẫn đến file checkpoint của model2 (drnet).")
    parser.add_argument("--distorted_dir", type=str, default='./distorted_eval_data/',
                        help="Thư mục chứa ảnh đầu vào (distorted) để đánh giá.")
    parser.add_argument("--gt_dir", type=str, default='./gt_eval_data/',
                        help="Thư mục chứa ảnh ground truth để đánh giá.")
    parser.add_argument("--limit_eval", type=int, default=None,
                        help="Giới hạn số lượng ảnh để đánh giá (để test nhanh).")
    parser.add_argument("--size", type=int, default=None,
                        help="Kích thước tối đa cho một chiều của ảnh đầu vào (pixel). Nếu vượt quá, ảnh sẽ được resize giữ tỉ lệ. Mặc định: không resize.")
    parser.add_argument("--save_output", action='store_true',
                        help="Chạy inference và lưu ảnh kết quả (không đánh giá metrics).")
    parser.add_argument("--save_input_dir", type=str, default='./eval/distorted/',
                        help="Thư mục chứa ảnh đầu vào khi chạy inference để lưu kết quả.")
    parser.add_argument("--save_output_dir", type=str, default='./eval/enhanced/',
                        help="Thư mục để lưu ảnh kết quả khi chạy inference.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng device: {device}")

    model1_path_config = args.model1_path
    model2_path_config = args.model2_path

    if not os.path.exists(model1_path_config):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint cho model1: {model1_path_config}")
    if not os.path.exists(model2_path_config):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint cho model2: {model2_path_config}")

    model1_eval = UNext_full_resolution_padding(num_classes=3, input_channels=3, img_size=512).to(device)
    state1_eval = convert_state_dict(torch.load(model1_path_config, map_location=device)['model_state'])
    model1_eval.load_state_dict(state1_eval)

    model2_eval = UNext_full_resolution_padding_L_py_L(num_classes=3, input_channels=6, img_size=512).to(device)
    state2_eval = convert_state_dict(torch.load(model2_path_config, map_location=device)['model_state'])
    model2_eval.load_state_dict(state2_eval)

    model1_eval.eval()
    model2_eval.eval()
    print(f"Đã tải xong mô hình PyTorch từ {model1_path_config} và {model2_path_config}.")

    if args.save_output:
        img_folder_orig_infer = args.save_input_dir
        sav_folder_orig_infer = args.save_output_dir
        os.makedirs(img_folder_orig_infer, exist_ok=True) # Đảm bảo thư mục input tồn tại
        os.makedirs(sav_folder_orig_infer, exist_ok=True)

        if not glob.glob(os.path.join(img_folder_orig_infer, '*')) and not os.path.exists(os.path.join(img_folder_orig_infer, "dummy_distorted_for_save.png")):
            print(f"Tạo ảnh giả trong {img_folder_orig_infer} để chạy test_model1_model2_original.")
            dummy_distorted_save = np.random.randint(0, 150, (280, 280, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(img_folder_orig_infer, "dummy_distorted_for_save.png"), dummy_distorted_save)

        im_paths_orig_infer = glob.glob(os.path.join(img_folder_orig_infer,'*.[jp][pn]g')) + \
                              glob.glob(os.path.join(img_folder_orig_infer,'*.jpeg')) + \
                              glob.glob(os.path.join(img_folder_orig_infer,'*.bmp'))
        if im_paths_orig_infer:
            print(f"\nChạy inference gốc và lưu kết quả từ {img_folder_orig_infer} vào {sav_folder_orig_infer}...")
            if args.size:
                 print(f"Ảnh đầu vào cho việc lưu sẽ được giới hạn kích thước tối đa một chiều là: {args.size} pixels.")
            test_model1_model2_original(
                model1_eval, model2_eval,
                im_paths_orig_infer,
                img_folder_orig_infer,
                sav_folder_orig_infer,
                device,
                max_side_len=args.size 
            )
            print("Đã hoàn thành inference gốc và lưu kết quả.")
        else:
            print(f"Không có ảnh trong {img_folder_orig_infer} để chạy inference lưu trữ.")
    else:
        distorted_eval_dir = args.distorted_dir
        gt_eval_dir = args.gt_dir
        LIMIT_IMAGES_FOR_EVAL_CONFIG = args.limit_eval

        os.makedirs(distorted_eval_dir, exist_ok=True)
        os.makedirs(gt_eval_dir, exist_ok=True)

        if not glob.glob(os.path.join(distorted_eval_dir, '*.[jp][pn]g')) and \
           not glob.glob(os.path.join(distorted_eval_dir, '*.jpeg')) and \
           not glob.glob(os.path.join(distorted_eval_dir, '*.bmp')):
            print(f"Tạo ảnh giả trong {distorted_eval_dir} và {gt_eval_dir} để thử nghiệm đánh giá.")
            for i in range(3):
                dummy_h, dummy_w = np.random.randint(200,301), np.random.randint(200,301)
                dummy_distorted = np.random.randint(0, 150, (dummy_h, dummy_w, 3), dtype=np.uint8)
                dummy_gt = np.random.randint(100, 255, (dummy_h, dummy_w, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(distorted_eval_dir, f"dummy_test_image_{i+1}_in.png"), dummy_distorted)
                cv2.imwrite(os.path.join(gt_eval_dir, f"dummy_test_image_{i+1}_gt.png"), dummy_gt) 

        print("\nBắt đầu đánh giá mô hình PyTorch (.pkl) với ảnh ground truth...")
        if args.size:
            print(f"Ảnh đầu vào sẽ được giới hạn kích thước tối đa một chiều là: {args.size} pixels.")

        avg_psnr_res, avg_ssim_res = evaluate_pytorch_model_with_gt(
            model1_eval, model2_eval,
            distorted_eval_dir,
            gt_eval_dir,
            limit_images=LIMIT_IMAGES_FOR_EVAL_CONFIG,
            device=device,
            max_side_len=args.size 
        )

        print("\n--- Kết quả đánh giá (PyTorch .pkl model vs Ground Truth) ---")
        print(f"PSNR trung bình: {avg_psnr_res:.4f} dB")
        print(f"SSIM trung bình: {avg_ssim_res:.4f}")
        print("-----------------------------------------------------------")
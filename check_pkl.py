import torch
import os
import argparse 

def check_pkl_file(file_path):
  
    if not os.path.exists(file_path):
        print(f"Lỗi: File không tồn tại tại đường dẫn '{file_path}'.")
        print("Vui lòng kiểm tra lại đường dẫn và đảm bảo file có tồn tại.")
        return

    try:        
        state_dict = torch.load(file_path, map_location='cpu')

        print(f"Kiểm tra file: {file_path}")
        print("---")
        print("Các key (tên tensor) trong state_dict (hiển thị tối đa 20 key đầu tiên):")
        
        for i, k in enumerate(list(state_dict.keys())):
            if i >= 20:
                break
            print(f"    {k}")

        print("---")
        print(f"Tổng số key trong state_dict: {len(state_dict)}")
        print("File .pkl đã được tải và kiểm tra thành công!")

    except Exception as e:
        print(f"Đã xảy ra lỗi khi tải state_dict bằng torch.load(): {e}")
        print("Có thể file không phải là một state_dict hợp lệ của PyTorch hoặc bị hỏng.")
        print("Hãy đảm bảo rằng file .pkl được lưu đúng định dạng của PyTorch (torch.save).")

def check_structure(file_path):
  
    if not os.path.exists(file_path):
        print(f"Lỗi: File không tồn tại tại đường dẫn '{file_path}'.")
    else:
        try:
            state_dict = torch.load(file_path, map_location='cpu')

            print("Các key (tên tensor) trong state_dict:")
            for k in state_dict.keys():
                print(f"  - {k}")

            print("\n--- Chi tiết nội dung của từng key ---")

            if 'epoch' in state_dict:
                print("\nNội dung của 'epoch':")
                print(f"  {state_dict['epoch']}")
                if isinstance(state_dict['epoch'], torch.Tensor):
                    print(f"  Giá trị số của epoch: {state_dict['epoch'].item()}")
            else:
                print("\nKhông tìm thấy key 'epoch' trong state_dict.")

            if 'model_state' in state_dict:           
                for model_k, model_v in state_dict['model_state'].items():
                    print(f"  - {model_k}: {model_v.shape} (tensor of shape) - Type: {model_v.dtype}")
                    
                print(f"Tổng số key trong 'model_state': {len(state_dict['model_state'])}")
            else:
                print("\nKhông tìm thấy key 'model_state' trong state_dict.")

            if 'optimizer_state' in state_dict:           
                for opt_k, opt_v in state_dict['optimizer_state'].items():
                    print(f"  - {opt_k}: {type(opt_v)}")
                    if isinstance(opt_v, dict) and 'param_groups' in opt_v:
                        print(f"    Param groups trong optimizer: {opt_v['param_groups']}")
                    if isinstance(opt_v, dict) and 'state' in opt_v:
                        print(f"    Số lượng state của tham số trong optimizer: {len(opt_v['state'])}")
            else:
                print("\nKhông tìm thấy key 'optimizer_state' trong state_dict.")

        except Exception as e:
            print(f"Đã xảy ra lỗi khi tải hoặc xử lý state_dict: {e}")
        print("Có thể file không phải là một state_dict hợp lệ của PyTorch hoặc bị hỏng.")

def check_weight(file_path):
    if not os.path.exists(file_path):
        print(f"Lỗi: File không tồn tại tại đường dẫn '{file_path}'.")
    else:
        try:
            state_dict = torch.load(file_path, map_location='cpu')

            print("--- Chi tiết nội dung của 'model_state' ---")

            if 'model_state' in state_dict:
                for model_k, model_v in state_dict['model_state'].items():
                    print(f"  - Key: {model_k}")
                    if isinstance(model_v, torch.Tensor):
                        print(f"    - Kiểu: PyTorch Tensor")
                        print(f"    - Shape: {model_v.shape}")
                        print(f"    - Kiểu dữ liệu (Dtype): {model_v.dtype}")
                        print(f"    - Thiết bị (Device): {model_v.device}")
                        print(f"    - Tổng số phần tử (Numel): {model_v.numel()}")

                        if model_k == 'module.encoder1.1.weight':
                            print(f"    - Giá trị mẫu (module.encoder1.1.weight[0, 0, 0, :5]):")
                            print(f"      {model_v[0, 0, 0, :5].tolist()}") 

                        elif model_v.numel() <= 10:
                            print(f"    - Giá trị: {model_v.tolist()}") 
                        else: 
                            print(f"    - Giá trị mẫu (model_v[0, :5] nếu có thể):")
                            try:
                                print(f"      {model_v[0, :5].tolist()}")
                            except IndexError:
                                print(f"      {model_v[:5].tolist()}")
                    else:
                        print(f"    - Kiểu: {type(model_v)}") 
                        print(f"    - Giá trị: {model_v}")

                print(f"Tổng số key trong 'model_state': {len(state_dict['model_state'])}")
            else:
                print("\nKhông tìm thấy key 'model_state' trong state_dict.")

        except Exception as e:
            print(f"Đã xảy ra lỗi khi tải hoặc xử lý state_dict: {e}")
            print("Có thể file không phải là một state_dict hợp lệ của PyTorch hoặc bị hỏng.")

def get_tensor_size_in_bytes(tensor: torch.Tensor):
    """
    Tính kích thước của một tensor PyTorch bằng byte.
    """
    if not isinstance(tensor, torch.Tensor):
        return 0 

    element_size_bytes = tensor.element_size() 
    total_bytes = tensor.numel() * element_size_bytes 
    return total_bytes

def print_model_state_memory_usage(state_dict: dict):
    """
    In ra dung lượng bộ nhớ của từng tham số và tổng dung lượng của model_state.
    """
    if 'model_state' not in state_dict:
        print("Không tìm thấy key 'model_state' trong state_dict.")
        return

    model_state = state_dict['model_state']
    total_model_state_bytes = 0

    print("\n--- Dung lượng bộ nhớ của từng tham số trong 'model_state' ---")
    for k, v in model_state.items():
        if isinstance(v, torch.Tensor):
            tensor_bytes = get_tensor_size_in_bytes(v)
            total_model_state_bytes += tensor_bytes
            size_str = ""
            if tensor_bytes < 1024:
                size_str = f"{tensor_bytes} B"
            elif tensor_bytes < (1024**2):
                size_str = f"{tensor_bytes / 1024:.2f} KB"
            elif tensor_bytes < (1024**3):
                size_str = f"{tensor_bytes / (1024**2):.2f} MB"
            else:
                size_str = f"{tensor_bytes / (1024**3):.2f} GB"

            print(f"  - {k}: {v.shape}, Dtype: {v.dtype}, Kích thước: {size_str}")
        else:
            print(f"  - {k}: (Không phải Tensor) Kiểu: {type(v)}")

    # In tổng dung lượng
    total_size_mb = total_model_state_bytes / (1024**2)
    print(f"\nTổng dung lượng bộ nhớ của 'model_state': {total_size_mb:.2f} MB ({total_model_state_bytes} bytes)")

def check_memory(file_path):
    if not os.path.exists(file_path):
        print(f"Lỗi: File không tồn tại tại đường dẫn '{file_path}'.")
    else:
        try:
            state_dict = torch.load(file_path, map_location='cpu')
            print_model_state_memory_usage(state_dict)

        except Exception as e:
            print(f"Đã xảy ra lỗi khi tải hoặc xử lý state_dict: {e}")
            print("Có thể file không phải là một state_dict hợp lệ của PyTorch hoặc bị hỏng.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kiểm tra các key trong file .pkl của PyTorch (state_dict).")

    parser.add_argument('-path', type=str, required=True,
                        help="Đường dẫn đến file .pkl cần kiểm tra.")

    args = parser.parse_args()

    check_pkl_file(args.path)
    print("--------------------------------")
    check_structure(args.path)
    print("--------------------------------")
    check_weight(args.path)
    print("--------------------------------")
    check_memory(args.path)
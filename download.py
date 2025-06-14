from huggingface_hub import hf_hub_download
import os
import shutil
from huggingface_hub import hf_hub_download
import subprocess
from tqdm import tqdm
import random  
import requests
import zipfile

def download_and_move_hf_file(repo_id: str, filename: str, target_folder: str = None, revision: str = None):    
    try:
       
        source_file_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        print(f"File '{filename}' đã được tải về cache: {source_file_path}")
        file_size_bytes = os.path.getsize(source_file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"Kích thước file đã tải: {file_size_mb:.2f} MB")
        if target_folder:
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
                print(f"Đã tạo thư mục đích: {target_folder}")
            destination_directory = target_folder
        else:
            destination_directory = '.'

        destination_file_path = os.path.join(destination_directory, filename)
        shutil.copy(source_file_path, destination_file_path)
        if os.path.exists(destination_file_path):
            print(f"File '{filename}' hiện đã có trong thư mục '{destination_directory}'.")
            return os.path.abspath(destination_file_path) # Trả về đường dẫn tuyệt đối
        else:
            print(f"Không thể di chuyển file '{filename}'.")
            return None

    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình tải hoặc di chuyển file: {e}")
        return None
    
def extract_and_remove(file_path, extract_to_dir):   
    if not os.path.exists(file_path):
        print(f"Tệp nén không tồn tại: {file_path}")
        return   
    os.makedirs(extract_to_dir, exist_ok=True)
  
    if file_path.endswith('.zip'):
        print(f"Đang giải nén ZIP {file_path} vào {extract_to_dir}...")
        shutil.unpack_archive(file_path, extract_to_dir)
    elif file_path.endswith('.tar') or file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
        print(f"Đang giải nén TAR {file_path} vào {extract_to_dir}...")
        import tarfile
        with tarfile.open(file_path) as tar:
            tar.extractall(path=extract_to_dir)
    else:
        print(f"Không hỗ trợ định dạng tệp nén: {file_path}")
        return

    print("Giải nén hoàn tất.")
   
    print(f"Đang xóa tệp nén {file_path}...")
    os.remove(file_path)
    print("Đã xóa tệp nén.")

def organize_realdae_images(source_base_dir: str, target_base_dir: str):   

    in_folder = os.path.join(target_base_dir, 'in')
    gt_folder = os.path.join(target_base_dir, 'gt')

    os.makedirs(in_folder, exist_ok=True)
    os.makedirs(gt_folder, exist_ok=True)

    print(f"Bắt đầu sắp xếp ảnh từ '{source_base_dir}'...")
    print(f"Ảnh _in sẽ được copy vào: '{in_folder}'")
    print(f"Ảnh _gt sẽ được copy vào: '{gt_folder}'")
    print("-" * 50)

    image_paths = []
    for root, _, files in os.walk(source_base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f"Không tìm thấy ảnh nào trong '{source_base_dir}'. Vui lòng kiểm tra đường dẫn.")
        return

    in_count = 0
    gt_count = 0

    for img_path in tqdm(image_paths, desc="Đang xử lý ảnh"):
        img_name = os.path.basename(img_path) 

        if img_name.endswith('_in.png') or img_name.endswith('_in.jpg') or img_name.endswith('_in.jpeg'):
            destination_path = os.path.join(in_folder, img_name)
            shutil.copy2(img_path, destination_path) 
            in_count += 1
        elif img_name.endswith('_gt.png') or img_name.endswith('_gt.jpg') or img_name.endswith('_gt.jpeg'):
            destination_path = os.path.join(gt_folder, img_name)
            shutil.copy2(img_path, destination_path)
            gt_count += 1

    print("-" * 50)
    print(f"Hoàn tất sắp xếp ảnh.")
    print(f"Số lượng ảnh '_in' đã được copy: {in_count}")
    print(f"Số lượng ảnh '_gt' đã được copy: {gt_count}")

def select_and_copy_random_images(
    source_in_dir: str,
    source_gt_dir: str,
    target_in_dir: str,
    target_gt_dir: str,
    num_samples: int
):
    os.makedirs(target_in_dir, exist_ok=True)
    os.makedirs(target_gt_dir, exist_ok=True)

    print(f"Bắt đầu chọn và sao chép ngẫu nhiên {num_samples} cặp ảnh...")
    print(f"Ảnh 'in' sẽ được copy vào: '{target_in_dir}'")
    print(f"Ảnh 'gt' sẽ được copy vào: '{target_gt_dir}'")
    print("-" * 50)

    ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

    in_image_filenames = [
        f for f in os.listdir(source_in_dir)
        if f.lower().endswith(ALLOWED_EXTENSIONS) and '_in.' in f 
    ]

    if not in_image_filenames:
        print(f"Lỗi: Không tìm thấy ảnh '_in' nào có đuôi hợp lệ trong thư mục '{source_in_dir}'.")
        return

    available_pairs = []
    for in_filename in in_image_filenames:        
        try:
            base_name_part = in_filename.rsplit('_in.', 1)[0]
        except IndexError:            
            continue

        gt_filename_prefix = base_name_part + '_gt' 

        found_gt_file = None
        for ext in ALLOWED_EXTENSIONS:
            potential_gt_filename = gt_filename_prefix + ext 
            if os.path.exists(os.path.join(source_gt_dir, potential_gt_filename)):
                found_gt_file = potential_gt_filename
                break 

        if found_gt_file:
            available_pairs.append((in_filename, found_gt_file))
       


    if not available_pairs:
        print(f"Không tìm thấy cặp ảnh (_in, _gt) hợp lệ nào trong '{source_in_dir}' và '{source_gt_dir}'.")
        return

    if num_samples > len(available_pairs):
        print(f"Cảnh báo: Số lượng mẫu yêu cầu ({num_samples}) lớn hơn số cặp có sẵn ({len(available_pairs)}).")
        print(f"Sẽ chọn tất cả {len(available_pairs)} cặp ảnh.")
        selected_pairs = available_pairs
    else:
        selected_pairs = random.sample(available_pairs, num_samples)

    copied_count = 0
    for in_filename, gt_filename in tqdm(selected_pairs, desc="Đang sao chép ảnh"):
        source_in_path = os.path.join(source_in_dir, in_filename)
        source_gt_path = os.path.join(source_gt_dir, gt_filename) 

        target_in_path = os.path.join(target_in_dir, in_filename)
        target_gt_path = os.path.join(target_gt_dir, gt_filename)

        try:
            shutil.copy2(source_in_path, target_in_path)
            shutil.copy2(source_gt_path, target_gt_path)
            copied_count += 1
        except Exception as e:
            print(f"Lỗi khi copy cặp ảnh {in_filename} và {gt_filename}: {e}")

    print("-" * 50)
    print(f"Đã sao chép thành công {copied_count} cặp ảnh ngẫu nhiên.")

def download_file_from_url(url: str, destination_path: str):    
    try:
        response = requests.head(url, allow_redirects=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte

        print(f"Đang tải file từ: {url}")
        print(f"Kích thước file ước tính: {total_size_in_bytes / (1024 * 1024):.2f} MB")

        with requests.get(url, stream=True) as r:
            r.raise_for_status() 
            with open(destination_path, 'wb') as f:
                with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, unit_divisor=1024, desc=os.path.basename(destination_path)) as pbar:
                    for chunk in r.iter_content(chunk_size=block_size):
                        if chunk: 
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        print(f"Đã tải file thành công về: {destination_path}")
        return destination_path
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi tải file: {e}")
        return None
    except Exception as e:
        print(f"Lỗi không mong muốn: {e}")
        return None

if __name__ == "__main__":

    repo_id = "FahNos/GCDRnet"
    filename = "gcnet.pkl"
    target_folder = "./pretrained_model"
    revision = "d724588c1ddb1d4aabc23e4bbe0120654c03fdd2"
    downloaded_path = download_and_move_hf_file(repo_id, filename, target_folder)

    repo_id = "FahNos/GCDRnet"
    filename = "drnet.pkl"
    target_folder = "./pretrained_model"
    revision = "454880682fcf06059911f380c70e2ba8e246f4d1363c94441005057dac5d7863"
    downloaded_path = download_and_move_hf_file(repo_id, filename, target_folder)


    hf_cdn_url = "https://cdn-lfs-us-1.hf.co/repos/c8/fb/c8fb190ae515336a77a403033e97d8e0ce2a5e92172e95fdeb786f05316906cf/c8da4205c47968bea73afc8caaff0c3f67b8645be031a7063fc1d0f381d2a702?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27RealDAE.zip%3B+filename%3D%22RealDAE.zip%22%3B&response-content-type=application%2Fzip&Expires=1749607696&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0OTYwNzY5Nn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zL2M4L2ZiL2M4ZmIxOTBhZTUxNTMzNmE3N2E0MDMwMzNlOTdkOGUwY2UyYTVlOTIxNzJlOTVmZGViNzg2ZjA1MzE2OTA2Y2YvYzhkYTQyMDVjNDc5NjhiZWE3M2FmYzhjYWFmZjBjM2Y2N2I4NjQ1YmUwMzFhNzA2M2ZjMWQwZjM4MWQyYTcwMj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoifV19&Signature=YsQwKr5pWgJPkdxC0yI%7Et-sjQhKEOqw5iZh91U0DkTRr3ifIG9tIsbeU1iFwgQLY7bRsBVN1FgE7eqbg7gd9ZIp5YrB5IJPgAX5bAu44Lcwwaafxc4FLHr-Vt0qA1VDLs3DfJrDVKzrUu7xF94V3%7E8OUrTegK8VNKk5EraFEXLtOh7EB-niwBLH8hg-NdiU3XzrWKi3D6DIMdSw5JRbh8ScnJV66aL6aWTn9dQJzs54BiKjPlEvDq9LVOJ0vSBCYeJkyrDBDFJDWAEPx2tusdyYXnL8rmIHoTHoN73VJ6qnhkat3dC04VErR-SzzWdKJe7PSEO2L31f-NdRAkSWrkQ__&Key-Pair-Id=K24J24Z295AEI9"
    output_filename = "RealDAE.zip"    
    target_folder = "./data"
    destination_file_path = os.path.join(target_folder, output_filename)    
    downloaded_path = download_file_from_url(hf_cdn_url, destination_file_path)

    if downloaded_path:
        print(f"File đã tải về và lưu tại: {downloaded_path}")

    my_zip_file = './data/RealDAE.zip'
    my_extract_folder = './data/realDAE'

    extract_and_remove(my_zip_file, my_extract_folder)

    organize_realdae_images('./data/realDAE', './data')
    select_and_copy_random_images(
        './data/in',
        './data/gt',
        './distorted_eval_data/',
        './gt_eval_data',
        num_samples=150
    )
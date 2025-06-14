import torch
import torch.nn.functional as F
import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
import argparse

_TORCH_EXPORT_DIM_AVAILABLE = False
try:
    from torch.export import Dim
    _TORCH_EXPORT_DIM_AVAILABLE = True
    print("Successfully imported torch.export.Dim.")
except ImportError:
    print("Warning: Failed to import torch.export.Dim. Will use None for dynamic shapes (deprecated).")

try:
    import ai_edge_torch
    print("ai-edge-torch imported successfully")
except ImportError:
    print("Error: ai-edge-torch not found. Please install it using:")
    print("pip install ai-edge-torch")
    exit(1)

from utils import convert_state_dict
from data.preprocess.crop_merge_image import stride_integral
os.sys.path.append('./models/UNeXt')
from models.UNeXt.unext import UNext_full_resolution_padding, UNext_full_resolution_padding_L_py_L


def convert_pytorch_to_tflite_both_precisions(model, model_name, sample_input, output_dir):

    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    dynamic_dims_for_input_tensor = {}
    dynamic_shapes_source_message = ""

    if _TORCH_EXPORT_DIM_AVAILABLE:
       
        dynamic_height_dim = Dim("height")
        dynamic_width_dim = Dim("width")
        
       
        if sample_input.ndim > 2:  
            dynamic_dims_for_input_tensor[2] = dynamic_height_dim
        if sample_input.ndim > 3:  
            dynamic_dims_for_input_tensor[3] = dynamic_width_dim
        dynamic_shapes_source_message = "(using torch.export.Dim)"
    else:
      
        if sample_input.ndim > 2:
            dynamic_dims_for_input_tensor[2] = None
        if sample_input.ndim > 3:
            dynamic_dims_for_input_tensor[3] = None
        dynamic_shapes_source_message = "(using None - deprecated)"


    if not dynamic_dims_for_input_tensor: 
        dynamic_shapes_spec = None
    else:
       
        dynamic_shapes_spec = (dynamic_dims_for_input_tensor,)

    print(f"Converting {model_name} to TFLite FP32 with dynamic_shapes_spec: {dynamic_shapes_spec} {dynamic_shapes_source_message}...")
    try:
        sample_args_tuple = (sample_input,) 
        
        edge_model_fp32 = ai_edge_torch.convert(
            model,
            sample_args_tuple,
            dynamic_shapes=dynamic_shapes_spec 
        )
        
        fp32_path = os.path.join(output_dir, f"{model_name}_fp32.tflite")
        edge_model_fp32.export(fp32_path)
        
        print(f"Successfully converted {model_name} to FP32 with dynamic shapes: {fp32_path}")
        results['fp32'] = fp32_path
        
    except Exception as e:
        import traceback
        print(f"Error converting {model_name} to FP32 with dynamic shapes: {str(e)}")
        print(traceback.format_exc()) 
        results['fp32'] = None
    
    # Xử lý FP16
    print(f"Creating TFLite FP16 placeholder for {model_name}...")
    try:
        if results['fp32']:
            fp16_path = os.path.join(output_dir, f"{model_name}_fp16.tflite")
            import shutil
            shutil.copy(results['fp32'], fp16_path)
            
            print(f"Created FP16 placeholder model (copied from FP32): {fp16_path}")
            results['fp16'] = fp16_path
        else:
            print(f"Skipping FP16 placeholder creation for {model_name} due to FP32 conversion failure.")
            results['fp16'] = None
            
    except Exception as e:
        print(f"Error creating FP16 placeholder for {model_name}: {str(e)}")
        results['fp16'] = None
            
    return results


def quantize_tflite_to_fp16_manual(input_tflite_path, output_tflite_path):
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=input_tflite_path)
        interpreter.allocate_tensors()
        print(f"Attempting FP16 quantization of {input_tflite_path}...")
        import shutil
        shutil.copy(input_tflite_path, output_tflite_path)
        print(f"Placeholder FP16 model created at {output_tflite_path}")
        print("For true FP16 quantization, use TensorFlow Lite converter with the original model")
        return True
    except Exception as e:
        print(f"Quantization failed: {str(e)}")
        return False


def test_tflite_model(tflite_path, path_list, in_folder, sav_folder, model_type='model1'):
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
   
    interpreter.allocate_tensors() 
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Testing {model_type} - Loaded Input shape from TFLite model: {input_details[0]['shape']}")
    print(f"Testing {model_type} - Loaded Output shape from TFLite model: {output_details[0]['shape']}")
    
    
    first_im_path = path_list[0]
    first_im_org_cv = cv2.imread(first_im_path)
    if first_im_org_cv is None:
        print(f"Error: Could not read the first image {first_im_path} for testing {model_type}")
        return

    first_im_padded_cv, _, _ = stride_integral(first_im_org_cv)
    h_actual, w_actual = first_im_padded_cv.shape[:2]
    
   
    current_input_shape = [1, input_details[0]['shape'][1], h_actual, w_actual] # Giữ batch=1, channels từ model
    
    
    print(f"Resizing {model_type} TFLite interpreter for actual test image shape: {current_input_shape}")
    try:
        interpreter.resize_tensor_input(input_details[0]['index'], current_input_shape, strict=False)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error resizing TFLite interpreter for {model_type}: {e}")
        print(f"Model's expected input shape was {input_details[0]['shape']}, tried to resize to {current_input_shape}")
        return

    if model_type == 'model1':
        im_input_data = (first_im_padded_cv.transpose(2, 0, 1).astype(np.float32) / 255.0)
        im_input_data = np.expand_dims(im_input_data, axis=0) # Shape [1, 3, h_actual, w_actual]
    elif model_type == 'model2':
        
        num_channels_model2 = input_details[0]['shape'][1] 
        if num_channels_model2 != 6: 
             print(f"Warning: Model 2 TFLite expects {num_channels_model2} channels, but test creates 6.")
        im_input_data = np.random.randn(1, num_channels_model2, h_actual, w_actual).astype(np.float32)
    else:
        print(f"Unknown model_type: {model_type}")
        return

    interpreter.set_tensor(input_details[0]['index'], im_input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    output_details_after_invoke = interpreter.get_output_details()
    print(f"Testing {model_type} - Actual Output shape after invoke: {output_details_after_invoke[0]['shape']}")


def test_model1_model2_tflite_pipeline(model1_path, model2_path, path_list, in_folder, sav_folder):
    import tensorflow as tf
    import numpy as np
    
    interpreter1 = tf.lite.Interpreter(model_path=model1_path)
    interpreter2 = tf.lite.Interpreter(model_path=model2_path)
    
    input_details1_orig = interpreter1.get_input_details()
    
    input_details2_orig = interpreter2.get_input_details()
    
    for im_path in tqdm(path_list):
        im_org_cv = cv2.imread(im_path)
        if im_org_cv is None:
            continue
        
        im_padded_cv, crop_padding_h, crop_padding_w = stride_integral(im_org_cv)
        h_padded, w_padded = im_padded_cv.shape[:2]
        
        # --- Model 1 ---
        new_shape_model1 = [1, input_details1_orig[0]['shape'][1], h_padded, w_padded] 
        
        interpreter1.resize_tensor_input(input_details1_orig[0]['index'], new_shape_model1, strict=False)
        interpreter1.allocate_tensors()
        
        im_input1 = (im_padded_cv.transpose(2, 0, 1).astype(np.float32) / 255.0)
        im_input1 = np.expand_dims(im_input1, axis=0) # Shape [1, 3, h_padded, w_padded]
        
        interpreter1.set_tensor(input_details1_orig[0]['index'], im_input1)
        interpreter1.invoke()
        output_details1_resized = interpreter1.get_output_details()
        shadow_output = interpreter1.get_tensor(output_details1_resized[0]['index'])
        
        shadow_torch = torch.from_numpy(shadow_output).float()
        shadow_resized = F.interpolate(shadow_torch, size=(h_padded, w_padded), mode='bilinear', align_corners=False)
        
        im_org_torch_for_enhancement = torch.from_numpy(im_input1).float()
        enhanced_torch = torch.clamp(im_org_torch_for_enhancement / shadow_resized, 0, 1)
        enhanced_np = enhanced_torch.numpy()
        
        # --- Model 2 ---
        im_input2_data = np.concatenate([im_input1, enhanced_np], axis=1) # Shape [1, 6, h_padded, w_padded]
        new_shape_model2 = [1, input_details2_orig[0]['shape'][1], h_padded, w_padded]

        interpreter2.resize_tensor_input(input_details2_orig[0]['index'], new_shape_model2, strict=False)
        interpreter2.allocate_tensors()
        
        interpreter2.set_tensor(input_details2_orig[0]['index'], im_input2_data)
        interpreter2.invoke()
        output_details2_resized = interpreter2.get_output_details()
        final_output = interpreter2.get_tensor(output_details2_resized[0]['index'])
        
        result = np.transpose(final_output[0], (1, 2, 0)) # Output shape [h_final, w_final, 3]
        result = (result * 255).astype(np.uint8)
        
       
        if result.shape[0] != h_padded or result.shape[1] != w_padded:
            print(f"Warning: Model 2 TFLite output spatial size ({result.shape[0]}x{result.shape[1]}) "
                  f"differs from padded input size ({h_padded}x{w_padded}). Resizing before crop.")
            final_output_torch = torch.from_numpy(final_output.astype(np.float32)/255.0) 
            final_output_resized_torch = F.interpolate(final_output_torch, size=(h_padded, w_padded), mode='bilinear', align_corners=False)
            result_resized_np = np.transpose(final_output_resized_torch.numpy()[0], (1,2,0))
            result = (result_resized_np * 255.0).astype(np.uint8)


        result_final = result[crop_padding_h:, crop_padding_w:]
        
        save_path = im_path.replace(in_folder, sav_folder)
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(save_path, result_final)

    print("Pipeline with TFLite completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch to TFLite Converter")
    parser.add_argument('-MODE', type=str, default='convert', 
                       help="'convert' to convert models, 'test' to test existing models, 'both' for both")
    
    args = parser.parse_args()
    mode = args.MODE.lower()
    
    model_dir = './pretrained_model'
    tflite_model_dir = './tflite_models/'
    os.makedirs(tflite_model_dir, exist_ok=True)
    
    model1_pkl_path = os.path.join(model_dir, 'gcnet.pkl')
    model2_pkl_path = os.path.join(model_dir, 'drnet.pkl')
    
    img_folder = './distorted/'
    sav_folder = './enhanced_tflite/'
    os.makedirs(sav_folder, exist_ok=True)
    
    im_paths = glob.glob(os.path.join(img_folder, '*'))
    if not im_paths:
        print(f"No images found in {img_folder}. Creating dummy image...")
        os.makedirs(img_folder, exist_ok=True)
        dummy_img_path = os.path.join(img_folder, "dummy_image.png")
        dummy_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(dummy_img_path, dummy_array)
        im_paths = [dummy_img_path]
    
    temp_img = cv2.imread(im_paths[0])
    temp_img_padded, _, _ = stride_integral(temp_img)
    dummy_h, dummy_w = temp_img_padded.shape[:2]
    print(f"Using sample dimensions for TFLite conversion: H={dummy_h}, W={dummy_w}")
    
    if mode in ['convert', 'both']:
        print("=== CONVERTING MODELS =====")
        
        print("\nConverting Model 1...")
        model1_pt = UNext_full_resolution_padding(num_classes=3, input_channels=3, img_size=512)
        state1 = convert_state_dict(torch.load(model1_pkl_path, map_location='cpu')['model_state'])
        model1_pt.load_state_dict(state1)
        model1_pt.eval()
        sample_input1 = torch.randn(1, 3, dummy_h, dummy_w, dtype=torch.float32)
        model1_results = convert_pytorch_to_tflite_both_precisions(model1_pt, 'gcnet', sample_input1, tflite_model_dir)
        del model1_pt
        
        print("\nConverting Model 2...")
        model2_pt = UNext_full_resolution_padding_L_py_L(num_classes=3, input_channels=6, img_size=512)
        state2 = convert_state_dict(torch.load(model2_pkl_path, map_location='cpu')['model_state'])
        model2_pt.load_state_dict(state2)
        model2_pt.eval()
        sample_input2 = torch.randn(1, 6, dummy_h, dummy_w, dtype=torch.float32)
        with torch.no_grad():
            pytorch_output_model2 = model2_pt(sample_input2)
            output_shape_to_print = pytorch_output_model2.shape if isinstance(pytorch_output_model2, torch.Tensor) else pytorch_output_model2[0].shape
            print(f"PyTorch Model 2 - Input shape: {sample_input2.shape}")
            print(f"PyTorch Model 2 - Output shape: {output_shape_to_print}")
        model2_results = convert_pytorch_to_tflite_both_precisions(model2_pt, 'drnet', sample_input2, tflite_model_dir)
        del model2_pt
        
        print(f"\nConversion Results:")
        print(f"Model 1 - FP32: {model1_results.get('fp32', 'Failed')}")
        print(f"Model 1 - FP16: {model1_results.get('fp16', 'Failed')}")
        print(f"Model 2 - FP32: {model2_results.get('fp32', 'Failed')}")
        print(f"Model 2 - FP16: {model2_results.get('fp16', 'Failed')}")
    
    if mode in ['test', 'both']:
        print("\n=== TESTING MODELS ===")
        model1_fp32 = os.path.join(tflite_model_dir, 'gcnet_fp32.tflite')
        model2_fp32 = os.path.join(tflite_model_dir, 'drnet_fp32.tflite')
        
        if os.path.exists(model1_fp32):
            print(f"\nTesting Model 1 FP32...")
            test_tflite_model(model1_fp32, im_paths, img_folder, sav_folder, 'model1')
        
        if os.path.exists(model2_fp32):
            print(f"\nTesting Model 2 FP32...")
            test_tflite_model(model2_fp32, im_paths, img_folder, sav_folder, 'model2')
        
        if os.path.exists(model1_fp32) and os.path.exists(model2_fp32):
            print(f"\nRunning complete TFLite pipeline...")
            test_model1_model2_tflite_pipeline(model1_fp32, model2_fp32, im_paths, img_folder, sav_folder)
            print("Pipeline completed!")
    
    print("\nScript finished!")
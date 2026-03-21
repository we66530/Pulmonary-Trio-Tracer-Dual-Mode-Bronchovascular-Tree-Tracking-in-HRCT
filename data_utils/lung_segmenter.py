import os
import cv2
import numpy as np
import SimpleITK as sitk
from lungmask import mask

class LungSegmenter:
    """
    專門處理肺部遮罩生成的專家類別。
    整合了從 JPG 模擬 HU 值到 U-Net R231 推理的完整流程。
    """
    def __init__(self, model_name="R231"):
        self.model_name = model_name

    def generate_mask(self, vol_np):
        """
        輸入: vol_np (Z, H, W) 0-255 uint8
        輸出: binary_mask (Z, H, W) 0 或 255 uint8
        """
        # --- 你的核心邏輯：HU 模擬 ---
        # 轉換為 float32 並線性映射到 -1000 ~ +400 (近似 HU)
        hu_vol = (vol_np.astype(np.float32) / 255.0) * 1400.0 - 1000.0
        
        # 轉為 SimpleITK 格式並設定 Spacing
        sitk_vol = sitk.GetImageFromArray(hu_vol)
        sitk_vol.SetSpacing((1.0, 1.0, 1.0)) 

        print(f" Running Lungmask Model ({self.model_name})...")
        try:
            # 執行分割
            segmentation = mask.apply(sitk_vol) 
            # 將 1(右肺), 2(左肺) 等標籤轉為二值 255
            binary_mask = (segmentation > 0).astype(np.uint8) * 255
            return binary_mask
        except Exception as e:
            print(f" Lungmask Inference Error: {e}")
            return None
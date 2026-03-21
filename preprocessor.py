import os
import cv2
import torch
import numpy as np
from pathlib import Path
from lungmask import mask as lung_segmenter
import SimpleITK as sitk

class MedicalImagePreprocessor:
    def __init__(self, device="cuda"):
        self.device = device

    def load_ct_volume(self, ct_dir):
        """
        讀取目錄下所有的 JPG 影像並轉為 3D Numpy Array (Z, H, W)
        """
        paths = sorted(list(Path(ct_dir).glob("*.jpg")))
        if not paths:
            raise ValueError(f"No JPG files found in {ct_dir}")
        
        # 讀取第一張確認尺寸
        sample = cv2.imread(str(paths[0]), 0)
        h, w = sample.shape
        vol = np.zeros((len(paths), h, w), dtype=np.uint8)
        
        for i, p in enumerate(paths):
            vol[i] = cv2.imread(str(p), 0)
        return vol, paths

    def generate_lung_mask(self, vol):
        """
        利用 lungmask library 生成肺部遮罩
        """
        print(" Generating Lung Mask using AI (R231)...")
        # lungmask 需要 SimpleITK 格式輸入
        itk_img = sitk.GetImageFromArray(vol)
        
        # 執行分割 (R231 是預設模型，適配性最強)
        segmentation = lung_segmenter.apply(itk_img) # 輸出 (Z, H, W)
        
        # 轉回 numpy 並二值化 (lungmask 輸出通常 1 代表左肺, 2 代表右肺)
        lung_mask = (segmentation > 0).astype(np.uint8)
        return lung_mask

    def postprocess_lung_mask(self, lung_mask, dilation_kernel_size=15):
        """
        對 Lung Mask 進行膨脹，建立「安全圍欄」，防止邊緣血管被切掉
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
        dilated_mask = np.zeros_like(lung_mask)
        
        for z in range(lung_mask.shape[0]):
            dilated_mask[z] = cv2.dilate(lung_mask[z], kernel, iterations=1)
        return dilated_mask

    def get_25d_stack(self, vol, z):
        """
        獲取 (z-1, z, z+1) 的堆疊張量
        """
        z_dim = vol.shape[0]
        indices = [np.clip(z-1, 0, z_dim-1), z, np.clip(z+1, 0, z_dim-1)]
        # 歸一化至 0~1
        stack = vol[indices].astype(np.float32) / 255.0
        return torch.from_numpy(stack).unsqueeze(0) # (1, 3, H, W)

    @staticmethod
    def save_mask_series(mask_vol, output_path, prefix=""):
        """
        將 3D Mask 存回一系列 PNG
        """
        os.makedirs(output_path, exist_ok=True)
        for z in range(mask_vol.shape[0]):
            filename = f"{z:05d}.png"
            cv2.imwrite(os.path.join(output_path, filename), mask_vol[z] * 255)
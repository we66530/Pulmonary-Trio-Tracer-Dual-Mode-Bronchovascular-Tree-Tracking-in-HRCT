import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from core.model import TrioContextUNet, BronchiExpertUNet

class GlobalInference:
    def __init__(self, trio_path, bronchi_path, device="cuda"):
        self.device = device
        
        # 載入 A/V Trio 模型 (3-channel output)
        self.trio_model = TrioContextUNet().to(device)
        self.trio_model.load_state_dict(torch.load(trio_path, map_location=device))
        self.trio_model.eval()

        # 載入 Bronchi 專家模型 (1-channel output)
        self.bronchi_model = BronchiExpertUNet().to(device)
        self.bronchi_model.load_state_dict(torch.load(bronchi_path, map_location=device))
        self.bronchi_model.eval()

    def run(self, vol, lung_vol, output_dir):
        z_dim, h, w = vol.shape
        os.makedirs(output_dir, exist_ok=True)
        for name in ["Artery", "Vein", "Bronchi"]:
            os.makedirs(os.path.join(output_dir, name), exist_ok=True)

        print(f" Running Hybrid Global Inference (Trio + Bronchi Expert)...")
        
        for z in tqdm(range(z_dim)):
            # 準備 2.5D 輸入
            indices = [np.clip(z-1, 0, z_dim-1), z, np.clip(z+1, 0, z_dim-1)]
            input_patch = vol[indices].astype(np.float32) / 255.0
            input_t = torch.from_numpy(input_patch).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # 1. 獲取 A/V Trio 預測 (前兩個 Channel 分別是 A, V)
                trio_pred = torch.sigmoid(self.trio_model(input_t)).squeeze(0).cpu().numpy()
                
                # 2. 獲特 Bronchi 專家預測 (1-Channel)
                bronchi_pred = torch.sigmoid(self.bronchi_model(input_t)).squeeze().cpu().numpy()
            
            current_lung_mask = lung_vol[z]

            # 3. 分別處理並儲存
            for i, name in enumerate(["Artery", "Vein", "Bronchi"]):
                if name == "Bronchi":
                    # 使用專家模型的輸出
                    mask = (bronchi_pred > 0.45).astype(np.uint8) * 255
                else:
                    # 使用 Trio 模型的對應 Channel (0:Artery, 1:Vein)
                    mask = (trio_pred[i] > 0.55).astype(np.uint8) * 255
                
                # 套用肺部約束 (氣管若是在肺門處，約束可視需求調整)
                mask_refined = cv2.bitwise_and(mask, mask, mask=current_lung_mask)
                
                save_path = os.path.join(output_dir, name, f"{z:05d}.png")
                cv2.imwrite(save_path, mask_refined)
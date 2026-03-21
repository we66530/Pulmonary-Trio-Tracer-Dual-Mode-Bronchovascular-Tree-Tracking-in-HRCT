import torch
import numpy as np
import cv2
from collections import deque
from core.model import TrioContextUNet, BronchiExpertUNet

class HybridSeedTracker:
    def __init__(self, trio_path, bronchi_path, device="cuda", patch_size=64):
        self.device = device
        self.patch_size = patch_size
        
        # 1. 載入 Trio 模型 (處理 Artery 與 Vein)
        self.trio_model = TrioContextUNet().to(device)
        self.trio_model.load_state_dict(torch.load(trio_path, map_location=device, weights_only=True))
        self.trio_model.eval()
        
        # 2. 載入 Bronchi 專家模型 (處理 Bronchi)
        self.bronchi_expert = BronchiExpertUNet().to(device)
        self.bronchi_expert.load_state_dict(torch.load(bronchi_path, map_location=device, weights_only=True))
        self.bronchi_expert.eval()

    def _get_25d_patch(self, vol, z, cy, cx):
        """ 提取 2.5D Patch (z-1, z, z+1) """
        half = self.patch_size // 2
        z_indices = [np.clip(z-1, 0, vol.shape[0]-1), z, np.clip(z+1, 0, vol.shape[0]-1)]
        patch = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
        h, w = vol.shape[1], vol.shape[2]
        
        y0, x0 = cy - half, cx - half
        ay0, ay1 = max(0, y0), min(h, y0 + self.patch_size)
        ax0, ax1 = max(0, x0), min(w, x0 + self.patch_size)
        
        if ay1 > ay0 and ax1 > ax0:
            for i, zz in enumerate(z_indices):
                # 取得局部區塊並歸一化
                roi = vol[zz, ay0:ay1, ax0:ax1]
                patch[i, ay0-y0:ay1-y0, ax0-x0:ax1-x0] = roi / 255.0
        return patch

    def track_all(self, vol, lung_mask, seed_dict, output_dir):
        """
        vol: CT 數值卷軸 (Z, H, W)
        lung_mask: 肺部遮罩 (Z, H, W)
        seed_dict: {"Artery": [pts], "Vein": [pts], "Bronchi": [pts]}
        """
        z_limit, h, w = vol.shape
        # master_masks 順序建議與 COLORS 字典對應: 0:Artery, 1:Vein, 2:Bronchi
        master_masks = np.zeros((3, z_limit, h, w), dtype=np.uint8)
        half = self.patch_size // 2
        
        # 建立標籤映射，確保索引正確
        labels = ["Artery", "Vein", "Bronchi"]

        for t_idx, label in enumerate(labels):
            print(f">>> Seed-based Tracking: {label}...")
            
            # 取得該類別的種子點
            seeds = seed_dict.get(label, [])
            queue = deque(seeds)
            visited = set()
            
            # 設定門檻：氣管模型通常需要較低門檻確保連通性
            threshold = 0.35 if label == "Bronchi" else 0.6

            while queue:
                z, cy, cx, direct = queue.popleft()
                nz = z + direct # 移動到下一層
                
                # 邊界檢查與去重 (visited 使用縮放座標減少計算負擔)
                if nz < 0 or nz >= z_limit: continue
                v_key = (nz, cy // 2, cx // 2, direct)
                if v_key in visited: continue
                visited.add(v_key)

                # 肺部約束：血管 (Artery/Vein) 必須在肺部遮罩內，氣管則不做嚴格限制
                if label != "Bronchi":
                    # 確保座標在範圍內且在肺部
                    if 0 <= cy < h and 0 <= cx < w:
                        if lung_mask[nz, cy, cx] == 0: continue
                    else: continue

                # 提取 Patch 並推理
                patch = self._get_25d_patch(vol, nz, cy, cx)
                input_t = torch.from_numpy(patch).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    if label == "Bronchi":
                        # 使用專家模型
                        pred = torch.sigmoid(self.bronchi_expert(input_t)).squeeze().cpu().numpy()
                    else:
                        # 使用 Trio 模型 (多通道輸出)
                        preds = torch.sigmoid(self.trio_model(input_t)).squeeze(0).cpu().numpy()
                        # 根據順序取得對應通道 (假設 Trio 輸出 0:A, 1:V, 2:B，我們只取 A 或 V)
                        pred = preds[t_idx]

                # 二值化
                mask_p = (pred > threshold).astype(np.uint8)
                
                if np.any(mask_p):
                    # 連通域分析，尋找下一層的中心點
                    n, lab, st, cents = cv2.connectedComponentsWithStats(mask_p)
                    for i in range(1, n):
                        if st[i, cv2.CC_STAT_AREA] < 3: continue # 濾除極小雜訊
                        
                        y_idx, x_idx = np.where(lab == i)
                        # 計算在全域影像中的座標
                        gy, gx = y_idx + cy - half, x_idx + cx - half
                        
                        # 確保座標不越界
                        valid = (gy >= 0) & (gy < h) & (gx >= 0) & (gx < w)
                        master_masks[t_idx, nz, gy[valid], gx[valid]] = 255
                        
                        # 更新下一層中心點 (使用 Connected Component 的質心)
                        ncy = int(cy - half + cents[i][1])
                        ncx = int(cx - half + cents[i][0])
                        
                        if 0 <= ncy < h and 0 <= ncx < w:
                            queue.append((nz, ncy, ncx, direct))
        
        return master_masks

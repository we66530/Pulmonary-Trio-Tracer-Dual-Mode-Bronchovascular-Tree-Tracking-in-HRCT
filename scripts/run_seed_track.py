import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from data_utils.preprocessor import MedicalImagePreprocessor
from data_utils.lung_segmenter import LungSegmenter
from data_utils.lung_mask_utils import LungMaskHandler
from core.tracker import HybridSeedTracker

def load_seeds_from_dir(seed_dir):
    """
    讀取種子點資料夾，將 .png 轉換為 (z, y, x, direction) 列表
    """
    seeds = []
    if not seed_dir or not os.path.exists(seed_dir):
        return seeds
        
    seed_files = sorted(list(Path(seed_dir).glob("*.png")))
    for f in seed_files:
        # 假設檔名格式如 '00050.png' 代表 slice 50
        try:
            z = int(f.stem)
        except ValueError:
            continue
            
        mask = cv2.imread(str(f), 0)
        if mask is None: continue
        
        # 找到 Mask 的所有白色像素座標 (y, x)
        ys, xs = np.where(mask > 127)
        if len(ys) > 0:
            # 取得中心點作為追蹤起點
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            # 每個點都加入「向上(-1)」與「向下(1)」兩個追蹤方向
            seeds.append((z, cy, cx, 1))
            seeds.append((z, cy, cx, -1))
            
    return seeds

def main(args):
    # 1. 初始化工具
    prep = MedicalImagePreprocessor(device=args.device)
    handler = LungMaskHandler()
    
    # 2. 載入 CT 影像
    print(f"Loading CT Volume: {args.ct_dir}")
    vol, img_paths = prep.load_ct_volume(args.ct_dir)
    z_limit, h, w = vol.shape
    
    # 3. 準備 Lung Mask
    if args.lung_mask_dir and os.path.exists(args.lung_mask_dir):
        print(f"Loading existing Lung Masks from {args.lung_mask_dir}...")
        # 建立一個與 vol 同尺寸的空白容器
        lung_vol = np.zeros_like(vol, dtype=np.uint8)
        for i, p in enumerate(img_paths):
            m_path = os.path.join(args.lung_mask_dir, p.stem + ".png")
            if os.path.exists(m_path):
                m_img = cv2.imread(m_path, 0)
                if m_img is not None:
                    # 確保尺寸與 CT 一致
                    if m_img.shape != (h, w):
                        m_img = cv2.resize(m_img, (w, h), interpolation=cv2.INTER_NEAREST)
                    lung_vol[i] = m_img
    else:
        print("Generating Lung Masks using segmenter...")
        segmenter = LungSegmenter()
        lung_vol = segmenter.generate_mask(vol)
    
    # 根據 GUI 參數進行膨脹
    if args.dilate > 0:
        print(f"Dilating lung mask with kernel {args.dilate}...")
        lung_vol = handler.dilate_mask(lung_vol, kernel_size=args.dilate)

    # 4. 關鍵修正：將「資料夾路徑」轉換為「種子座標列表」
    print("Converting seed masks to coordinates...")
    seed_data = {
        "Artery": load_seeds_from_dir(args.seed_a),
        "Vein": load_seeds_from_dir(args.seed_v),
        "Bronchi": load_seeds_from_dir(args.seed_b)
    }

    # 5. 執行追蹤
    tracker = HybridSeedTracker(
        trio_path=args.trio_model,
        bronchi_path=args.bronchi_model,
        device=args.device,
        patch_size=args.patch_size
    )
    
    # 執行追蹤，取得三通道 master_masks (3, Z, H, W)
    master_masks = tracker.track_all(
        vol=vol, 
        lung_mask=lung_vol, 
        seed_dict=seed_data, 
        output_dir=args.output_dir
    )

    # 6. 儲存追蹤結果
    print(f"Saving tracking results to: {args.output_dir}")
    labels = ["Artery", "Vein", "Bronchi"]
    for idx, label in enumerate(labels):
        label_dir = os.path.join(args.output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # 逐層儲存為 PNG
        for z in range(z_limit):
            slice_mask = master_masks[idx, z]
            # 只有當該層有東西時才存，或者全部存 (為了 Overlay 建議全部存)
            out_name = f"{z:05d}.png"
            cv2.imwrite(os.path.join(label_dir, out_name), slice_mask)

    print(f"✅ Tracking complete. Total slices: {z_limit}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed-based Tracking Script")
    parser.add_argument("--ct_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--trio_model", type=str, required=True)
    parser.add_argument("--bronchi_model", type=str, required=True)
    parser.add_argument("--seed_a", type=str, required=True)
    parser.add_argument("--seed_v", type=str, required=True)
    parser.add_argument("--seed_b", type=str, required=True)
    parser.add_argument("--lung_mask_dir", type=str, default=None)
    parser.add_argument("--dilate", type=int, default=15)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    main(args)

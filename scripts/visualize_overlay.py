import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

class TrioVisualizer:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        # BGR 顏色定義
        self.colors = {
            "Artery": [0, 0, 255],   # 紅色
            "Vein": [255, 0, 0],     # 藍色
            "Bronchi": [0, 255, 0]   # 綠色
        }

    def process(self, ct_dir, mask_base_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        ct_files = sorted(list(Path(ct_dir).glob("*.jpg")))
        
        print(f" Generating Overlays in: {output_dir}")

        for i, ct_path in enumerate(tqdm(ct_files)):
            ct_img = cv2.imread(str(ct_path))
            if ct_img is None: continue
            h, w = ct_img.shape[:2]
            
            # 準備彩色疊加層
            overlay_layer = ct_img.copy()
            found_any = False

            for name, color in self.colors.items():
                # 修改路徑邏輯，對齊你的輸出結構
                mask_path = Path(mask_base_dir) / name / f"{i:05d}.png"
                
                if mask_path.exists():
                    mask_img = cv2.imread(str(mask_path), 0)
                    if mask_img is not None:
                        # 確保尺寸對齊
                        if mask_img.shape[:2] != (h, w):
                            mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        overlay_layer[mask_img > 127] = color
                        found_any = True

            # 混合原圖與色彩層
            if found_any:
                output = cv2.addWeighted(overlay_layer, self.alpha, ct_img, 1 - self.alpha, 0)
            else:
                output = ct_img.copy()

            # 加上 GUI 友善的標籤
            self._add_info_text(output, i)

            # 儲存
            cv2.imwrite(os.path.join(output_dir, f"overlay_{i:05d}.jpg"), output)

    def _add_info_text(self, img, slice_idx):
        """在影像左上角繪製圖例"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"Slice: {slice_idx:03d}", (20, 30), font, 0.7, (255, 255, 255), 2)
        y_offset = 60
        for name, color in self.colors.items():
            cv2.putText(img, f"• {name}", (20, y_offset), font, 0.6, color, 2)
            y_offset += 25

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    viz = TrioVisualizer(alpha=args.alpha)
    viz.process(args.ct_dir, args.mask_dir, args.output_dir)

if __name__ == "__main__":
    main()
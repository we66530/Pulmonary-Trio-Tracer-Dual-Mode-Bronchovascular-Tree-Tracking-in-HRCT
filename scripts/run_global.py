import os
import argparse
from data_utils.preprocessor import MedicalImagePreprocessor
from data_utils.lung_segmenter import LungSegmenter
from data_utils.lung_mask_utils import LungMaskHandler
from core.inference import GlobalInference

def main(args):
    # 1. 初始化工具
    prep = MedicalImagePreprocessor(device=args.device)
    handler = LungMaskHandler()
    
    # 2. 載入 CT 影像
    print(f" Loading CT from: {args.ct_dir}")
    vol, img_paths = prep.load_ct_volume(args.ct_dir)
    
    # 3. 處理 Lung Mask (如果沒提供路徑就現場生一個)
    if args.lung_mask_dir and os.path.exists(args.lung_mask_dir):
        print(f" Loading existing Lung Masks...")
        # 這裡假設現有的 mask 與 CT 同名
        lung_vol = np.stack([cv2.imread(os.path.join(args.lung_mask_dir, p.stem + ".png"), 0) for p in img_paths])
    else:
        segmenter = LungSegmenter()
        lung_vol = segmenter.generate_mask(vol)
    
    # 根據 GUI 設定決定是否膨脹
    if args.dilate > 0:
        lung_vol = handler.dilate_mask(lung_vol, kernel_size=args.dilate)

    # 4. 執行全域推理 (注意：這裡傳入兩個專家模型路徑)
    engine = GlobalInference(
        trio_path=args.trio_model, 
        bronchi_path=args.bronchi_model, 
        device=args.device
    )
    
    engine.run(vol, lung_vol, args.output_dir)
    print(f" Inference complete. Results saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global Inference Script")
    # 這些參數未來會由 GUI 自動生成指令傳入
    parser.add_argument("--ct_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--trio_model", type=str, required=True)
    parser.add_argument("--bronchi_model", type=str, required=True)
    parser.add_argument("--lung_mask_dir", type=str, default=None)
    parser.add_argument("--dilate", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    main(args)
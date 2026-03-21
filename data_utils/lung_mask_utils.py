import cv2
import numpy as np

class LungMaskHandler:
    """
    專門處理肺部遮罩的後處理、膨脹與約束邏輯。
    """
    def __init__(self):
        pass

    @staticmethod
    def dilate_mask(mask_vol, kernel_size=15):
        """
        對 3D 遮罩進行 2D 逐層膨脹。
        用於應對肺部邊緣（Danger Zone）的血管偵測。
        """
        if kernel_size <= 0:
            return mask_vol
            
        print(f" Applying Morphological Dilation (size={kernel_size})...")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_vol = np.zeros_like(mask_vol)
        
        for z in range(mask_vol.shape[0]):
            # 確保輸入是 binary (0 or 255)
            m = (mask_vol[z] > 0).astype(np.uint8) * 255
            dilated_vol[z] = cv2.dilate(m, kernel, iterations=1)
            
        return dilated_vol

    @staticmethod
    def apply_constraints(prediction, lung_mask):
        """
        將模型預測結果與肺部遮罩進行點對點相乘 (Masking)。
        確保所有預測的血管/氣管都在肺部範圍內。
        """
        # 確保兩者維度一致
        return cv2.bitwise_and(prediction, prediction, mask=lung_mask)

    @staticmethod
    def remove_small_noise(mask_slice, min_area=5):
        """
        去除遮罩中的孤立雜訊點，這在 Global Inference 模式下很有用。
        """
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_slice)
        refined = np.zeros_like(mask_slice)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                refined[labels == i] = 255
        return refined
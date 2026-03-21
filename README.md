# Pulmonary-Trio-Tracer-Dual-Mode-Bronchovascular-Tree-Tracking-in-HRCT
Developed based on the LUNA-16 and AirRC datasets, this 2.5D deep learning system enables simultaneous tracking of arteries, veins, and bronchi.

# 🫁 Pulmonary Trio Tracer: AI-Driven A/V/B Segmentation

A high-precision deep learning pipeline for the simultaneous segmentation and tracking of **Pulmonary Arteries (A)**, **Veins (V)**, and **Bronchi (B)** from Computed Tomography (CT) scans.

---

## 📺 Demo & Performance

### 🚀 Result Visualizations

| Seed-based Expert Tracking | Global Inference (Fully Automated) |
| :---: | :---: |
| <video src="results/WithSeed.mp4" autoplay loop muted playsinline width="100%"></video><br>[Download Video](results/WithSeed.mp4) | <video src="results/NoSeed.mp4" autoplay loop muted playsinline width="100%"></video><br>[Download Video](results/NoSeed.mp4) |
| *High precision tracking from expert seeds* | *Rapid end-to-end automated segmentation* |



### 📊 Model Benchmark Results

#### 1. Bronchial Expert Model (Specialized)
| Metric | Value |
| :--- | :--- |
| 📊 **Mean DICE** | **0.8910** |
| 📊 **Mean clDice** (Skeleton) | **0.9311** |
| 📊 **Mean HD95** | **4.0236 px** |

#### 2. Trio Tracer Model (Artery/Vein/Bronchi)
| Structure | Dice ↑ | clDice ↑ | HD95 ↓ |
| :--- | :--- | :--- | :--- |
| **Artery (A)** | 0.8269 | 0.8862 | 6.07 |
| **Vein (V)** | 0.8046 | 0.8558 | 7.60 |
| **Bronchi (B)** | 0.7570 | 0.7940 | 7.48 |

> [!IMPORTANT]
> **⚠️ A-V Overlap Rate: 0.00%**
> This indicates zero classification errors between arteries and veins, ensuring perfect topological separation.

---

## 📥 Model Checkpoints
Download the weights and place them in the `checkpoint/` folder:

* 📦 **Bronchial Expert**: [Download bronchial_expert.pt](https://drive.google.com/file/d/1yFWxRzH9S79bI9oAN7NYqoOsZz3qusKy/view?usp=drive_link)
* 📦 **Trio Tracer (Fast)**: [Download trio_tracer_fast.pt](https://drive.google.com/file/d/1HOaNkO6rJJMZGntvbUtffvMR4pEIuGKV/view?usp=drive_link)

---

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/YourUsername/Pulmonary-Trio-Tracer.git](https://github.com/YourUsername/Pulmonary-Trio-Tracer.git)
cd Pulmonary-Trio-Tracer

# Install requirements
pip install -r requirements.txt

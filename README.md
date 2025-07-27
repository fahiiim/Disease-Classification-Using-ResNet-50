# Disease-Classification-Using-ResNet-50
# Disease Classification Using ResNet-50

![Disease Classification Banner](assets/banner.jpg)

## Overview

This repository presents a deep learning-based approach for disease classification leveraging the powerful **ResNet-50** convolutional neural network. The project is designed to classify medical images (such as X-rays, CT scans, or microscopic slides) into disease categories, aiding healthcare professionals in making faster and more accurate diagnoses.

The full workflow, from data preprocessing to model evaluation, is implemented in **Jupyter Notebooks** for transparency, reproducibility, and educational purposes.

---

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation & Usage](#installation--usage)
- [Results](#results)
- [Visualizations](#visualizations)
- [Citations](#citations)
- [References](#references)
- [License](#license)

---

## Introduction

Accurate and early detection of diseases from medical images is critical for effective treatment and patient outcomes. Deep learning models, especially those based on convolutional neural networks (CNNs), have demonstrated significant success in medical image analysis tasks[^1]. In this project, we utilize **ResNet-50**—a widely adopted deep residual network—to classify images into disease categories.

---

## Dataset

![Sample Images](assets/sample_images.png)

The dataset used in this project consists of labeled medical images. Each image is annotated with its respective disease class. Please refer to the `notebooks/` directory for details on data exploration and preprocessing steps.

*Note: For privacy and licensing reasons, the dataset is not included in this repository. Please use your own dataset or check the references for publicly available alternatives.*

---

## Methodology

1. **Data Preprocessing:**  
   - Image resizing, normalization, data augmentation.
   - Splitting into training, validation, and test sets.

2. **Model Architecture:**  
   - Utilized the pretrained ResNet-50 model.
   - Fine-tuned top layers for the disease classification task.

3. **Training:**  
   - Used transfer learning to leverage learned features.
   - Monitored performance using accuracy and loss metrics.

4. **Evaluation:**  
   - Evaluated on unseen data.
   - Detailed confusion matrix and class-wise performance metrics.

---

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fahiiim/Disease-Classification-Using-ResNet-50.git
   cd Disease-Classification-Using-ResNet-50
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebooks:**
   ```bash
   jupyter notebook
   ```
   Open and run the notebooks in order as described in the `notebooks/` directory.

---

## Results

| Metric      | Value      |
|-------------|------------|
| Accuracy    | 82.4%      |
| Precision   | 92.1%      |
| Recall      | 78.8%      |
| F1-Score    | 91.9%      |

![Confusion Matrix](assets/confusion_matrix.png)

The results demonstrate that ResNet-50 can effectively classify diseases from medical images, achieving high accuracy and balanced precision/recall.

---

## Visualizations

- **Sample predictions:**  
  ![Sample Predictions](assets/predictions.png)

- **Training Curves:**  
  ![Training and Validation Loss/Accuracy](assets/training_curves.png)

---

## Citations

If you use this code or build upon this work in your research, please cite:

```
@misc{fahiiim2025disease,
  title={Disease Classification Using ResNet-50},
  author={Fahiiim},
  year={2025},
  howpublished={\url{https://github.com/fahiiim/Disease-Classification-Using-ResNet-50}}
}
```

---

## References

[^1]: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)*, 770-778. [Paper](https://arxiv.org/abs/1512.03385)
[^2]: Esteva, A., Kuprel, B., Novoa, R. A., et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542(7639), 115-118. [Paper](https://www.nature.com/articles/nature21056)
[^3]: Litjens, G., Kooi, T., Bejnordi, B. E., et al. (2017). A survey on deep learning in medical image analysis. *Medical image analysis*, 42, 60-88. [Paper](https://www.sciencedirect.com/science/article/pii/S1361841516301831)

---

## License

This repository is licensed under the [MIT License](LICENSE).

---

> **For questions or collaboration, please open an issue or contact [@fahiiim](https://github.com/fahiiim).**

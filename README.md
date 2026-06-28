# Facial Expression Recognition Framework

### Facial Emotion Classification Using Custom CNN, MobileNetV2, and ResNet50 on RAF-DB

**Author:** Bhanu Prasad Mudraboina
**Matriculation ID:** 81013048
**Course:** Machine Learning — Master’s in Data Science
**Professor:** Raja Hashmi Ali
**Phase:** 3 — Final Report, Code Implementation, and Presentation

---

## Project Links

* **Dataset Link:** https://www.kaggle.com/datasets/nishchalchandel/raf-db-face-emotion-dataset/data
* **Public Kaggle Notebook Link:** https://www.kaggle.com/code/bhanuprasadspirit/ml-project-notebook
* **GitHub Repository Link:** https://github.com/Bhanu-ue05/Facial-Expression-Recognition-Framework
* **Overleaf Final Report Link:** https://www.overleaf.com/read/pptpcpxmqxkp#f52988

---

## Project Overview

This project implements a complete deep learning pipeline for **facial expression recognition** using the **RAF-DB Face Emotion Dataset**. The objective is to classify facial images into one of seven emotion categories using convolutional neural networks and transfer learning.

The project compares three CNN-based approaches:

1. **Custom CNN** trained from scratch.
2. **MobileNetV2** using transfer learning and fine-tuning.
3. **ResNet50** using transfer learning and fine-tuning.

The final notebook includes dataset preparation, preprocessing, augmentation, model training, evaluation, model comparison, Grad-CAM visualisation, SHAP explanation outputs, saved figures, saved tables, and trained model files.

---

## Dataset

* **Dataset:** RAF-DB Face Emotion Dataset
* **Source:** Kaggle
* **Task:** Facial emotion image classification
* **Image format:** RGB facial expression images
* **Image size used:** 224 × 224
* **Number of classes:** 7

### Emotion Classes

```text
angry
disgust
fear
happy
neutral
sad
surprise
```

### Dataset Split Used in Notebook

| Split      |     Images |
| ---------- | ---------: |
| Training   |     12,271 |
| Validation |      1,533 |
| Testing    |      1,535 |
| **Total**  | **15,339** |

---

## Research Questions

This project focuses on the following machine learning research questions:

1. Can CNN-based deep learning models classify facial expressions from RAF-DB images effectively?
2. Does transfer learning improve performance compared with a custom CNN trained from scratch?
3. Which model performs best among Custom CNN, MobileNetV2, and ResNet50?
4. How do the models perform across different emotion classes?
5. Can interpretability methods such as Grad-CAM and SHAP help explain model predictions?

---

## Models Implemented

| Model       | Type              | Training Strategy                             |
| ----------- | ----------------- | --------------------------------------------- |
| Custom CNN  | From scratch      | Main training with overfitting-aware stopping |
| MobileNetV2 | Transfer Learning | Head training followed by fine-tuning         |
| ResNet50    | Transfer Learning | Head training followed by fine-tuning         |

---

## Model Performance

| Model       | Accuracy | Macro F1-Score | Weighted F1-Score | Parameters |
| ----------- | -------: | -------------: | ----------------: | ---------: |
| Custom CNN  |   0.6984 |         0.5553 |            0.6876 |    457,927 |
| MobileNetV2 |   0.7342 |         0.6356 |            0.7300 |  2,587,719 |
| ResNet50    |   0.7537 |         0.6544 |            0.7486 | 24,114,055 |

**Best performing model:** ResNet50
**Best lightweight model:** MobileNetV2
**Smallest model:** Custom CNN

---

## Notebook Output Summary

The final notebook execution generated the following outputs:

| Output Type | Count |
| ----------- | ----: |
| Figures     |    15 |
| Tables      |    37 |
| Model files |     8 |

---

## Repository Structure

```text
Facial-Expression-Recognition-Framework/
│
├── FER Framework notebook.ipynb
├── Phase2_Proposal_Bhanu_Prasad_Mudraboina.docx
├── README.md
│
└── outputs/
    │
    ├── figures/
    │   ├── figure_01_dataset_samples.pdf
    │   ├── figure_02_class_distribution.pdf
    │   ├── figure_03_model_design_workflow.pdf
    │   ├── figure_04_all_models_training_curves.pdf
    │   ├── training_curves_Custom_CNN.pdf
    │   ├── training_curves_MobileNetV2.pdf
    │   ├── training_curves_ResNet50.pdf
    │   ├── confusion_matrix_Custom_CNN.pdf
    │   ├── confusion_matrix_MobileNetV2.pdf
    │   ├── confusion_matrix_ResNet50.pdf
    │   ├── figure_05_gradcam_correct_predictions.pdf
    │   ├── figure_06_gradcam_misclassified_predictions.pdf
    │   ├── figure_07_shap_explanations_Custom_CNN.pdf
    │   ├── figure_07_shap_explanations_MobileNetV2.pdf
    │   └── figure_07_shap_explanations_ResNet50.pdf
    │
    ├── tables/
    │   ├── dataset_summary.csv
    │   ├── class_distribution.csv
    │   ├── Custom_CNN_layer_summary.csv
    │   ├── MobileNetV2_layer_summary.csv
    │   ├── ResNet50_layer_summary.csv
    │   ├── model_architecture_table.csv
    │   ├── Custom_CNN_main_training_log.csv
    │   ├── MobileNetV2_head_training_log.csv
    │   ├── MobileNetV2_fine_tune_training_log.csv
    │   ├── ResNet50_head_training_log.csv
    │   ├── ResNet50_fine_tune_training_log.csv
    │   ├── Custom_CNN_main_history.csv
    │   ├── MobileNetV2_head_history.csv
    │   ├── MobileNetV2_fine_tune_history.csv
    │   ├── ResNet50_head_history.csv
    │   ├── ResNet50_fine_tune_history.csv
    │   ├── training_time_and_parameters.csv
    │   ├── all_model_training_history.csv
    │   ├── classification_report_Custom_CNN.csv
    │   ├── classification_report_MobileNetV2.csv
    │   ├── classification_report_ResNet50.csv
    │   ├── confusion_matrix_Custom_CNN.csv
    │   ├── confusion_matrix_MobileNetV2.csv
    │   ├── confusion_matrix_ResNet50.csv
    │   ├── test_predictions_Custom_CNN.csv
    │   ├── test_predictions_MobileNetV2.csv
    │   ├── test_predictions_ResNet50.csv
    │   ├── model_evaluation_metrics.csv
    │   ├── all_models_test_predictions.csv
    │   ├── final_model_comparison_table.csv
    │   ├── gradcam_selected_correct_examples.csv
    │   ├── gradcam_selected_misclassified_examples.csv
    │   ├── shap_explanation_summary_Custom_CNN.csv
    │   ├── shap_explanation_summary_MobileNetV2.csv
    │   ├── shap_explanation_summary_ResNet50.csv
    │   ├── research_question_answer_helpers.csv
    │   └── output_file_inventory.csv
    │
    └── models/
        ├── Custom_CNN_best.keras
        ├── Custom_CNN_main_best.weights.h5
        ├── MobileNetV2_best.keras
        ├── MobileNetV2_head_best.weights.h5
        ├── MobileNetV2_fine_tune_best.weights.h5
        ├── ResNet50_best.keras
        ├── ResNet50_head_best.weights.h5
        ├── ResNet50_fine_tune_best.weights.h5
        └── large_models_note.txt
```

---

## Complete Model Files Generated

The notebook generated a total of **8 model files**:

| File Name                               | Description                       |
| --------------------------------------- | --------------------------------- |
| `Custom_CNN_best.keras`                 | Full saved Custom CNN model       |
| `Custom_CNN_main_best.weights.h5`       | Best Custom CNN weights           |
| `MobileNetV2_best.keras`                | Full saved MobileNetV2 model      |
| `MobileNetV2_head_best.weights.h5`      | MobileNetV2 head training weights |
| `MobileNetV2_fine_tune_best.weights.h5` | MobileNetV2 fine-tuned weights    |
| `ResNet50_best.keras`                   | Full saved ResNet50 model         |
| `ResNet50_head_best.weights.h5`         | ResNet50 head training weights    |
| `ResNet50_fine_tune_best.weights.h5`    | ResNet50 fine-tuned weights       |

Some large model files may not be uploaded directly to GitHub because they exceed GitHub browser upload limits. Those files can be regenerated by running the notebook. The unavailable large files are documented in:

```text
outputs/models/large_models_note.txt
```

---

## Output Tables

The notebook generated **37 CSV table files**:

```text
dataset_summary.csv
class_distribution.csv
Custom_CNN_layer_summary.csv
MobileNetV2_layer_summary.csv
ResNet50_layer_summary.csv
model_architecture_table.csv
Custom_CNN_main_training_log.csv
MobileNetV2_head_training_log.csv
MobileNetV2_fine_tune_training_log.csv
ResNet50_head_training_log.csv
ResNet50_fine_tune_training_log.csv
Custom_CNN_main_history.csv
MobileNetV2_head_history.csv
MobileNetV2_fine_tune_history.csv
ResNet50_head_history.csv
ResNet50_fine_tune_history.csv
training_time_and_parameters.csv
all_model_training_history.csv
classification_report_Custom_CNN.csv
classification_report_MobileNetV2.csv
classification_report_ResNet50.csv
confusion_matrix_Custom_CNN.csv
confusion_matrix_MobileNetV2.csv
confusion_matrix_ResNet50.csv
test_predictions_Custom_CNN.csv
test_predictions_MobileNetV2.csv
test_predictions_ResNet50.csv
model_evaluation_metrics.csv
all_models_test_predictions.csv
final_model_comparison_table.csv
gradcam_selected_correct_examples.csv
gradcam_selected_misclassified_examples.csv
shap_explanation_summary_Custom_CNN.csv
shap_explanation_summary_MobileNetV2.csv
shap_explanation_summary_ResNet50.csv
research_question_answer_helpers.csv
output_file_inventory.csv
```

---

## Output Figures

The notebook generated **15 figure files**:

```text
figure_01_dataset_samples.pdf
figure_02_class_distribution.pdf
figure_03_model_design_workflow.pdf
figure_04_all_models_training_curves.pdf
training_curves_Custom_CNN.pdf
training_curves_MobileNetV2.pdf
training_curves_ResNet50.pdf
confusion_matrix_Custom_CNN.pdf
confusion_matrix_MobileNetV2.pdf
confusion_matrix_ResNet50.pdf
figure_05_gradcam_correct_predictions.pdf
figure_06_gradcam_misclassified_predictions.pdf
figure_07_shap_explanations_Custom_CNN.pdf
figure_07_shap_explanations_MobileNetV2.pdf
figure_07_shap_explanations_ResNet50.pdf
```

---

## Methodology

The complete implementation pipeline includes:

1. Loading RAF-DB train, validation, and test folders.
2. Resizing all images to 224 × 224 RGB.
3. Applying image preprocessing and data augmentation.
4. Training a Custom CNN model from scratch.
5. Training MobileNetV2 using transfer learning and fine-tuning.
6. Training ResNet50 using transfer learning and fine-tuning.
7. Evaluating models using accuracy, precision, recall, F1-score, and confusion matrices.
8. Comparing the three models using final evaluation metrics.
9. Generating Grad-CAM visual explanations.
10. Generating SHAP explanation outputs.
11. Saving all tables, figures, predictions, training histories, and model files.

---

## Experimental Setup

The implementation was completed using Python and TensorFlow/Keras in a Kaggle Notebook environment. GPU acceleration was used for model training. Model performance was evaluated using classification accuracy, precision, recall, F1-score, confusion matrices, and model comparison tables.

---

## How to Run the Notebook

### Kaggle Recommended

1. Open the public Kaggle notebook listed in the **Project Links** section.
2. Add the RAF-DB Face Emotion Dataset listed in the **Project Links** section.
3. Enable GPU accelerator.
4. Run all notebook cells from top to bottom.

The notebook output files are saved to:

```python
/kaggle/working/rafdb_outputs
```

---

## Requirements

The notebook uses the following main libraries:

```text
tensorflow
keras
numpy
pandas
matplotlib
Pillow
scikit-learn
shap
```

---

## Key Features

* Custom CNN model built from scratch.
* Transfer learning using MobileNetV2 and ResNet50.
* Overfitting-aware stopping during model training.
* CSVLogger, ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau callbacks.
* Complete model evaluation using classification reports and confusion matrices.
* Model comparison between Custom CNN, MobileNetV2, and ResNet50.
* Grad-CAM visualisation for correct and incorrect predictions.
* SHAP explainability for model interpretation.
* Complete saved outputs: 15 figures, 37 tables, and 8 model files.

---

## Phase 3 Final Submission Materials

The final Phase 3 submission includes:

1. Overleaf final report link.
2. Final PDF report.
3. Presentation video link or file.
4. Final GitHub repository link.
5. Final Kaggle Notebook link.

All project links are provided once in the **Project Links** section above.

---

## Key References

* Li, S., & Deng, W. (2020). Deep Facial Expression Recognition: A Survey. IEEE Transactions on Affective Computing.
* Li, S., Deng, W., & Du, J. (2017). Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild. CVPR.
* He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.
* Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR.
* Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. ICCV.

---

## Author

**Bhanu Prasad Mudraboina**
**Matriculation ID:** 81013048
Master’s in Data Science
Machine Learning Project Phase 3 — Final Report and Presentation
Facial Expression Recognition Framework

**Matriculation ID:** 81013048

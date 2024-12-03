# AlexNet Model Card

## Model Description

- **Model Name:** Plant Classification using Convolutional Neural Networks
- **Model Type:** AlexNet CNN
- **Purpose:** This model is used to classify 30 different plant species based on images. 

**Input:** 
- RGB images transformed as follows:
 Training:
 - Resize to 256x256
 - Random crop to 224x224
 - Random horizontal flip
 - Color jitter (brightness=0.2, contrast=0.2)
 - ToTensor conversion
 - Normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)

 Evaluation:
 - Resize to 256x256
 - Center crop to 224x224
 - ToTensor conversion
 - Normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)

**Output:**
- Probability distribution across num_classes
- Values between 0-1 representing confidence for each class
- Sum of probabilities equals 1

**Model Architecture:**
Features Network:
- Conv1: 3->64 channels, 11x11 kernel, stride 4, padding 2, ReLU
- MaxPool1: 3x3 kernel, stride 2
- Conv2: 64->192 channels, 5x5 kernel, padding 2, ReLU
- MaxPool2: 3x3 kernel, stride 2
- Conv3: 192->384 channels, 3x3 kernel, padding 1, ReLU
- Conv4: 384->256 channels, 3x3 kernel, padding 1, ReLU
- Conv5: 256->256 channels, 3x3 kernel, padding 1, ReLU
- MaxPool3: 3x3 kernel, stride 2

Classifier:
- Dropout
- Linear: 256 * 6 * 6 -> 4096, ReLU
- Dropout
- Linear: 4096 -> 4096, ReLU
- Linear: 4096 -> num_classes



## Performance
### Training Metrics
   <div>
    <img style="width:1000px" src="https://github.com/jefftl/Capstone-Project-2024/blob/main/images/alexnet_time.png">
   </div>

- Training progression:
 - Training loss decreases from ~3.3 to ~0.15
 - Validation loss decreases from ~3.0 to ~0.85
 - Validation accuracy improves to ~82%
 - Initial epoch time ~8000s, stabilizing to ~200s

### Classification Results
   <div>
    <img style="width:1000px" src="https://github.com/jefftl/Capstone-Project-2024/blob/main/images/alexnet.png">
   </div>

- Final model achieves:
 - Overall test accuracy: 75.13%
 - High variation in per-class performance
 - Best performing classes:
   - Peperchili (93.5% accuracy)
   - Paddy (90.0% accuracy) 
   - Shallot (90.0% accuracy)

- Detailed metrics for final epoch:
 - Training loss: 0.1385
 - Validation loss: 0.8596
 - Validation accuracy: 81.67%

### Notable Class Performances
- Strong performers (>85% accuracy):
 - Shallot (93.5%)
 - Waterapple (90.0%)
 - Eggplant (90.0%)
 - Watermelon (91.0%)

- Poor performers (<65% accuracy):
 - Canteloupe (9.0%)
 - Coconut (43.0%)
 - Spinach (59.5%)
 - Mango (62.0%)
 - There is also signifigant confusion between cateloupe's and melons

## Limitations
- Significant performance inconsistency:
 - Large gap between best and worst performing classes (84.5%)
 - Complete failure on some classes (<10% accuracy)
 - High confusion between visually similar classes
 
- Training characteristics:
 - Very high initial epoch time (~8000s)
 - Significant overfitting after epoch 30
 - Large divergence between training and validation loss
 - Unstable validation accuracy progression

- Resource intensive:
 - High initial computational requirements
 - Large memory footprint
 - Long training time to reach convergence

## Trade-offs
- Accuracy vs Resources:
 - Lower accuracy (75.13%) despite high computational cost
 - Requires significant GPU memory
 - Long training times, especially initially

- Model Architecture:
 - Simple architecture enables faster inference
 - But limits feature extraction capability
 - Struggles with complex visual distinctions

- Training Stability vs Performance:
 - Fast initial learning
 - But shows significant overfitting
 - Unstable validation metrics

- Class Balance:
 - Some classes show excellent performance (>90%)
 - Others show very poor performance (<10%)
 - Large performance disparity between classes

- Practicality:
 - Simpler implementation than modern architectures
 - Lower memory requirements during inference
 - But significantly lower accuracy and reliability

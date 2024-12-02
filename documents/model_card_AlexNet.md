# AlexNet Model Card

## Model Description

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
  - Training loss decreases smoothly from ~3.3 to ~0.2
  - Validation loss decreases from ~3.3 to ~0.8
  - Validation accuracy improves steadily to ~83%
  - Initial epoch time ~2300s, stabilizing to ~150s after epoch 5

### Classification Results
   <div>
    <img style="width:1000px" src="https://github.com/jefftl/Capstone-Project-2024/blob/main/images/alexnet.png">
   </div>

- Final model achieves:
- Strong performers:
  - Longbeans (0.91)
  - Papaya (0.91)
  - Peperchili (0.91)
  - Watermelon (0.92)
  - Cucumber (0.92)
  - Waterapple (0.92)
- Challenging classes:
  - Melon (0.02)
  - Cantaloupe (0.59)
  - Coconut (0.66)
  - Mango (0.68)

## Limitations
- Model shows significant performance variance:
  - Near complete failure on melon class (1% accuracy)
  - Struggles with certain fruits (cantaloupe, coconut, mango)
  - Notable overfitting after epoch 20 (diverging training/validation loss)
- Computational demands:
  - High initial epoch time (~2300s)
  - Large memory requirements
  - Significant training time to reach convergence
- Classification challenges:
  - Confusion between visually similar classes
  - Inconsistent performance across different plant types
  - Some classes show poor precision-recall balance

## Trade-offs
- Accuracy vs Computational Cost:
  - Achieves 81.48% accuracy but requires substantial computational resources
  - Long training times (~150s per epoch after stabilization)
  - High memory usage due to deep architecture
- Model Complexity vs Performance:
  - Deep network provides good feature extraction
  - But shows diminishing returns after epoch 40
  - Significant gap between training and validation loss
- Class Balance:
  - Excellent performance on some classes (>90%)
  - Poor performance on others (<70%)
  - Trade-off between overall accuracy and class-specific reliability
- Training Stability vs Speed:
  - Stable training progression
  - But requires many epochs to reach optimal performance
  - Clear overfitting trends in later epochs

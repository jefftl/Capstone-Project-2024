# VGG16 Model Card

## Model Description

**Input:** 
- RGB images transformed as follows:
 Training:
 - Resize to 256x256
 - Random crop to 224x224
 - Random horizontal flip
 - Random rotation (±15 degrees)
 - Color jitter (brightness=0.2, contrast=0.2, saturation=0.2)
 - ToTensor conversion
 - Normalize with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

 Evaluation:
 - Resize to 256x256
 - Center crop to 224x224
 - ToTensor conversion
 - Normalize with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

**Output:**
- Probability distribution across num_classes
- Values between 0-1 representing confidence for each class
- Sum of probabilities equals 1

**Model Architecture:**
Features Network (5 blocks):
- Block 1:
 - Conv 3->64, 3x3, pad 1, ReLU
 - Conv 64->64, 3x3, pad 1, ReLU
 - MaxPool 2x2, stride 2

- Block 2:
 - Conv 64->128, 3x3, pad 1, ReLU
 - Conv 128->128, 3x3, pad 1, ReLU
 - MaxPool 2x2, stride 2

- Block 3:
 - Conv 128->256, 3x3, pad 1, ReLU
 - Conv 256->256, 3x3, pad 1, ReLU
 - Conv 256->256, 3x3, pad 1, ReLU
 - MaxPool 2x2, stride 2

- Block 4:
 - Conv 256->512, 3x3, pad 1, ReLU
 - Conv 512->512, 3x3, pad 1, ReLU
 - Conv 512->512, 3x3, pad 1, ReLU
 - MaxPool 2x2, stride 2

- Block 5:
 - Conv 512->512, 3x3, pad 1, ReLU
 - Conv 512->512, 3x3, pad 1, ReLU
 - Conv 512->512, 3x3, pad 1, ReLU
 - MaxPool 2x2, stride 2

Classifier:
- Flatten
- Linear: 512*7*7 -> 4096, ReLU
- Dropout
- Linear: 4096 -> 4096, ReLU
- Dropout
- Linear: 4096 -> num_classes

Initialization:
- Conv layers: Kaiming normal (fan_out, relu)
- Conv biases: Constant(0)
- Linear weights: Normal(0, 0.01)
- Linear biases: Constant(0)

## Performance
### Training Metrics
   <div>
    <img style="width:1000px" src="https://github.com/jefftl/Capstone-Project-2024/blob/main/images/VGG_time.png">
   </div>

- Training progression:
  - Training loss decreases smoothly from ~3.3 to ~0.2
  - Validation loss decreases from ~3.3 to ~0.8
  - Validation accuracy improves steadily to ~83%
  - Initial epoch time ~2300s, stabilizing to ~150s after epoch 5

### Classification Results
   <div>
    <img style="width:1000px" src="https://github.com/jefftl/Capstone-Project-2024/blob/main/images/vgg.png">
   </div>

- Final model achieves:
  - Overall test accuracy: 81.48%
  - Most classes achieve >80% accuracy
  - Highest performing classes:
    - Shallot (97.5% accuracy)
    - Peperchili (94.5% accuracy)
    - Galangal (94.0% accuracy)

   Notable class performances:
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

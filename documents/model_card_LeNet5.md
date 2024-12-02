# LeNet-5 Model Card

## Model Description

**Input:** 
- RGB images resized to 32x32 pixels
- Normalized using mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Data augmentation during training:
  - Random horizontal flips
  - Random rotation (±10 degrees)
  - Random affine translations (±10%)
  - ToTensor conversion
  - Normalization

**Output:**
- Probability distribution across num_classes
- Values between 0-1 representing confidence for each class
- Sum of probabilities equals 1

**Model Architecture:**
- Modified LeNet-5 for RGB images
- Structure:
  - Feature Extractor:
    - Conv1: 3->6 channels, 5x5 kernel
    - ReLU activation
    - MaxPool 2x2
    - Conv2: 6->16 channels, 5x5 kernel
    - ReLU activation
    - MaxPool 2x2
  - Classifier:
    - Flatten layer
    - Linear: 16*5*5 -> 120
    - ReLU activation
    - Linear: 120 -> 84
    - ReLU activation
    - Linear: 84 -> num_classes
- Uses ReLU instead of traditional tanh
- Uses MaxPool instead of traditional AvgPool
- Adapted for RGB input (3 channels)
  
## Performance
### Training Metrics
   <div>
    <img style="width:1000px" src="https://github.com/jefftl/Capstone-Project-2024/blob/main/images/lenet_time.png">
   </div>

- Training progression:
 - Training loss decreases from ~3.3 to ~0.15
 - Validation loss decreases from ~3.0 to ~0.85
 - Validation accuracy improves to ~82%
 - Initial epoch time ~8000s, stabilizing to ~200s

### Classification Results
   <div>
    <img style="width:1000px" src="https://github.com/jefftl/Capstone-Project-2024/blob/main/images/lenet5.png">
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
 - Average epoch time: ~120 seconds

### Notable Class Performances
- Strong performers (>85% accuracy):
 - Peperchili (93.5%)
 - Paddy (90.0%)
 - Shallot (90.0%)
 - Watermelon (91.0%)

- Poor performers (<65% accuracy):
 - Bilimbi (9.0%)
 - Orange (63.5%)
 - Spinach (59.5%)
 - Mango (62.0%)

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

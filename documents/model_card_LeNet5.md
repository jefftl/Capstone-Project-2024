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
 - Training loss decreases from ~2.9 to ~1.9
 - Validation loss decreases from ~2.5 to ~1.9
 - Validation accuracy plateaus at ~44%
 - Initial epoch time ~2500s, stabilizing to ~100s

### Classification Results
   <div>
    <img style="width:1000px" src="https://github.com/jefftl/Capstone-Project-2024/blob/main/images/lenet5.png">
   </div>

- Final model achieves:
 - Overall test accuracy: 42.82%
 - High variance in class performance
 - Best performing classes:
   - Watermelon (74.0% accuracy)
   - Papaya (71.5% accuracy)
   - Eggplant (65.0% accuracy)

### Detailed Performance Analysis
- Strong performers (>60% accuracy):
 - Watermelon: 74.0%
 - Papaya: 71.5%
 - Eggplant: 65.0%
 - Waterapple: 65.0%

- Poor performers (<30% accuracy):
 - Bilimbi: 9.5%
 - Cantaloupe: 7.5%
 - Mango: 14.0%
 - Kale: 24.5%

- Average metrics:
 - Macro avg F1-score: 0.42
 - Weighted avg F1-score: 0.42

## Limitations
- Severe performance issues:
 - Very low overall accuracy (42.82%)
 - Many classes performing below 30% accuracy
 - Large disparity between best and worst classes
 - High confusion between classes

- Training limitations:
 - Quick plateau in validation accuracy
 - Limited improvement after epoch 10
 - High initial computational requirements
 - Minimal convergence in loss after early epochs

- Architectural constraints:
 - Simple architecture struggles with complex dataset
 - Limited feature extraction capability
 - Poor performance on fine-grained distinctions
 - Inadequate capacity for 30-class problem

## Trade-offs
- Simplicity vs Performance:
 - Simple architecture enables fast inference
 - But severely limits classification capability
 - Significant accuracy sacrifice for simplicity

- Resource Usage vs Results:
 - Lower memory requirements
 - Faster training after initial epoch
 - But delivers poor classification performance

- Learning Capacity:
 - Quick initial learning
 - But early plateau in performance
 - Unable to capture complex class differences

- Practical Considerations:
 - Lightweight deployment
 - Fast inference time
 - But accuracy too low for practical use
 - Not suitable for real-world applications with this dataset

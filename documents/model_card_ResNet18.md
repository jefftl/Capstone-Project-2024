# ResNet18 Model Card

## Model Description

- **Model Name:** Plant Classification Convolutional Neural Net
- **Model Type:** ResNet18 CNN
- **Purpose:** This model is used to classify 30 different plant species based on images.  

**Input:** 
- RGB images transformed as follows:
 Training:
 - Resize to 256x256
 - Random crop to 224x224
 - Random horizontal flip
 - Random rotation (±15 degrees)
 - Color jitter (brightness=0.2, contrast=0.2, saturation=0.2)
 - Normalize with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

 Evaluation:
 - Resize to 256x256
 - Center crop to 224x224
 - Normalize with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

**Output:**
- Probability distribution across num_classes
- Values between 0-1 representing confidence for each class
- Sum of probabilities equals 1

**Model Architecture:**
- 18-layer deep residual network
- Structure:
 - Initial Conv: 7x7, 64 filters, stride 2
 - BatchNorm + ReLU
 - MaxPool: 3x3, stride 2
 - Layer1: 2 BasicBlocks (64 channels)
 - Layer2: 2 BasicBlocks (128 channels)
 - Layer3: 2 BasicBlocks (256 channels)
 - Layer4: 2 BasicBlocks (512 channels)
 - Adaptive Average Pooling to 1x1
 - Fully Connected: 512 -> num_classes

BasicBlock Structure:
- Two 3x3 conv layers with BatchNorm and ReLU
- Shortcut connection that can handle different dimensions
- Kaiming normal weight initialization
- BatchNorm initialization with weight=1, bias=0

## Performance
- Training metrics show:
  - Training loss decreases smoothly from ~3.0 to ~0.1
  - Validation loss decreases from ~2.5 to ~0.5
  - Validation accuracy improves steadily to ~85%
  - High initial epoch time (~4000s) that stabilizes to ~100s

### Training Curves
   <div>
    <img style="width:1000px" src="https://github.com/jefftl/Capstone-Project-2024/blob/main/images/resnet_time.png">
   </div>

### Classification Results
   <div>
    <img style="width:1000px" src="https://github.com/jefftl/Capstone-Project-2024/blob/main/images/resnet.png">
   </div>

- Final model achieves:
  - Overall accuracy of 85.37%
  - Most classes achieve >85% accuracy
  - Some classes show perfect or near-perfect classification (>95%)
- Best performing classes:
  - Watermelon (0.95)
  - Longbeans (0.94)
  - Peperchili (0.94)
  - Paddy (0.94)
- Challenging classes:
  - Melon (0.00)
  - Cantaloupe (0.62)
  - Coconut (0.72)
  - Banana (0.74)
   
## Limitations
- Shows class-specific weaknesses:
  - Some classes show significantly lower accuracy (e.g., Cantaloupe at ~63.5%)
  - Confusion between visually similar classes (visible in confusion matrix)
  - Uneven performance across different plant types
- Training characteristics:
  - Long initial epoch time (~4000s)
  - Takes ~30 epochs to reach optimal performance
  - Shows signs of plateauing after 40 epochs
  - Persistent gap between training and validation loss indicating some overfitting

## Trade-offs
- Accuracy vs Training Time:
  - Achieves good accuracy (85.37%) but requires significant training time
  - Initial epochs are particularly time-intensive
- Class Balance vs Overall Performance:
  - Performs very well on majority of classes
  - Some classes suffer from lower accuracy
  - Trade-off between overall accuracy and per-class consistency
- Model Complexity vs Results:
  - Complex enough to achieve good accuracy
  - But shows diminishing returns after 40 epochs
  - Some classes might benefit from more specialized architecture
- Generalization vs Specialization:
  - Good general performance across most classes
  - Struggles with specific challenging cases
  - Balance between broad applicability and specialized recognition
--- 

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
- Training approach:
  - Modern data augmentation techniques
  - Evaluation transformation:
    - Resize to 32x32
    - ToTensor conversion
    - Standard normalization
- Metrics monitored:
  - Training/validation loss
  - Validation accuracy
  - Per-class accuracy
  - Confusion matrix

## Limitations
- Small input size (32x32) may lose detail
- Relatively shallow architecture
- Limited receptive field
- May struggle with:
  - Complex natural images
  - Fine-grained visual differences
  - Large-scale classification tasks
- Simple architecture compared to modern standards
- No batch normalization or dropout

## Trade-offs
- Architecture Simplicity vs Capacity:
  - Clean, straightforward implementation
  - Limited feature extraction capability
- Input Size vs Detail:
  - Small input size processes quickly
  - May miss fine details in larger images
- Memory vs Depth:
  - Very efficient memory usage
  - Shallow network limits learning capacity
- Training Speed vs Sophistication:
  - Fast training and inference
  - Lacks modern architectural improvements

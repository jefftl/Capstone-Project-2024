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
- Training approach:
 - Basic data augmentation
   - Random cropping
   - Horizontal flipping
   - Color jittering
 - Dropout layers for regularization
 - ReLU activations for non-linearity
 - Large initial conv layer stride for efficient processing
- Metrics monitored:
 - Training/validation loss
 - Validation accuracy
 - Per-class accuracy
 - Confusion matrix

## Limitations
- Large kernel size (11x11) in first layer may miss fine details
- Aggressive stride in first conv layer (4) reduces spatial information
- No batch normalization
- Fixed input size requirements
- May struggle with:
 - Very small objects
 - Complex textures
 - Modern high-resolution tasks
- Memory intensive fully connected layers

## Trade-offs
- Architecture Design:
 - Large kernels process more context but lose detail
 - Aggressive downsampling reduces computation but loses spatial information
- Memory Usage:
 - Large fully connected layers (4096 neurons)
 - High memory requirement for feature maps
- Regularization:
 - Dropout helps prevent overfitting
 - But may need longer training time
- Input Processing:
 - Simple augmentation pipeline
 - Limited color augmentation (only brightness and contrast)

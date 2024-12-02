# LeNet-5 Model Card

## Model Description

**Input:** 
- Grayscale images resized to 32x32 pixels
- Normalized using mean=[0.5] and std=[0.5]
- Simple preprocessing with no data augmentation for original implementation
- Can be adapted for RGB images with minor modifications

**Output:**
- Probability distribution across output classes 
- Values between 0-1 representing confidence for each class
- Sum of probabilities equals 1
- Originally designed for 10 digit classes (MNIST)

**Model Architecture:**
- Classic 7-layer convolutional neural network
- Structure:
 - Conv1: 6 filters of size 5x5, stride 1
 - Average pooling: 2x2, stride 2
 - Conv2: 16 filters of size 5x5, stride 1 
 - Average pooling: 2x2, stride 2
 - Conv3: 120 filters of size 5x5, stride 1
 - Fully connected: 84 neurons
 - Output layer: num_classes neurons
- Activation functions: Tanh (originally), ReLU (modern versions)
- Total parameters: ~60K (much smaller than modern architectures)

## Performance
- Training approach:
 - SGD optimizer
 - Learning rate: typically 0.01
 - Simple step learning rate decay
 - Batch size: flexible, commonly 32-128
 - No modern training techniques required
- Metrics monitored:
 - Training/validation loss
 - Validation accuracy
 - Per-class accuracy
 - Confusion matrix
- Historical performance:
 - Achieved ~99% accuracy on MNIST
 - Set benchmark for digit recognition in 1990s

## Limitations
- Very simple architecture by modern standards
- Limited capacity for complex features
- Originally designed for small grayscale images
- May struggle with:
 - High resolution images
 - Complex natural images
 - Large numbers of classes
 - Fine-grained visual differences
- No modern architectural features like:
 - Batch normalization
 - Residual connections
 - Dropout
 - Advanced pooling methods

## Trade-offs
- Simplicity vs Power:
 - Very simple to implement and understand
 - Limited learning capacity compared to modern networks
- Resource Usage vs Capability:
 - Extremely lightweight (~60K parameters)
 - May not capture complex features
- Training Speed vs Accuracy:
 - Fast training and inference
 - Lower accuracy ceiling than modern architectures
- Input Constraints vs Flexibility:
 - Optimized for small images
 - Needs significant modification for modern tasks
- Historical Value vs Modern Use:
 - Excellent educational tool
 - Still useful for simple tasks
 - Not suitable for state-of-the-art applications

# AlexNet Model Card

## Model Description

**Input:**
- RGB images resized to 224x224 pixels
- Normalized using mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Data augmentation during training: random crops, horizontal flips, color jittering

**Output:**
- Probability distribution across N classes
- Values between 0-1 representing confidence for each class
- Sum of probabilities equals 1

**Model Architecture:**
- 8 layers (5 convolutional + 3 fully connected)
- Large initial filter size (11x11)
- Structure:
  - 5 convolutional layers with max pooling
  - 3 fully connected layers
  - ReLU activations
  - Dropout for regularization
- Total parameters: ~61 million

## Performance
- Training approach:
  - SGD optimizer with momentum 0.9
  - Initial learning rate: 0.01
  - Cosine annealing scheduler
  - Batch size: 64
  - Mixed precision training
- Metrics monitored:
  - Training/validation loss
  - Validation accuracy
  - Per-class accuracy
  - Confusion matrix

## Limitations
- Relatively shallow architecture
- Large filter sizes reduce spatial information
- No modern architectural features (no residual connections, limited batch normalization)
- May struggle with:
  - Complex feature hierarchies
  - Fine-grained classification
  - Modern high-resolution images
- Limited feature reuse

## Trade-offs
- Simplicity vs Capability:
  - Simple architecture, easy to understand
  - Limited feature extraction capability
- Speed vs Accuracy:
  - Faster training and inference
  - Generally lower accuracy than modern architectures
- Memory vs Depth:
  - Moderate memory requirements
  - Shallower architecture limits learning capacity
- Training Stability vs Performance:
  - More stable training due to simplicity
  - May not achieve state-of-the-art performance 

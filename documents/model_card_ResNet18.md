# ResNet18 Model Card

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
- 18-layer deep residual network
- Residual connections to combat vanishing gradients
- Structure:
  - Initial 7x7 conv layer, stride 2
  - 4 layers of residual blocks (2 blocks each)
  - Global average pooling
  - Final fully connected layer
- Total parameters: ~11.7 million

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
- Fixed input size requirement (224x224)
- Limited receptive field in early layers
- May struggle with:
  - Very small objects
  - Fine-grained visual differences
  - Highly cluttered scenes
- Requires careful learning rate tuning
- Performance depends on quality of residual block optimization

## Trade-offs
- Speed vs Depth:
  - Faster than deeper models like VGG16
  - May miss some complex features
- Memory vs Accuracy:
  - Moderate memory footprint
  - Good balance of accuracy and resource usage
- Training stability vs Speed:
  - Residual connections improve training
  - Additional computations for skip connections
- Batch size limitations:
  - Smaller batches possible
  - May affect batch normalization effectiveness

--- 

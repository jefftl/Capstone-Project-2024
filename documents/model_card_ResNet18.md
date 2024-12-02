# ResNet18 Model Card

## Model Description

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
- Training approach:
 - Extensive data augmentation
 - Separate evaluation transforms
 - Kaiming weight initialization
 - BatchNorm for training stability
 - Residual connections for gradient flow
- Metrics monitored:
 - Training/validation loss
 - Validation accuracy
 - Per-class accuracy
 - Confusion matrix

## Limitations
- Fixed input processing pipeline
- Memory requirements increase with batch size
- Requires careful learning rate tuning
- May struggle with:
 - Very small objects due to aggressive downsampling
 - Extreme rotations beyond ±15 degrees
 - Color variations beyond jittering parameters
- BatchNorm dependencies between training/inference

## Trade-offs
- Resolution vs Speed:
 - Initial high resolution (256) provides detail
 - Multiple stride-2 operations reduce computation
- Depth vs Complexity:
 - 18 layers deep, but efficient residual design
 - Each BasicBlock adds parameters but improves feature learning
- Augmentation vs Training Time:
 - Comprehensive augmentation improves robustness
 - Increases training time per epoch
- Memory vs Batch Size:
 - BatchNorm requires reasonable batch sizes
 - GPU memory limits maximum batch size
--- 

## Session 7: Advanced Training Concepts

In this assignment we are supposed to train a CIFAR10 dataset using Resnet18 architecture.
Also we need to show the layer activations using **gradcam**


## Files
I am making use of a code repository [StarterKit](https://github.com/TSAI-EVA8/StarterKit)

* The repo is first cloned in this project using 
```
git clone https://github.com/TSAI-EVA8/StarterKit
```

* The trainiing notebook is [Session7_Solution.ipynb](https://github.com/TSAI-EVA8/advancedTraining/blob/master/Session7_Solution.ipynb) which makes use of the code in the [StarterKit](https://github.com/TSAI-EVA8/StarterKit) to perform the training. All the logic is present in the [StarterKit](https://github.com/TSAI-EVA8/StarterKit)

## Transformations
I have used the following transformations
* transforms.RandomCrop(32,4)
* transforms.RandomRotation((-18.0, 18.0), fill=(1,))
* Cutout(n_holes=1, length=16) . The Cutout was implemented from scratch as it is not present in the torchvision.transforms

Following is a set of images after transformations (training images)

![alt text](images/training_sample.png "Title")



I have not applied any transformation on the testing imaages as shown below
![alt text](images/test_sample.png "Title")

## Training Configuration
1. Epoch : 20
2. Batch Size: 64

Used the SGD optimizer with LR=0.01 and momentum=0.9

## Results
After training for 20 epochs the model achieved a test accuracy of 88.33%

The test accuracy is better than the training accuracy (84.21%) which shows that the model is doing better on the test images as we had made the training task hard by adding all the transformations

Here are the training logs
Epoch 1:
Loss=1.72 Batch_ID=781 Train_Accuracy=38.38: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:35<00:00, 21.81it/s]
Test set: Average loss: 0.0249, Accuracy: 4723/10000 (47.23%)

Epoch 2:
Loss=1.51 Batch_ID=781 Train_Accuracy=52.32: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:36<00:00, 21.62it/s]
Test set: Average loss: 0.0175, Accuracy: 6110/10000 (61.10%)

Epoch 3:
Loss=1.67 Batch_ID=781 Train_Accuracy=59.49: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:36<00:00, 21.21it/s]
Test set: Average loss: 0.0143, Accuracy: 6792/10000 (67.92%)

Epoch 4:
Loss=1.50 Batch_ID=781 Train_Accuracy=64.69: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:36<00:00, 21.22it/s]
Test set: Average loss: 0.0143, Accuracy: 6964/10000 (69.64%)

Epoch 5:
Loss=1.61 Batch_ID=781 Train_Accuracy=67.72: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:36<00:00, 21.15it/s]
Test set: Average loss: 0.0121, Accuracy: 7513/10000 (75.13%)

Epoch 6:
Loss=1.20 Batch_ID=781 Train_Accuracy=70.49: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:37<00:00, 21.10it/s]
Test set: Average loss: 0.0097, Accuracy: 7903/10000 (79.03%)

Epoch 7:
Loss=0.93 Batch_ID=781 Train_Accuracy=72.65: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:37<00:00, 21.03it/s]
Test set: Average loss: 0.0093, Accuracy: 8021/10000 (80.21%)

Epoch 8:
Loss=0.75 Batch_ID=781 Train_Accuracy=74.34: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:37<00:00, 21.06it/s]
Test set: Average loss: 0.0113, Accuracy: 7627/10000 (76.27%)

Epoch 9:
Loss=0.58 Batch_ID=781 Train_Accuracy=75.96: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:37<00:00, 20.95it/s]
Test set: Average loss: 0.0090, Accuracy: 8123/10000 (81.23%)

Epoch 10:
Loss=0.64 Batch_ID=781 Train_Accuracy=76.94: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:37<00:00, 20.92it/s]
Test set: Average loss: 0.0092, Accuracy: 8082/10000 (80.82%)

Epoch 11:
Loss=0.34 Batch_ID=781 Train_Accuracy=78.04: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:37<00:00, 21.00it/s]
Test set: Average loss: 0.0084, Accuracy: 8319/10000 (83.19%)

Epoch 12:
Loss=0.29 Batch_ID=781 Train_Accuracy=78.96: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:37<00:00, 20.98it/s]
Test set: Average loss: 0.0082, Accuracy: 8268/10000 (82.68%)

Epoch 13:
Loss=0.49 Batch_ID=781 Train_Accuracy=79.94: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:37<00:00, 21.08it/s]
Test set: Average loss: 0.0075, Accuracy: 8360/10000 (83.60%)

Epoch 14:
Loss=0.21 Batch_ID=781 Train_Accuracy=80.47: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:37<00:00, 20.91it/s]
Test set: Average loss: 0.0072, Accuracy: 8537/10000 (85.37%)

Epoch 15:
Loss=0.95 Batch_ID=781 Train_Accuracy=81.35: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:37<00:00, 20.94it/s]
Test set: Average loss: 0.0071, Accuracy: 8447/10000 (84.47%)

Epoch 16:
Loss=0.44 Batch_ID=781 Train_Accuracy=81.83: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:37<00:00, 21.00it/s]
Test set: Average loss: 0.0068, Accuracy: 8591/10000 (85.91%)

Epoch 17:
Loss=0.19 Batch_ID=781 Train_Accuracy=82.58: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:37<00:00, 20.92it/s]
Test set: Average loss: 0.0063, Accuracy: 8706/10000 (87.06%)

Epoch 18:
Loss=0.18 Batch_ID=781 Train_Accuracy=83.33: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:37<00:00, 20.95it/s]
Test set: Average loss: 0.0061, Accuracy: 8722/10000 (87.22%)

Epoch 19:
Loss=1.13 Batch_ID=781 Train_Accuracy=83.65: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:37<00:00, 20.87it/s]
Test set: Average loss: 0.0055, Accuracy: 8874/10000 (88.74%)

Epoch 20:
Loss=0.55 Batch_ID=781 Train_Accuracy=84.21: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:37<00:00, 20.96it/s]
Test set: Average loss: 0.0058, Accuracy: 8833/10000 (88.33%)




Here are the plots for the Training and Testing
![alt text](images/losses.png "Title")


## GradCam
We used the gradcam algorithm to test out the what the model is looking at different layers. Here are few images 
![alt text](images/gradcam_horse.png "Title")
![alt text](images/gradcam_ship.png "Title")

![alt text](images/gradcam_truck.png "Title")

## Misclassification & Class accuracy

Here are few of the misclassified images

![alt text](images/misclassification.png "Title")


The class level accuracy on the entire test data
```
Accuracy of plane : 87 %
Accuracy of   car : 94 %
Accuracy of  bird : 86 %
Accuracy of   cat : 72 %
Accuracy of  deer : 87 %
Accuracy of   dog : 85 %
Accuracy of  frog : 94 %
Accuracy of horse : 84 %
Accuracy of  ship : 96 %
Accuracy of truck : 94 %
```
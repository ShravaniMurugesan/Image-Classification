# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The objective of this project is to create a CNN that can categorize images of fashion items from the Fashion MNIST dataset. This dataset includes grayscale images of clothing and accessories such as T-shirts, trousers, dresses, and footwear. The task is to accurately predict the correct category for each image while ensuring the model is efficient and robust.


## Neural Network Model

<img width="1037" height="406" alt="image" src="https://github.com/user-attachments/assets/60881028-0fc5-41af-8e76-2ef55e94f1b2" />

## DESIGN STEPS


### STEP 1:
Import the necessary libraries such as NumPy, Matplotlib, and PyTorch.

### STEP 2:
Load and preprocess the dataset:

Resize images to a fixed size (128×128). Normalize pixel values to a range between 0 and 1. Convert labels into numerical format if necessary.

### STEP 3:
Define the CNN Architecture, which includes:

Input Layer: Shape (8,128,128) Convolutional Layer 1: 8 filters, kernel size (16×16), ReLU activation Max-Pooling Layer 1: Pool size (2×2) Convolutional Layer 2: 24 filters, kernel size (8×8), ReLU activation Max-Pooling Layer 2: Pool size (2×2) Fully Connected (Dense) Layer: First Dense Layer with 256 neurons Second Dense Layer with 128 neurons Output Layer for classification

### STEP 4:
Define the loss function (e.g., Cross-Entropy Loss for classification) and optimizer (e.g., Adam or SGD).

### STEP 5:
Train the model by passing training data through the network, calculating the loss, and updating the weights using backpropagation.

### STEP 6:
Evaluate the trained model on the test dataset using accuracy, confusion matrix, and other performance metrics.

### STEP 7:
Make predictions on new images and analyze the results.

## PROGRAM

### Name: SHRAVANI M
### Register Number: 212224230263
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

     def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x




```

```python
# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        
        print('Name: SHRAVANI M ')
        print('Register Number:  212224230263 ')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```
## OUTPUT
### Training Loss per Epoch

<img width="847" height="747" alt="image" src="https://github.com/user-attachments/assets/b9c2d075-7fed-43b9-9b74-11cf7418c282" />


### Confusion Matrix

<img width="1042" height="753" alt="image" src="https://github.com/user-attachments/assets/76da0a26-0839-4461-b050-453fa86856fa" />


### Classification Report

<img width="683" height="436" alt="image" src="https://github.com/user-attachments/assets/d26c8f51-f8d1-4c32-9213-623451bd13aa" />



### New Sample Data Prediction

<img width="664" height="623" alt="image" src="https://github.com/user-attachments/assets/e4801598-b65c-4ed4-8a8f-9f75d19a29c3" />


## RESULT

Thus the development of a convolutional deep neural network for image classification is executed successfully.

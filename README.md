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
        # write your code here
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128*3*3, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,10)

    def forward(self, x):
        # write your code here
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.relu(self.conv4(x)) # Removed the last pooling layer
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

```

```python
# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):

    # write your code here
    print('Name: SHRAVANI M')
    print('Register Number: 212224230263')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```
## OUTPUT
### Training Loss per Epoch

<img width="441" height="125" alt="image" src="https://github.com/user-attachments/assets/f68767b6-d8bd-4dcd-87f2-2b21c1bbaa64" />



### Confusion Matrix

<img width="1027" height="754" alt="image" src="https://github.com/user-attachments/assets/74b4a62e-0d0d-4148-9452-442fe95375db" />


### Classification Report

<img width="681" height="449" alt="image" src="https://github.com/user-attachments/assets/e3adf6ef-6af7-4527-b134-ca705d011103" />


### New Sample Data Prediction

<img width="676" height="627" alt="image" src="https://github.com/user-attachments/assets/d02b5a5c-f7f5-4406-bd72-8d81e73899c5" />



## RESULT

Thus the development of a convolutional deep neural network for image classification is executed successfully.

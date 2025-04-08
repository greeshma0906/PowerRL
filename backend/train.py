# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from carbontracker.tracker import CarbonTracker

# # Step 1: Load the MNIST dataset
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# # Step 2: Define a simple CNN model
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(32 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, 10)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.relu = nn.ReLU()
    
#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(-1, 32 * 7 * 7)  # Flatten the tensor
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Step 3: Train the model while tracking energy consumption
# def train_model():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = SimpleCNN().to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     max_epochs = 10
#     tracker = CarbonTracker(epochs=max_epochs)  # Start carbon tracking

#     for epoch in range(max_epochs):
#         tracker.epoch_start()  # Track energy at start of epoch

#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#         tracker.epoch_end()  # Track energy at end of epoch
#         print(f"Epoch [{epoch+1}/{max_epochs}] completed.")

#     tracker.stop()  # Stop tracking energy consumption

# # Run training
# if __name__ == "__main__":
#     train_model()
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from carbontracker.tracker import CarbonTracker

# Step 1: Load the CIFAR-10 dataset with data augmentation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform_train, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

# Step 2: Define a more complex model (ResNet-18)
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetClassifier, self).__init__()
        self.resnet = resnet18(weights=None)  # No pre-trained weights
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # Adjust final layer

    def forward(self, x):
        return self.resnet(x)

# Step 3: Train the model with CarbonTracker
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    max_epochs = 30  # Increased epochs for longer training time
    tracker = CarbonTracker(epochs=max_epochs)

    for epoch in range(max_epochs):
        tracker.epoch_start()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        tracker.epoch_end()
        print(f"Epoch [{epoch+1}/{max_epochs}] completed.")

    tracker.stop()

# Run training
if __name__ == "__main__":
    train_model()

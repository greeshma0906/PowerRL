import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from carbontracker.tracker import CarbonTracker

# Step 1: Define the test dataset and DataLoader
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform_test, download=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# Step 2: Define the same model structure used during training
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetClassifier, self).__init__()
        self.resnet = resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Step 3: Testing function with CarbonTracker
def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = ResNetClassifier().to(device)

    # Load trained model weights if saved (optional)
    # model.load_state_dict(torch.load("resnet_cifar10.pth"))

    #model.eval()  # Set model to evaluation mode
    #criterion = nn.CrossEntropyLoss()
    tracker = CarbonTracker(epochs=1)  # Only one test epoch needed
    correct = 0
    total = 0
    test_loss = 0.0

    tracker.epoch_start()
    print("hello")
    tracker.stop()

# Run testing
if __name__ == "__main__":
    test_model()

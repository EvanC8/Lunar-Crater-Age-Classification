# Converted from a Google Colab file

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from time import time
import matplotlib.pyplot as plt
import numpy as np

# Disable pixel limit for crater images
Image.MAX_IMAGE_PIXELS = None

# Gather TIFF image files
from google.colab import drive
drive.mount('/content/drive', force_remount = True)

folder = 'cropped_craters'

tiff_files = glob.glob(os.path.join(folder, "*.tiff"))
train_files, test_files = train_test_split(tiff_files, test_size=0.2, random_state=41)

# Chose feedforward parameters (Currently all)
params = ["Radius", "Transient cavity diameter [km]", "Floor diameter [km]", "Rim to floor depth [km]", "Apparent depth [km]", "Interior volume [km^3]", "Rim flank width [km]", "Height of central peak [km]"]
num_params = len(params)

# Create convolutional neural network
class CraterCNN(nn.Module):
    def __init__(self, embed_dim=128, dropout_rate=0.25):
        super().__init__()
        # Convolutions, pooling, and dropout
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc_input_size = self._get_fc_input_size(512, 512)
        self.fc1 = nn.Linear(self.fc_input_size, embed_dim)
        self.dropout_fc = nn.Dropout(dropout_rate)

    # Calculates the input size for the fully connected layer.
    def _get_fc_input_size(self, height, width):
        x = torch.randn(1, 1, height, width)
        x = self.pool1(F.relu(self.conv1(x)))
        
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = self.pool3(F.relu(self.conv3(x)))
        
        return x.view(1, -1).shape[1]  # return the flattened size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Output embedding
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        
        return x

# Create feed forward neural network
class ParamNet(nn.Module):
    def __init__(self, num_params=num_params, embed_dim=8, dropout_rate=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_params, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)

# Create classifier model CNN + FNN -> Age Classification
class CraterAgeNet(nn.Module):
    def __init__(self,
                 cnn_embed_dim=128,     # dimension from CNN
                 radius_embed_dim=8,    # dimension from FFN
                 num_params=num_params,
                 num_classes=6,         # There are 6 age categories
                 dropout_rate=0.20):
        super().__init__()

        self.cnn = CraterCNN(embed_dim=cnn_embed_dim)
        self.radius_net = ParamNet(num_params=num_params,embed_dim=radius_embed_dim)

        fused_dim = cnn_embed_dim + radius_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes),
        )

    def forward(self, images, radius):
        cnn_feat = self.cnn(images)
        radius_feat = self.radius_net(radius)

        fused = torch.cat([cnn_feat, radius_feat], dim=1)
        logits = self.classifier(fused)
        return logits

class MoonCraterDataset(Dataset):
    def __init__(self, image_paths, params_dict, labels_dict, transform=None):
        self.image_paths = image_paths
        self.df = df
        self.transform = transform
        self.params_dict = params_dict
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        # Suppose the crater name is part of the file name or can be extracted
        crater_name = img_path.split("/")[-1][:-5].replace("_", " ").replace("'", " ")

        # get params
        item_params = self.params_dict[crater_name]

        # get label
        # label
        label = self.labels_dict[crater_name]

        # radius needs to be a tensor of shape (num_params,)
        params_tensor = torch.tensor(item_params, dtype=torch.float)

        return image, params_tensor, label

def train_crater_age_model(model, train_loader, test_loader, num_epochs=5, lr=1e-3):
    train_accuracies = []
    test_accuracies = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()  # for multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training on device: {device}")
    start = time()
    for epoch in range(num_epochs):
        epoch_time = time()
        model.train()
        running_loss = 0.0

        for images, _params, labels in train_loader:
            images = images.to(device)
            _params = _params.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(images, _params)
            loss = criterion(logits, labels)

            # Back propogation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # --- Evaluate on training set ---
        print("Evaluating on TRAIN set:", end = " ")
        _, _, train_score = evaluate_model(model, train_loader)
        train_accuracies.append(train_score)

        # --- Evaluate on test set ---
        print("Evaluating on TEST set:", end = " ")
        _, _, test_score = evaluate_model(model, test_loader)
        test_accuracies.append(test_score)

        # Save epoch accuracies for analysis
        torch.save(model.state_dict(), f"combined_crater_age_net{trial_num}_epoch{epoch}.pth")
        print("Model saved.")
        print(f"Epoch time: {((time() - epoch_time)/60):.2f}m")
        print(f"Elapsed Time: {((time() - start)/60):.2f}m")
        print("------------")

    return model, train_accuracies, test_accuracies

def evaluate_model(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total = 0
    correct = 0

    # Collect all predictions/labels for advanced metrics
    all_preds = []
    all_labels = []

    with torch.no_grad():
      for images, _params, labels in val_loader:
            images, _params, labels = images.to(device), _params.to(device), labels.to(device)

            outputs = model(images, _params)
            preds = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")

    return all_preds, all_labels, accuracy

# 6 Classification Age Groups
ages = {"pre-Nectarian": 0,
        "Nectarian": 1,
        "Lower Imbrian": 2,
        "Upper Imbrian": 3,
        "Eratosthenian": 4,
        "Copernican": 5
        }


# Transformation on Image
transform = T.Compose([
    T.Resize((512, 512)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor()
])

params_dict = {}
labels_dict = {}

# Access feed forward parameters
df = pd.read_csv("/content/drive/My Drive/filtered.csv")

for i, row in df.iterrows():
    crater_name = row["Name"]
    params_dict[crater_name] = []
    for param in params:
        params_dict[crater_name].append(float(row[param]))
        
    # Convert Age to integer class
    labels_dict[crater_name] = ages[row["Age"].strip()]

# Build dataset
train_dataset = MoonCraterDataset(
    image_paths=train_files,
    params_dict=params_dict,
    labels_dict=labels_dict,
    transform=transform
)

test_dataset = MoonCraterDataset(
    image_paths=test_files,
    params_dict=params_dict,
    labels_dict=labels_dict,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=4, shuffle=False)

# Train Model:
# 1 Epoch is roughly 5 minutes
trial_num = "1"
epochs = 10
combined_model = CraterAgeNet(cnn_embed_dim=128, radius_embed_dim=8, num_classes=6)
trained_combined_model, train_accuracies, test_accuracies = train_crater_age_model(
    model=combined_model,
    train_loader=train_loader,
    test_loader=test_loader,
    num_epochs=epochs,
    lr=1e-3
)

torch.save(trained_combined_model.state_dict(), f"combined_crater_age_net{trial_num}.pth")
np.save(f'train_accuracies{trial_num}.npy', train_accuracies)
np.save(f'test_accuracies{trial_num}.npy', test_accuracies)

train_accuracies = np.load('/train_accuracies.npy')
test_accuracies = np.load('/test_accuracies.npy')

print("Train accuracies:", train_accuracies)
print("Test accuracies:", test_accuracies)

x_vals = np.arange(1,epochs+1,1)
plt.plot(x_vals, train_accuracies, label="Train")
plt.plot(x_vals, test_accuracies, label="Test")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


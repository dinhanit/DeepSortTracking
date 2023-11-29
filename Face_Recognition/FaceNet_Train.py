import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image

# Define the FaceNet model
class FaceNetModel(nn.Module):
    def __init__(self):
        super(FaceNetModel, self).__init__()
        self.facenet = InceptionResnetV1(pretrained='vggface2',classify=False)

    def forward(self, x):
        embeddings = self.facenet(x)
        return embeddings

# Define a custom dataset
class CustomFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            for filename in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, filename), self.class_to_idx[cls]))

        # Ensure the dataset size is a multiple of 3
        self.samples = self.samples[:len(self.samples) - len(self.samples) % 3]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, target = self.samples[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, target

# Set the data directory
data_dir = 'DataSet/FaceData/processed'

# Define transformation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create a custom dataset
custom_dataset = CustomFaceDataset(root_dir=data_dir, transform=transform)

# Create data loader
batch_size = 9  # Set the batch size to a multiple of 3
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Create the FaceNet model
model = FaceNetModel()
# Move the model to the desired device (e.g., CUDA if available)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# Define triplet margin loss
class TripletMarginLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.norm(anchor - positive, p=2, dim=1)
        distance_negative = torch.norm(anchor - negative, p=2, dim=1)
        loss = torch.relu(distance_positive - distance_negative + self.margin)
        return loss.mean()

criterion = TripletMarginLoss(margin=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        embeddings = model(inputs)
        embeddings = embeddings.view(embeddings.size(0), -1)

        # Split the embeddings into anchor, positive, and negative
        anchor, positive, negative = embeddings.chunk(3)

        loss = criterion(anchor, positive, negative)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 10 == 9:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(data_loader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

# Save the trained model
model_save_path = 'Models/facenet_model.pth'
torch.save(model.state_dict(), model_save_path)

print('Finished Training')

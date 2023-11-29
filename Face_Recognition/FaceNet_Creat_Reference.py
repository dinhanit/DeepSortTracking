import os
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torch.utils.data import DataLoader

# Define a custom dataset for the reference database
class ReferenceFaceDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for the reference face database.
    """
    def __init__(self, root_dir, transform=None):
        """
        Initializes the ReferenceFaceDataset.

        Args:
        - root_dir (str): Path to the root directory of the dataset.
        - transform (callable, optional): Optional transform to be applied to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            for filename in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, filename), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, target = self.samples[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, target

# Set the data directory for the reference database
reference_data_dir = 'DataSet/FaceData/processed'

# Create the FaceNet model
model = InceptionResnetV1(pretrained='vggface2')
# Move the model to the desired device (e.g., CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define transformation for the reference dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create a custom dataset for the reference database
reference_dataset = ReferenceFaceDataset(root_dir=reference_data_dir, transform=transform)

# Create a data loader for the reference dataset
reference_loader = DataLoader(reference_dataset, batch_size=1, shuffle=False)

# Create empty lists to store reference embeddings and labels
reference_embeddings = []
reference_labels = []

# Extract and store embeddings and labels for the reference database
with torch.no_grad():
    for inputs, labels in reference_loader:
        embeddings = model(inputs)
        reference_embeddings.append(embeddings)
        reference_labels.append(labels.item())

# Save the reference embeddings and labels for future recognition
torch.save(reference_embeddings, 'Models/reference_embeddings.pth')
torch.save(reference_labels, 'Models/reference_labels.pth')

print('Reference database created and saved.')

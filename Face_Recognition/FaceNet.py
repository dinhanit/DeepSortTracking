import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
# Define the FaceNet model
class FaceNetModel(nn.Module):
    def __init__(self):
        super(FaceNetModel, self).__init__()
        self.facenet = InceptionResnetV1(pretrained='vggface2')

    def forward(self, x):
        embeddings = self.facenet(x)
        return embeddings
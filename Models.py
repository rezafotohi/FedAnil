import torch
import torch.nn as nn
import torch.nn.functional as F
# FedAnil: K-Medoids
from sklearn_extra.cluster import KMedoids
# FedAnil: Silhouette Index
from sklearn.metrics import silhouette_score
import numpy as np

# Define ResNet50 model
#resnet50 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)

# Define the ResNet50 architecture using nn.Sequential
resnet50 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2), 
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    nn.Linear(7*7*64, 512),
    nn.Linear(512, 10),)

# Define GloVe model
glove = nn.Sequential(
    nn.Linear(100, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
)

# Define CNN model
cnn = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    nn.Flatten(),
    nn.Linear(3136, 512),
    nn.Linear(512, 10),
    nn.ReLU(),
)

# Define concatenated model
class ConcatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = resnet50
        self.glove = glove
        self.cnn = cnn
        self.fc3 = nn.Linear(1000 + 16 + 128, 256) # Concatenated output size is 1000+16+128 = 1144
        self.fc4 = nn.Linear(256, 10) # Output size is 10 for classification
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        resnet_outetput = tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        glove_output = tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7*7*64)
        cnn_output = tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        #concat = torch.cat((resnet_outetput, glove_output, cnn_output), dim=1)
        #x = F.relu(self.fc1(concat))
        return tensor
    
    # FedAnil: Sparsification
    def first_filter(self, global_parameters):
        selected_parameters = {}
        for var in self.state_dict():
            shape_of_original_gradients = self.state_dict()[var].shape 
            reshape_of_local_gradients = self.state_dict()[var].view(-1)
            reshape_of_global_gradients = global_parameters[var].view(-1)
            combine_gradients = reshape_of_global_gradients
            index = 0
            for item1, item2 in zip(reshape_of_local_gradients, reshape_of_global_gradients):
                if item1 > item2:
                    combine_gradients[index] = item1
                else:
                    combine_gradients[index] = 0
                index += 1
            selected_parameters[var] = combine_gradients.reshape(shape_of_original_gradients)
        return selected_parameters
    # FedAnil: K-Medoids
    def kmedoids_update(self, max_k = 10):
        # FedAnil: Silhouette Index
        max_silhouette_scores = 0
        best_k = 2
        best_kmedoids_data = {}
        for k in range(2, max_k + 1):
            kmedoids_clusters_and_labels = dict()
            # FedAnil: Silhouette Index
            sum_silhouette_scores = 0
            vars_count = 0
            for var in self.state_dict():
                shape_of_datas = self.state_dict()[var].shape
                datas = self.state_dict()[var].reshape(shape_of_datas[0], -1)
                k = min(k, shape_of_datas[0] - 1)
                datakm = KMedoids(n_clusters=k, random_state=0).fit(datas)
                kmedoids_clusters_and_labels[var] = datakm
                if (np.unique(datakm.labels_).size > 1):
                    sum_silhouette_scores += silhouette_score(datas, datakm.labels_)
                vars_count += 1
            avg_silhouette_score = sum_silhouette_scores / vars_count
            if avg_silhouette_score > max_silhouette_scores:
                max_silhouette_scores = avg_silhouette_score
                best_k = k
                best_kmedoids_data = kmedoids_clusters_and_labels
        return best_kmedoids_data

class CombinedModel(nn.Module):
    def __init__(self, glove_model = glove, resnet_model = resnet50, cnn_model = cnn):
        super().__init__()
        self.glove_model = glove_model
        self.resnet_model = resnet_model
        self.cnn_model = cnn_model

    def forward(self, x, model_choice = "cnn"):
        #print(f"X input size {x.size()}")
        x = x.view(-1, 1, 28, 28)
        #print(f"X view input size {x.size()}")
        if model_choice == "glove":
            x = self.glove_model(x)
        elif model_choice == "resnet":
            x = self.resnet_model(x)
        elif model_choice == "cnn":
            for layer in self.cnn_model:
                x = layer(x)
                #print(f"layers {x.size()}")
        else:
            raise ValueError("Invalid model choice.")
        return x
    # FedAnil: Sparsification
    def first_filter(self, global_parameters):
        selected_parameters = {}
        for var in self.state_dict():
            shape_of_original_gradients = self.state_dict()[var].shape 
            reshape_of_local_gradients = self.state_dict()[var].view(-1)
            reshape_of_global_gradients = global_parameters[var].view(-1)
            combine_gradients = reshape_of_global_gradients
            index = 0
            for item1, item2 in zip(reshape_of_local_gradients, reshape_of_global_gradients):
                if item1 > item2:
                    combine_gradients[index] = item1
                else:
                    combine_gradients[index] = 0
                index += 1
            selected_parameters[var] = combine_gradients.reshape(shape_of_original_gradients)
        return selected_parameters
    # FedAnil: K-Medoids
    def kmedoids_update(self, max_k = 10):
        # FedAnil: Silhouette Index
        max_silhouette_scores = 0
        best_k = 2
        best_kmedoids_data = {}
        for k in range(2, max_k + 1):
            kmedoids_clusters_and_labels = dict()
            # FedAnil: Silhouette Index
            sum_silhouette_scores = 0
            vars_count = 0
            for var in self.state_dict():
                shape_of_datas = self.state_dict()[var].shape
                datas = self.state_dict()[var].reshape(shape_of_datas[0], -1)
                k = min(k, shape_of_datas[0] - 1)
                datakm = KMedoids(n_clusters=k, random_state=0).fit(datas)
                kmedoids_clusters_and_labels[var] = datakm
                if (np.unique(datakm.labels_).size > 1):
                    sum_silhouette_scores += silhouette_score(datas, datakm.labels_)
                vars_count += 1
            avg_silhouette_score = sum_silhouette_scores / vars_count
            if avg_silhouette_score > max_silhouette_scores:
                max_silhouette_scores = avg_silhouette_score
                best_k = k
                best_kmedoids_data = kmedoids_clusters_and_labels
        return best_kmedoids_data

class Generator(nn.Module):
    def __init__(self, model='cnn'):
        super().__init__()
        mm = None
        if model == 'resnet':
            self.fc = nn.Linear(10, 512)
            self.fc2 = nn.Linear(512, 7764)
            mm = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.Sigmoid()
            )
        elif model == 'glove':
            mm = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Linear(128, 784),
            )
        elif model == 'cnn':
            mm = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 784),
            )
        self.model = mm
    
    def forward(self, x, model_type = "cnn"):
        output = None
        if model_type == "resnet":
            x = self.fc(x)
            x = self.fc2(x)
            x = x.view(-1, 64, 7, 7)  # Reshape into feature maps
            output = self.model(x)
        else:
            output = self.model(x)
        return output

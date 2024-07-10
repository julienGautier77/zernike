import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
from torchinfo import summary
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset



x_train = np.load('x_train2.npy')
y_train = np.load('y_train2.npy')
x_train = np.around(x_train,5)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
y_train = torch.reshape(y_train,(10000,7))
# x_train = x_train[:100]
# y_train = y_train[:100]
print('x_train',x_train.shape, x_train.min(), x_train.max())
print('y_train',y_train.shape, y_train.min(), y_train.max())

device = torch.device('cuda')

class ResNet50Custom(nn.Module):
    def __init__(self):
        super(ResNet50Custom, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        #Adapter la première couche pour accepter 1 canal au lieu de 3
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remplacer la dernière couche fully connected
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 7) # nombre de para de sortie
    
    def forward(self, x):
        x = self.resnet50(x)
        #x = torch.round(x * 1000) / 1000# Arrondir les sorties à deux décimales
        return x
    
# Transformation pour adapter la taille d'entrée attendue par ResNet50
transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

# Appliquer la transformation sur les entrées
x_train = torch.stack([transform(input) for input in x_train]).to(device)

model = ResNet50Custom().to(device)
criterion = nn.MSELoss()#nn.CrossEntropyLoss() #
optimizer = optim.Adam(model.parameters(), lr=0.01)#optim.SGD(model.parameters(), lr=0.01)#




def compute_accuracy(outputs, targets):
    """Fonction pour calculer l'accuracy"""
    with torch.no_grad():
        # Convertir les prédictions en classes prédites
        _, predicted = torch.max(outputs, 1)
        # Calculer le nombre de prédictions correctes
        correct = (predicted == targets).sum().item()
        # Calculer l'accuracy
        accuracy = correct / targets.size(0)
    return accuracy


# optimisation stochastique , mini-batch
epochs = 2# nombre d'epoch
losses = []
batch_size = 64
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
losses = []
accuracies = []
with tqdm(range(epochs), unit='epoch') as tepoch :
    for epoch in tepoch:
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        
        for batch_inputs, batch_targets in dataloader:
            # Déplacer les données sur l'appareil
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            #print(batch_inputs.shape,batch_targets.shape)
            # Forward pass
            outputs = model(batch_inputs)
            #print('outputs',outputs.shape)
            loss = criterion(outputs, batch_targets )
            losses.append(loss.item())
        
            # Backward pass and optimization
            optimizer.zero_grad () # clean up step
            loss.backward()
            optimizer.step()
            batch_accuracy = compute_accuracy(outputs, batch_targets.argmax(dim=1))
            epoch_accuracy += batch_accuracy
        
        # Calculer l'accuracy moyenne par époque
        epoch_loss /= len(dataloader)
        epoch_accuracy /= len(dataloader)
    
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        tepoch.set_postfix(loss=loss.item())


# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.title('Perte d\'entraînement au fil des époques')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Accuracy')
plt.xlabel('Époque')
plt.ylabel('Accuracy')
plt.title('Accuracy d\'entraînement au fil des époques')
plt.legend()

plt.tight_layout()
plt.show()


# test 
with torch.no_grad():
    # Extraire la deuxième image de inputs
    test_input = x_train[4].unsqueeze(0)  # Ajouter une dimension batch
    predicted_output = model(test_input)
print('predicted value : ', predicted_output)
print('real value : ',y_train[4].unsqueeze(0))
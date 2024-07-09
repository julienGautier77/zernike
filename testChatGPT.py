import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models, transforms

# 1. Télécharger et adapter le modèle ResNet50
class ResNet50Custom(nn.Module):
    def __init__(self, output_size):
        super(ResNet50Custom, self).__init__()
        # Utiliser 'weights' au lieu de 'pretrained'
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Adapter la première couche pour accepter 1 canal au lieu de 3
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remplacer la dernière couche fully connected
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, output_size)
    
    def forward(self, x):
        return self.resnet50(x)

# Paramètres
output_size = 3

# Initialiser le modèle
model = ResNet50Custom(output_size)

# 2. Définir les données d'entraînement (exemple fictif)
# Générer des données d'entrée : 10 exemples, chacun étant une image de taille 1x30x30 (1 canal) avec valeurs dans [-0.2, 0.2]
inputs = torch.tensor(np.random.uniform(-0.2, 0.2, (10, 1, 30, 30)), dtype=torch.float32)
# Générer des données de sortie : 10 exemples, chacun étant un vecteur de taille `output_size` avec valeurs dans [-0.2, 0.2]
targets = torch.tensor(np.random.uniform(-0.2, 0.2, (10, output_size)), dtype=torch.float32)

# Transformation pour adapter la taille d'entrée attendue par ResNet50
transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

# Appliquer la transformation sur les entrées
inputs = torch.stack([transform(input) for input in inputs])

# 3. Définir la fonction de perte et l'optimiseur
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Boucle d'entraînement avec enregistrement des pertes
num_epochs = 1000
losses = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    losses.append(loss.item())
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. Tracer le graphique de la perte d'entraînement
plt.plot(losses)
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.title('Perte d\'entraînement au fil des époques')
plt.show()

# Exemple d'inférence
with torch.no_grad():
    test_input = torch.tensor(np.random.uniform(-0.2, 0.2, (1, 1, 30, 30)), dtype=torch.float32)
    test_input = transform(test_input)
    predicted_output = model(test_input.unsqueeze(0))
    print(f'Test Input: {test_input}')
    print(f'Predicted Output: {predicted_output}')
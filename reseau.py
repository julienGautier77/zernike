import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
from torchinfo import summary
from tqdm import tqdm
from torchvision import models, transforms


x_train = np.load('x_train2.npy')
y_train = np.load('y_train2.npy')
x_train = np.around(x_train,5)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
y_train = torch.reshape(y_train,(10000,7))
print('x_train',x_train.shape, x_train.min(), x_train.max())
print('y_train',y_train.shape, y_train.min(), y_train.max())



model_linear = nn.Sequential(
    nn.Flatten(),
    nn.Linear(30*30,7),
)


hidden_layer=512

model_ffn= nn.Sequential(
    nn.Flatten(),
    nn.Linear(40*40,hidden_layer),
    nn.Dropout(p=0.1),
    nn.ReLU(),
    nn.Linear(hidden_layer,7)
        
)



class ResNet50Custom(nn.Module):
    def __init__(self):
        super(ResNet50Custom, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        #Adapter la première couche pour accepter 1 canal au lieu de 3
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remplacer la dernière couche fully connected
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 7) # nombre de para de sortie
    
    def forward(self, x):
        return self.resnet50(x)
    
# Transformation pour adapter la taille d'entrée attendue par ResNet50
transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

# Appliquer la transformation sur les entrées
x_train = torch.stack([transform(input) for input in x_train])



#summary(model_ffn,input_size=(512,1,30,30))
model = ResNet50Custom()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)#Adam(model.parameters(), lr=0.005)#

epochs = 10

n_train  = len(y_train)
loss_train = []
accuracy_train = []
with tqdm(range(epochs), unit='epoch') as tepoch :
    for epoch in tepoch:
        optimizer.zero_grad () # clean up step
        scores = model(x_train) # on calcul le score du modele 
        
        loss = criterion(scores, y_train) # calul de la perte (score par ropport a la solution)
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
        with torch.no_grad():
            scores = model_ffn(x_train)
            correct = (scores==y_train).sum().item()
            accuracy_train.append(correct/n_train)
            tepoch.set_postfix(loss=loss.item(),accuracy=accuracy_train[-1])
print(scores[2,:])
fig, (ax1,ax2)=plt.subplots(ncols=2)
ax1.plot(np.arange(epochs),loss_train)
ax2.plot(np.arange(epochs),accuracy_train)
plt.title('Perte d\'entraînement au fil des époques')
plt.show()


I=x_train[2,0,:,:]
print(I.shape)
I=I.cpu().data.numpy()
labels = model_ffn(x_train[2,0,:,:])
print('labels',labels)
fig2=plt.figure()
ax3= fig2.add_subplot(121)

ax3.imshow(I,cmap='jet')
plt.show()

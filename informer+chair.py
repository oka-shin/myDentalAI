import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys

np.set_printoptions(threshold=np.inf, suppress=True)

args = sys.argv
C=int(args[1])

# Informer model implementation
class Informer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout=0.1):
        super(Informer, self).__init__()
        self.embedding_dim = 32  # Change this to be divisible by num_heads
        self.input_projection = nn.Linear(input_dim, self.embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(self.embedding_dim, output_dim)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x

# Parameters
input_dim = 31
hidden_dim = 64
output_dim = 4
num_layers = 3
num_heads = 4
dropout = 0.1
learning_rate = 0.001
num_epochs = C
batch_size = 32

input = np.load("./data/cat4all+chair.npy")
input2 = np.sum(input, axis=1)
output8 = np.load("./data/cat4allans.npy")
result = np.zeros((input2.shape[0], output8.shape[1]))

for i in tqdm(range(input2.shape[0])):
    x = np.delete(input2, i, 0)
    y = np.delete(output8, i, 0)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=0)
    for j in range(output8.shape[1]):
        Yj_train = Y_train[:, j]
        Yj_test = Y_test[:, j]

        # Convert data to tensors
        Xj_train = torch.from_numpy(X_train).float()
        Xj_test = torch.from_numpy(X_test).float()
        Yj_train = torch.from_numpy(Yj_train).long()
        Yj_test = torch.from_numpy(Yj_test).long()
        train_dataset = TensorDataset(Xj_train, Yj_train)
        test_dataset = TensorDataset(Xj_test, Yj_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Model, loss, and optimizer
        model = Informer(input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)  # Change shape back to (batch_size, output_dim)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            X_input = torch.from_numpy(input2[i]).float().unsqueeze(0)  # Add batch and sequence length dimensions and permute
            yosoku = model(X_input)

        result[i][j] = yosoku.argmax(dim=1).item()  # Convert the output to the predicted class

np.savetxt('informer+chair-results.txt', result, fmt='%d')

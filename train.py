import json
from utils import tokenize
from utils import stem
from utils import bagOfWords
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('intents.json', 'r') as file:
    intents = json.load(file)

all_words = []
tags = []
xy = []  # both patterns and intents

# tokenize
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

punctuation = ["?", "!", ".", ","]
all_words = [stem(w) for w in all_words if w not in punctuation]
all_words = sorted(set(all_words))

tags = sorted(set(tags)) #to get rid of any duplicate values and sort the rest

x_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bagOfWords(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)  

x_train = np.array(x_train)
y_train = np.array(y_train)


class C():
    def __repr__(self):
        return "done"


class chatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


batch_size = 8 
hidden_layer_size = 8
output_size = len(tags)  # equal to number of classes/tags we have
input_size = len(x_train[0])  # length of each bag of words we created
learning_rate = 0.001
num_epochs = 1000 # number of iterations in which the whole data will be processed

dataset = chatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_layer_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        #forward propogation
        outputs = model(words)
        loss = criterion(outputs, labels)

        #backward propogation and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_layer_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"

torch.save(data, FILE)

print(f'Training complete and file saved to {FILE}')

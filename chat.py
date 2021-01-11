import random
import json
import torch
from model import NeuralNet
from utils import bagOfWords, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as file:
    intents = json.load(file)

FILENAME = "data.pth"
data = torch.load(FILENAME)
input_size = data["input_size"]
hidden_layer_size = data["hidden_size"]
output_size = data["output_size"]

all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_layer_size, output_size).to(device)
model.load_state_dict(model_state)

model.eval()

bot_name = "Sumit"

print("Welcome to the chatbot. Enter 'quit' to exit.")

while (True):
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    input_to_model = bagOfWords(sentence, all_words)
    input_to_model = input_to_model.reshape(1, input_to_model.shape[0])
    input_to_model = torch.from_numpy(input_to_model).to(device)

    output = model(input_to_model)  # gives the predictions
    _, predictions = torch.max(output, dim=1)
    tag = tags[predictions.item()]

    probabilities = torch.softmax(output, dim=1) 
    probability = probabilities[0][predictions.item()]

    if probability.item() > 0.75:
        # loop over all the intents and check if the chat matches that
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand")

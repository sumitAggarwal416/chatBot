#chatbot

This is a chatBot that uses Natural Language Processing (NLP), Bag of Words Pipeline and Artificial Neural Networks reply with appropriate responses to your input.

The punctuation removal step was pretty straight forward - removing all punctuation characters from the strings.

Tokenization was done to split the string by non-alphanumeric characters. For example, it converts the word "what's" to "what s". Whereas the punctuation step converts the same string to "whats".

For Stemming I used the LancasterStemmer from NLTK library. This transformed words into their stems like - "thinking" became "think".

In this model, I trained a neural network having 3 linear layers. I use ReLU function as the activation function in the hidden layers and Softmax function as the activation function of the final output layer.

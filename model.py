import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]   # Last layer truncated with -1
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embedding_size = embed_size
        self.hidden_feature_size = hidden_size
        self.vocab_size = vocab_size
        self.number_layers = num_layers

        # embedding layer
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_size)

        # RNN model
        self.rnn = nn.GRU(self.embedding_size, self.hidden_feature_size, self.number_layers, batch_first=True)

        # FC layer
        self.fc = nn.Linear(self.hidden_feature_size, self.vocab_size)
    
    def forward(self, features, captions):
        # discard the end token
        captions = captions[:, :-1]
        caption_embeddings = self.word_embeddings(captions)

        model_input = torch.cat((features.unsqueeze(1), caption_embeddings), dim=1) # REMEMBER to check the shape!
        model_output, _ = self.rnn(model_input)
        vocabulary_probabilities = self.fc(model_output)
        return vocabulary_probabilities

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        image_caption = []
        
        for i in range(max_len):
            rnn_output, states = self.rnn(inputs, states)
            vocab_probabilities = self.fc(rnn_output)

            _, word_index = vocab_probabilities.max(dim=2)
            inputs = self.word_embeddings(word_index)
            image_caption.append(word_index.item())

        return image_caption
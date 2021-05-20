#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torchvision
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderCNN_SEN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN_SEN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.AvgPooling = nn.AdaptiveAvgPool2d(1,1)
        self.Sen_method = self.Sen_method1()
#        self.Sen_method = self.Sen_method2()
        self.senF1 = nn.Conv2d(in_channels=2049,out_channels=2048,kernel_size=1,stride=1)
        self.senF2 = nn.Conv2d(in_channels=2048,out_channels=2048,kernel_size=1,stride=1)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def Sen_method1(self, encoder_output):
        # method1: add a new channel to indicate the sentiment;no skip connection
        ## idea one
        s = torch.ones(encoder_output.size(0), 1, encoder_output.size(2), encoder_output.size(3)) if sentiment == 'positive' else -torch.ones_like(encoder_output)  # b * 1 * height * width

        ## idea two
        s = torch.eye(encoder_output.size(2)) # height * width
        s = s.unsqueeze(0) # 1 * height * width
        s = torch.cat([s.unsqueeze(0)]*encoder_output.size(0), dim=0) # b * 1 * height * width
        if sentiment != 'positive':
            s *= -1

        # sentiment injection
        new_encoder_output = torch.cat([s, encoder_output], dim=1) # b * (1 + channel) * height * width <---> b * (1 + 2048) * height * width
        # trainable part
        decoder_input = self.senF(new_encoder_output) # b * (channel + 1) * height * width --> b * channel * height * width
        return decoder_input

    def Sen_method2(self, encoder_output):
        # method2: add noise
        if sentiment == 'positive':
            noise = torch.normal(mean=1,std=1e-2,size=(2048, 1, 1)) # channel * height * width
        else:
            noise = torch.normal(mean=-1,std=1e-2,size=(2048, 1, 1))
        # tranable part
        decoder_input = encoder_output + self.senF2(noise)
        return decoder_input

    def forward(self, images):
        """Extract feature vectors from input images."""
        features = self.resnet(images)  # 64 * 2048  ---> batch_size * 2048 * 1 * 1
        features = self.Sen_method1(features) # b * 2048 * 1 * 1
        features = features.reshape(features.size(0), -1)  # batch_size * 2048
        features = self.bn(self.linear(features))  # batch_size * 512
        return features


class DecoderRNN_SEN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout=0.5):  # 512 512 9490 1
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN_SEN, self).__init__()
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.decode_step = nn.LSTMCell(embed_size, hidden_size, bias=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, images, captions, length):
        batch_size = images.size(0)  # 64
        vocab_size = self.vocab_size  # vocab size

        caption_lengths, sort_ind = length.squeeze(1).sort(dim=0, descending=True)
        features = images[sort_ind]  # 2048
        captions = captions[sort_ind]  # 512

        embeddings = self.embedding(captions)
        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        h, c = self.decode_step(features)  # (batch_size_t, decoder_dim)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            h, c = self.decode_step(embeddings[:batch_size_t, t, :],
                                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds

        return predictions, captions, decode_lengths, sort_ind


'''
def Sen_method1(encoder_output):
    # method1: add a new channel to indicate the sentiment;no skip connection
    ## idea one
    s = torch.ones(encoder_output.size(0), 1, encoder_output.size(2), encoder_output.size(3)) if sentiment == 'positive' else -torch.ones_like(encoder_output)  # b * 1 * height * width

    ## idea two
    s = torch.eye(encoder_output.size(2)) # height * width
    s = s.unsqueeze(0) # 1 * height * width
    s = torch.cat([s.unsqueeze(0)]*encoder_output.size(0), dim=0) # b * 1 * height * width
    if sentiment != 'positive' :
        s *= -1
    new_encoder_output = torch.cat([s, encoder_output], dim=1) # b * (1 + channel) * height * width
    # trainable part
    decoder_input = f(new_encoder_output) # b * channel * height 8 width
    return decoder_input
'''

'''
def Sen_method2(encoder_output):
    # method2: add noise
    if sentiment == 'positive':
        noise = torch.normal(mean=1,std=1e-2,size=()) # channel * height * width
    else:
        noise = torch.normal(mean=-1,std=1e-2,size=())
    # tranable part
    decoder_input = encoder_output + f(noise)
    return decoder_input
'''

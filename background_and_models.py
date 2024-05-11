# Load relevant packages
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import tabulate
from sklearn.metrics import f1_score
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel, logging
import re
from matplotlib import pyplot as plt
import json
from negate import Negator
from loss import UnlikelihoodLoss
import os

pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None
# from negate import Negator

class Datum():
    def __init__(self):
        self.sent1 = None
        self.sent2 = None
        self.sent1_POS = None
        self.sent2_POS = None
        self.gold_label = None
        self.genre = None
    
    def get_sent1(self):
        return self.sent1
    def get_sent2(self):
        return self.sent2
    def get_sent1_POS(self):
        return self.sent1_POS
    def get_sent2_POS(self):
        return self.sent2_POS
    def get_gold_label(self):
        return self.gold_label
    def get_genre(self):
        return self.genre
    def set_sent1(self, sent1):
        self.sent1 = sent1
    def set_sent2(self, sent2):
        self.sent2 = sent2
    def set_sent1_POS(self, sent1_POS):
        self.sent1_POS = sent1_POS
    def set_sent2_POS(self, sent2_POS):
        self.sent2_POS = sent2_POS
    def set_gold_label(self, gold_label):
        self.gold_label = gold_label
    def set_genre(self, genre):
        self.genre = genre
        
    # print key elements in datum object
    def __str__(self):
        return f"Sent1: {self.sent1}\nSent2: {self.sent2}\nLabel: {self.gold_label}\n"
    
    # write to dictionary to save datasets
    def to_dict(self):
        return {
            "sent1": self.sent1,
            "sent2": self.sent2,
            "gold_label": self.gold_label,
            "genre": self.genre
        }

    @classmethod
    def from_dict(cls, data):
        datum = cls()
        datum.set_sent1(data["sent1"])
        datum.set_sent2(data["sent2"])
        datum.set_gold_label(data["gold_label"])
        datum.set_genre(data["genre"])
        return datum

def get_samples(data, sample_num):
    
    random.shuffle(data)

    unique_genres = set()
    for datum in data:
        unique_genres.add(datum.get_genre())

    # for each genre, create sample_data to include sample_num examples from each label
    # divide by (len genres * 3) because we want to sample evenly from each genre to get to sample_num samples
    samples_per_label = round(sample_num / (len(unique_genres) * 3))

    sample_data = []
    for genre in unique_genres:
        genre_data = [datum for datum in data if datum.get_genre() == genre]
        genre_entailments = [datum for datum in genre_data if datum.get_gold_label() == "entailment"]
        genre_neutrals = [datum for datum in genre_data if datum.get_gold_label() == "neutral"]
        genre_contradictions = [datum for datum in genre_data if datum.get_gold_label() == "contradiction"]
        random.shuffle(genre_entailments)
        random.shuffle(genre_neutrals)
        random.shuffle(genre_contradictions)
        sample_data += genre_entailments[:samples_per_label] + genre_neutrals[:samples_per_label] + genre_contradictions[:samples_per_label]

    print(f"Sample data: {len(sample_data)} data points, with {samples_per_label} points per label and {len(unique_genres)} genres")
    return sample_data


# helper function to load data from jsonl files
def load_data(path, sample_num=0):
    data = []
    filtered = []

    with open(path, "r") as file:
        for line in file:
            line = json.loads(line)
            d = Datum()
            d.set_sent1(line["sentence1"])
            d.set_sent2(line["sentence2"])
            d.set_gold_label(line["gold_label"])
            d.set_genre(line["genre"])
            # remove examples of label "-", where annotators couldn't agree on a label
            if line["gold_label"] == "-":
                filtered.append(d)
            else:
                data.append(d)
    
    # for each genre, create sample_data to include sample_num examples from each label
    # divide by (len genres * 3) because we want to sample evenly from each genre to get to sample_num samples
    if sample_num > 0:
        return get_samples(data, sample_num)
        
    print(f"Loaded {len(data)} data points from {path}")
    print(f"Filtered {len(filtered)} instances of '-' from {path}\n")
    return data

negator = Negator(use_transformers=True, use_gpu=True, fail_on_unsupported=True)

def create_negated(data, sample_num=0):
    new_contradictions_list = []
    new_entailments_list = []
    new_neutrals_list = []
    
    print(f"Creating negated samples for {len(data)} data points...")
    for datum in tqdm(data):
        try:
            if datum.get_gold_label() == "entailment":

                # take random number (1 or 2) to determine which negation to apply
                rand = random.randint(1, 2)
                if rand == 1:
                    # by definition of contradiction, A contradicts notB
                    negated_datum = Datum()
                    negated_datum.set_sent1(datum.get_sent1())
                    negated_datum.set_sent2(negator.negate_sentence(datum.get_sent2()))
                    negated_datum.set_gold_label("contradiction")
                    negated_datum.set_genre(datum.get_genre())
                    new_contradictions_list.append(negated_datum)
                if rand == 2:    
                    # by modus tollens, notB contradicts A
                    negated_datum = Datum()
                    negated_datum.set_sent2(datum.get_sent1())
                    negated_datum.set_sent1(negator.negate_sentence(datum.get_sent2()))
                    negated_datum.set_gold_label("contradiction")
                    negated_datum.set_genre(datum.get_genre())
                    new_contradictions_list.append(negated_datum)

            elif datum.get_gold_label() == "neutral":
                # A is neutral to both B and notB
                negated_datum = Datum()
                negated_datum.set_sent1(datum.get_sent1())
                negated_datum.set_sent2(negator.negate_sentence(datum.get_sent2()))
                negated_datum.set_gold_label("neutral")
                negated_datum.set_genre(datum.get_genre())
                new_neutrals_list.append(negated_datum)
                
            elif datum.get_gold_label() == "contradiction":
                # take random number (1 to 3) to determine which negation to apply
                rand = random.randint(1, 3)
                
                if rand == 1:
                    # by definition of contradiction, A entails notB
                    negated_datum = Datum()
                    negated_datum.set_sent1(datum.get_sent1())
                    negated_datum.set_sent2(negator.negate_sentence(datum.get_sent2()))
                    negated_datum.set_gold_label("entailment")
                    negated_datum.set_genre(datum.get_genre())
                    new_entailments_list.append(negated_datum)
                
                if rand == 2:
                    # B entails notA
                    negated_datum = Datum()
                    negated_datum.set_sent1(datum.get_sent2())
                    negated_datum.set_sent2(negator.negate_sentence(datum.get_sent1()))
                    negated_datum.set_gold_label("entailment")
                    negated_datum.set_genre(datum.get_genre())
                    new_entailments_list.append(negated_datum)

                if rand == 3:
                    # B contradicts A
                    negated_datum = Datum()
                    negated_datum.set_sent1(datum.get_sent2())
                    negated_datum.set_sent2(datum.get_sent1())
                    negated_datum.set_gold_label("contradiction")
                    negated_datum.set_genre(datum.get_genre())
                    new_contradictions_list.append(negated_datum)
            
        except RuntimeError:
            pass  # skip unsupported sentence
        
    negated_data = []
        
    # get the minimum number of samples for each label
    balancer = min(len(new_contradictions_list), len(new_entailments_list), len(new_neutrals_list))

    negated_data = new_contradictions_list[:balancer] + new_entailments_list[:balancer] + new_neutrals_list[:balancer]

    random.shuffle(negated_data)

    if sample_num > 0 and len(negated_data) >= sample_num:
        get_samples(negated_data, sample_num)
            
    print(f"\nCreated {len(negated_data)} negated samples: with {balancer} of each label\n")
    return negated_data

# ### Examining the dataset
# - training set labels are balanced, with 130,900 examples of each type of label
# - removed 185 and 168 examples of "-" label from the matched and mismatched datasets
# - edited initial load_data function to make balanced samples for faster training based on labels/genres

labels = ['entailment', 'contradiction', 'neutral']
# %%
# class config for global variables
class Config:
    max_length = 512 # max length of input sequence
    learning_rate = 2e-5 # learning rate
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# BERT-specific training functions
labels = ['entailment', 'contradiction', 'neutral']
def train_epoch_BERT(train_data, classifier, optimizer, criterion, batch_size=32):
    total_loss = 0
    random.shuffle(train_data)
    loss = None
    # Loop over the train_data
    for datum_i, datum in enumerate(tqdm(train_data)):
        
        # getting sentences 1 and 2 from datum
        sent1 = datum.get_sent1()
        sent2 = datum.get_sent2()

        # target indexes one hot encoded
        target = labels.index(datum.get_gold_label())
        target_array = [0] * 3
        target_array[target] = 1
        target_tensor = torch.tensor([target_array], dtype=torch.float, device=Config.device)

        outputs = classifier(sent1, sent2)
        if criterion == "cross_entropy":
            _loss = nn.CrossEntropyLoss()(outputs.squeeze(), target_tensor.squeeze()) # outputs is of shape (1, 3) and target_tensor is of shape (1, 3)
        
        if criterion == "unlikelihood":
            _loss = UnlikelihoodLoss()(outputs.squeeze(), target_tensor.squeeze())

        if loss is None:
            loss = _loss
        else:
            loss += _loss

        # If the datum_i is divisible by 32 perform backprop and update step
        if datum_i % batch_size == 0:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            loss = None
    
    # return average loss -- normalized by length of training data
    return total_loss / len(train_data)

# Train BERT classifier
def train_classifier_BERT(train_data, classifier, optimizer, criterion, n_epochs, batch_size=32):

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch_BERT(train_data, classifier, optimizer, criterion, batch_size)
        print('Average loss for epoch %d: %.4f' % (epoch, loss))

# test BERT classifier
def test_classifier_BERT(test_data, classifier):
    all_outputs = []
    all_targets = []
    random.shuffle(test_data)
    with torch.no_grad():
        
        for datum in tqdm(test_data):
            
            sent1 = datum.get_sent1()
            sent2 = datum.get_sent2()
            classifier_outputs = classifier(sent1, sent2)
            classifier_outputs = classifier_outputs.cpu().numpy()
            outputs = np.argmax(classifier_outputs, axis=1)

            targets = [labels.index(datum.get_gold_label())]
            
            all_outputs.extend(outputs)
            all_targets.extend(targets)

    print(classification_report(all_targets, all_outputs, target_names=labels))
    
    # generate examples where the model predicts the wrong label
    wrong_predictions = [(test_data[i], labels[all_outputs[i]], labels[all_targets[i]]) for i in range(len(all_outputs)) if all_outputs[i] != all_targets[i]]
    print(f"Wrong predictions: {len(wrong_predictions)}")
    # get the first 3 wrong predictions
    for i in range(3):
        tuple = wrong_predictions[i]
        print(f"Sent1: {tuple[0].get_sent1()}\nSent2: {tuple[0].get_sent2()}\nPredicted: {tuple[1]}\nTrue: {tuple[2]}\n")

    # adding this to add to list
    return f1_score(all_targets, all_outputs, average='macro')


def initialize_DB_models(lr=Config.learning_rate):
    distilbert_model = AutoModel.from_pretrained("distilbert-base-uncased")
    distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    classifier_base = DistilBERTClassifier(distilbert_model, distilbert_tokenizer).to(Config.device)
    classifier_aug = DistilBERTClassifier(distilbert_model, distilbert_tokenizer).to(Config.device)
    optimizer_base = optim.Adam(classifier_base.parameters(), lr)
    optimizer_aug = optim.Adam(classifier_aug.parameters(), lr)
    return classifier_base, classifier_aug, optimizer_base, optimizer_aug

# def initialize_B_models():
#     bert_model = BertModel.from_pretrained('bert-base-uncased')
#     bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
#     classifier_base = BERTClassifier(bert_model, bert_tokenizer).to(Config.device)
#     classifier_aug = BERTClassifier(bert_model, bert_tokenizer).to(Config.device)
#     optimizer_base = optim.Adam(classifier_base.parameters(), Config.learning_rate)
#     optimizer_aug = optim.Adam(classifier_aug.parameters(), Config.learning_rate)
#     return classifier_base, classifier_aug, optimizer_base, optimizer_aug

# class FeedForwardNetworkClassifier(nn.Module):
#     def __init__(self, input_size, embedding_size, hidden_size):
#         super(FeedForwardNetworkClassifier, self).__init__()
        
#         self.embedding = nn.Embedding(input_size, embedding_size)
#         # taking two sentences with max padding of 384 sentence size
#         self.linear_1 = nn.Linear(embedding_size * 2 * Config.max_length, hidden_size)
#         self.relu = nn.ReLU()
#         self.linear_2 = nn.Linear(hidden_size, 3)

#     # take both sentences as arguments, embed them
#     # concatenate the embeddings and flatten
#     # linear_1 layer -> relu function -> linear_2
#     # Return the output of the linear_2 layer
    
#     def forward(self, sent1, sent2):
#         sent1 = self.embedding(sent1)
#         sent2 = self.embedding(sent2)
#         input = torch.cat((sent1, sent2), dim=1)
#         input = torch.flatten(input, start_dim=1)
#         linear_1_output = self.linear_1(input)
#         relu_output = self.relu(linear_1_output)
#         linear_2_output = self.linear_2(relu_output)
#         return linear_2_output


# --------------------------------------------------------------------------------
# MODELS

embedding_size = 300
batch_size = 32
num_epochs = 3

logging.set_verbosity_error()

# BERT Classifier -- using CLS token embeddings when specified
class BERTClassifier(nn.Module):
    def __init__(self, llm, tokenizer, use_pooling=True, frozen_layer=11):
        super(BERTClassifier, self).__init__()
        
        self.llm = llm.to(Config.device)
        self.tokenizer = tokenizer
        self.use_pooling = use_pooling

        self.linear_1 = nn.Linear(768, int(768 / 3)).to(Config.device)
        self.relu = nn.ReLU().to(Config.device)
        self.linear_2 = nn.Linear(int(768 / 3), 3).to(Config.device)

        modules = [self.llm.embeddings, *self.llm.encoder.layer[:frozen_layer]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

        for param in self.llm.encoder.layer[frozen_layer].parameters():
            param.requires_grad = True

    def forward(self, sent1, sent2):
        encoded_sentences = self.tokenizer(
            [sent1, sent2],  # Pass both sentences at once
            padding='max_length',
            max_length=Config.max_length,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True
        )
        encoded_sentences = {k: v.to(Config.device) for k, v in encoded_sentences.items()} #of shape (batch_size, max_length)
        output = self.llm(**encoded_sentences)

        cls_embedding1 = output.last_hidden_state[0, 0, :]  # CLS token of the first sentence
        cls_embedding2 = output.last_hidden_state[1, 0, :]  # CLS token of the second sentence


        pooler_output = output.pooler_output # This is the output of the pooler layer, shape (2,768)


        if self.use_pooling:
            # turn into (1,768) by averaging the two sentences along the batch dimension
            avg_pooler_output = torch.mean(pooler_output, dim=0).unsqueeze(0) # shape (1, 768)
            linear_1_output = self.linear_1(avg_pooler_output) #avg of pooled sentences (pooled pool)
        else:
            # concatenate and average the two whole sentence embeddings
            cls_output = torch.mean(torch.stack([cls_embedding1.unsqueeze(0), cls_embedding2.unsqueeze(0)], dim=0), dim=0) # shape (1, 768)
            linear_1_output = self.linear_1(cls_output)

        relu_output = self.relu(linear_1_output)
        output = self.linear_2(relu_output)
        return output
    
class DistilBERTClassifier(nn.Module):
    def __init__(self, llm, tokenizer):
        super(DistilBERTClassifier, self).__init__()
        
        self.llm = llm.to(Config.device)
        self.tokenizer = tokenizer

        self.linear_1 = nn.Linear(768, int(768 / 3)).to(Config.device)
        self.relu = nn.ReLU().to(Config.device)
        self.linear_2 = nn.Linear(int(768 / 3), 3).to(Config.device)


        frozen_layer = 5 # freeze the first 5 layers of the BERT model (DistilBERT has 6 layers)
        modules = [self.llm.embeddings, *self.llm.transformer.layer[:frozen_layer]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

        for param in self.llm.transformer.layer[frozen_layer].parameters():
            param.requires_grad = True

    def forward(self, sent1, sent2):
        encoded_sentences = self.tokenizer(
            [sent1, sent2],  # Pass both sentences at once
            padding='max_length',
            max_length=Config.max_length,
            truncation=True,
            return_tensors='pt',
        )
        encoded_sentences = {k: v.to(Config.device) for k, v in encoded_sentences.items()} #of shape (batch_size, max_length)
        
        output = self.llm(**encoded_sentences)

        # output.last_hidden_state is of shape (batch_size, max_length, hidden_size) = (2, 512, 768)

        cls_embedding1 = output.last_hidden_state[0, 0, :]  # CLS token of the first sentence
        cls_embedding2 = output.last_hidden_state[1, 0, :]  # CLS token of the second sentence

        cls_output = torch.mean(torch.stack([cls_embedding1.unsqueeze(0), cls_embedding2.unsqueeze(0)], dim=0), dim=0) # shape (1, 768)

        # pooled_output = output.pooler_output
        
        linear_1_output = self.linear_1(cls_output)
        relu_output = self.relu(linear_1_output)
        output = self.linear_2(relu_output)
        return output
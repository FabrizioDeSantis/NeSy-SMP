import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration class for LSTM model parameters"""
    hidden_size: int
    num_layers: int
    dropout_rate: float = 0.1
    sequence_length: int = 26
    learning_rate: float = 0.001
    num_features: int = 4

class LSTMModel(nn.Module):
    def __init__(self, vocab_sizes: List[int], config: ModelConfig, num_classes: int, feature_names: List[str]):
        super(LSTMModel, self).__init__()
        self.config = config
        self.feature_names = feature_names
        self.numerical_features = [name for name in feature_names if name not in vocab_sizes.keys()]
        self.num_classes = num_classes
        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(vocab_size + 1, config.hidden_size, padding_idx=0)
            for feature, vocab_size in vocab_sizes.items()
        })
        self.linear_numerical = nn.Linear(len(self.numerical_features), config.hidden_size)
        #lstm_input_size = (config.hidden_size * len(self.embeddings)) + len(self.numerical_features)
        lstm_input_size = (config.hidden_size * len(self.embeddings)) + config.hidden_size
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout_rate,
            bidirectional=True
        )
        # self.fc = nn.Linear(config.hidden_size*2, self.num_classes)
        self.attention_weight = nn.Linear(config.hidden_size*2, config.hidden_size*2)
        self.attention_combine = nn.Linear(config.hidden_size*2, 1)
        self.dropout = nn.Dropout(0.3)

        self.lin_com = nn.Linear(23, config.hidden_size)
        self.fc_lstm = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.fc_comb = nn.Linear(config.hidden_size*3, config.hidden_size)
        self.fc_last = nn.Linear(config.hidden_size, self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = self.config.sequence_length
        embeddings_list = []
        numerical_features = []
        # Process categorical features
        for name in self.embeddings.keys():
            index = self.feature_names.index(name)
            index = index * seq_len
            end_idx = index + seq_len
            feature_data = x[:, index:end_idx].long()
            embeddings_list.append(self.embeddings[name](feature_data))
            
        # Process numerical features
        for name in self.numerical_features:
            index = self.feature_names.index(name)
            index = index * seq_len
            end_idx = index + seq_len
            feature_data = x[:, index:end_idx]
            numerical_features.append(feature_data)

        numerical_features = torch.stack(numerical_features, dim=2)
        numerical_features = self.linear_numerical(numerical_features)
        output = torch.cat(embeddings_list + [numerical_features], dim=2)
        # Concatenate all features
        return output

    def forward(self, x):
        x_com = x[:, -23:]  # Extract comorbidities
        x = x[:, :-23]  # Remove comorbidities from main input
        cat = self._get_embeddings(x)
        
        out, _ = self.lstm(cat)

        lengths = (x[:, :self.config.sequence_length] != 0).sum(1)  # Mask padding
        out = self.dropout(out)
        attention = self.attention_combine(torch.tanh(self.attention_weight(out)))
        attention_weights = F.softmax(attention, dim=1)
        output = torch.sum(out * attention_weights, dim=1)
       
        # output = out[torch.arange(out.size(0)), lengths - 1]

        # Process comorbidities
        x_com = self.lin_com(x_com)
        x_com = torch.relu(x_com)

        # Concatenate comorbidities
        output = torch.cat((output, x_com), dim=1)
        output = self.fc_comb(output)
        output = torch.relu(output)
        return self.sigmoid(self.fc_last(output))
    
class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(SimpleMLP, self).__init__()
        #self.fc1 = nn.Linear(243, hidden_size)  # 6h
        # self.fc1 = nn.Linear(289, hidden_size) # 12h
        self.fc1 = nn.Linear(input_size, hidden_size)  # 24h
        # self.fc1 = nn.Linear(275, hidden_size)  # 48h
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.elu(x)
        x = self.fc3(x)
        return self.sigmoid(x)
    
class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(SimpleMLP, self).__init__()
        #self.fc1 = nn.Linear(243, hidden_size)  # 6h
        # self.fc1 = nn.Linear(289, hidden_size) # 12h
        self.fc1 = nn.Linear(input_size, hidden_size)  # 24h
        # self.fc1 = nn.Linear(275, hidden_size)  # 48h
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.elu(x)
        x = self.fc3(x)
        return self.sigmoid(x)
    
class SimpleMLPAge(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(SimpleMLPAge, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # concat x1 and x2
        if x1.dim() == 1:
            x1 = x1.unsqueeze(1)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.elu(x)
        x = self.fc3(x)
        return self.sigmoid(x)

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(254, hidden_size)  # 6h
        # self.fc1 = nn.Linear(289, hidden_size) # 12h
        # self.fc1 = nn.Linear(261, hidden_size)  # 24h
        # self.fc1 = nn.Linear(275, hidden_size)  # 48h
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_1, x_2, x_3):
        # print(f"Input shapes: x_1: {x_1.shape}, x_2: {x_2.shape}, x_3: {x_3.shape}")
        if x_3.dim() == 1:
            x_3 = x_3.unsqueeze(1)
        x = torch.cat((x_1, x_2, x_3), dim=1)  # Concatenate inputs
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.elu(x)
        x = self.fc3(x)
        return self.sigmoid(x)
    
class LogitsToPredicateWSoftmax(torch.nn.Module):
    def __init__(self, logits_model):
        super(LogitsToPredicateWSoftmax, self).__init__()
        self.logits_model = logits_model
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x, l):
        logits = self.logits_model(x)
        probs = self.softmax(logits)
        out = torch.sum(probs * l, dim=1)
        return out
    
class LogitsToPredicateWSigmoid(torch.nn.Module):
    def __init__(self, logits_model):
        super(LogitsToPredicateWSigmoid, self).__init__()
        self.logits_model = logits_model
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        logits = self.logits_model(x)
        probs = self.sigmoid(logits)
        return probs
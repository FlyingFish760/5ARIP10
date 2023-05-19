import torch.nn as nn
import torch.nn.functional as F

# Option 1: Classifier, learn with both faulty and normal HD data
class Classifier(nn.Module):
  """
  Classifier network structure
  :Learn it with faulty and normal HD data
  WOrks 
  """
  def __init__(self,input_dim=100):
    super().__init__()

    self.input = input_dim

    self.fc1 = nn.Linear(input_dim, 0.5*input_dim)
    self.fc2 = nn.Linear(0.5*input_dim, 0.2*input_dim)
    self.fc3 = nn.Linear(0.2*input_dim, 2)
    self.prob = nn.Sigmoid()

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.prob(x)

    return x

# Option 2: Recurrent autoencoder, learn with both faulty and normal HD data
class RecurrentAutoencoder(nn.Module):
  """
  Recurrent Autoencoder
  :Learn it with normal HD data only. 
  Autoencoder learns the features from the normal HD data. Set threshold on loss value above which it is considered faulty input. 
  Above a certain level the features of the faulty HD are too much different from the features of the normal HD (learned inside the NN)
  """
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()
    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
  
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x 

class Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()
    
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )
  
  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return hidden_n.reshape((self.n_features, self.embedding_dim))
    
class Decoder(nn.Module):
  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()
    
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features
    
    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.output_layer = nn.Linear(self.hidden_dim, n_features)
  
  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))
    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))
    return self.output_layer(x)
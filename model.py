import torch 
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import math

# define word embedding class

class WordEmbedding(nn.Module):
    def __init__(self,d_model,vocab_size,dropout):
        super(WordEmbedding,self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding=nn.Embedding(d_model,vocab_size)
    def forward(self,input):
        return self.embedding(input)

# define the positional embedding

class PositionalEmbedding(nn.Module):
  def __init__(self,d_len:int,seq_len:int,dropout:float):
    super().__init__()
    self.d_len=d_len
    self.seq_len=seq_len
    self.dropout=nn.Dropout(dropout) # how much we gonna drop the connection 

    # create a matrix of shape seq_len,d_len
    PE=torch.zeros(seq_len,d_len)

    position=torch.arange(0,seq_len).unsqueeze(1) # shape (seq_len,1)

    div=torch.exp(torch.arange(0,d_len,2).float()*(-math.log(10000.0)/d_len))

    #for even position use sin and for odd position use cos
    PE[:,0::2]=torch.sin(position*div)
    PE[:,1::2]=torch.cos(position*div)

    #add batch dimension

    PE=PE.unsqueeze(0)
    self.register_buffer('PE',PE)
  
  def forward(self,input):
    input=input+(self.PE[:,:input.shape[1]:]).requires_grad(False)
    return self.dropout(input)
  
# define layer normalization
class LayerNormalization(nn.Module):
  def __init__(self,epsilon:float=10**-5):
    super().__init__()
    self.eps=epsilon
    self.alpha=nn.Parameter(torch.ones(1))
    self.bias=nn.Parameter(torch.zeros(1))
  
  def forward(self,input):
    mean=input.mean(dim=-1,keepdims=True)
    std=input.std(dim=-1,keepdims=True)
    return self.alpa*(input -mean)/(std+self.eps)+self.bias


# Define the multi-head self-attention layer
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        
        # Linear transformations for queries, keys, and values
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        # Split the embedding into heads and perform linear transformations
        batch_size, seq_len, _ = x.size()
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.embed_dim // self.num_heads, dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Linear transformation for output
        output = self.out(attention_output)
        return output

# Define the feed-forward network
class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.linear2(x)
        return x



# Define the GPT-2 model using the components defined above
class GPT2(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers):
        super(GPT2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding =PositionalEmbedding(embed_dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MultiHeadSelfAttention(embed_dim, num_heads),
                nn.LayerNorm(embed_dim),
                FeedForwardNetwork(embed_dim, ff_dim),
                nn.LayerNorm(embed_dim)
            ]) for _ in range(num_layers)
        ])
        self.out = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for attn, norm1, ff, norm2 in self.layers:
            x = norm1(x + attn(x))
            x = norm2(x + ff(x))
        x = self.out(x)
        return x

# Initialize the model, optimizer, and loss function
vocab_size = 10000  # Replace with actual vocab size
embed_dim = 256  # Replace with desired embedding dimension
num_heads = 4
ff_dim = 512
num_layers = 4

model = GPT2(vocab_size, embed_dim, num_heads, ff_dim, num_layers)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Assuming you have a DataLoader for your dataset
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch  # Replace this with how your data is prepared
        
        # Forward pass
        outputs = model(input_ids)
        
        # Flatten the predictions and labels
        outputs = outputs.view(-1, outputs.size(-1))
        labels = input_ids.view(-1)
        
        # Calculate loss and perform backpropagation
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'gpt2_model.pth')


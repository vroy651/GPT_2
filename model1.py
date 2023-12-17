import torch 
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import math

if torch.cuda.is_available():
    device = torch.device("cuda")  # CUDA device object
    print("CUDA is available.")
else:
    device = torch.device("cpu")   # Use CPU if CUDA is not available
    print("CUDA is not available. Using CPU.")

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
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_seq_len):
        super(RotaryPositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        self.rotary_dim = embed_dim // 2  # Divide embedding dimension by 2 for rotational components
        
        # Create sinusoidal positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, self.rotary_dim)
        self.pos_embedding.weight.data.uniform_(-1, 1)  # Initialize with random values between -1 and 1
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        # Get sinusoidal positional embeddings
        sin_emb = self.pos_embedding(positions)
        cos_emb = torch.roll(self.pos_embedding.weight, shifts=-1, dims=0)
        
        # Apply rotational operations
        sin_pos = torch.sin(sin_emb)
        cos_pos = torch.cos(cos_emb)
        
        # Combine rotational embeddings
        rot_embeddings = torch.cat((sin_pos, cos_pos), dim=-1)
        
        return rot_embeddings

  
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


class GroupQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups):
        super(GroupQueryAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_groups = num_groups
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        assert embed_dim % num_groups == 0, "Embedding dimension must be divisible by the number of groups"
        
        self.group_size = embed_dim // num_groups
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.kv_linear = nn.Linear(embed_dim, embed_dim * 2)  # Combine linear layers for k and v
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Calculate queries
        queries = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate keys and values and split into groups
        k_v = self.kv_linear(x).view(batch_size, seq_len, 2, self.num_groups, self.group_size)
        keys, values = k_v[:, :, 0], k_v[:, :, 1]
        
        # Split keys and values into groups
        keys = keys.transpose(2, 3).contiguous().view(batch_size, seq_len, self.num_heads, self.num_groups, self.head_dim)
        values = values.transpose(2, 3).contiguous().view(batch_size, seq_len, self.num_heads, self.num_groups, self.head_dim)
        
        # Calculate attention scores within and across groups
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_scores = attention_scores.softmax(dim=-1)
        
        # Attend to values within and across groups
        attended_values = torch.matmul(attention_scores, values)
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, self.num_heads, seq_len, -1)
        
        return attended_values


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
        self.rotary_positional_encoding =RotaryPositionalEmbedding(embed_dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                GroupQueryAttention(embed_dim, num_heads),
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
vocab_size = 10000  
embed_dim = 256  
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
        input_ids = batch.to(device)  
        
        # Forward pass
        outputs = model(input_ids)
        
        # Flatten the predictions and labels
        outputs = outputs.view(-1, outputs.size(-1))
        labels = input_ids.view(-1).to(device)
        
        # Calculate loss and perform backpropagation
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'gpt2_model.pth')


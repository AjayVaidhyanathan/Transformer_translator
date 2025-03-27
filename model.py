import torch
import torch.nn as nn
import math
import numpy

## This model takes the input as vector tokens or int. You inorder to process them, you need to have vocabulary mapped to words. 
# If done, we can use and get embeddings.

class InputEmbeddings(nn.Module):
    '''
    The nn.Embedding layer does not directly take words as input. It requires integer indices representing words from a vocabulary.
    If you want to input words, you must first convert them to indices using a vocabulary (word-to-index mapping)
    
    i.e
    embedding_layer = InputEmbeddings(d_model, vocab_size)
    sample_input = torch.tensor([2, 5, 8, 3, 1])  # A sequence of token indices
    output = embedding_layer(sample_input) # Get the embedding output

    where,
        d_model is dimension: 256, 512
        vocab_size is total vocabulary in the dataset: 100 unique words 
    '''
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model) 

    def forward(self,x):
        return self.embeddings(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_length:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        #Creating a matrix of shape (seq_length, d_model)
        pe = torch.zeros(seq_length, d_model)

        #Creating vector shape (seq_length, 1)
        '''
        torch.arange is a PyTorch function that creates a 1D tensor with evenly spaced values in a specified range. It works similarly to Python’s range() but for tensors.
        torch.arange(start, end, step, dtype)
        where,
            start → (Optional) The starting value (default = 0).
            end → The stopping value (not included in the result).
            step → (Optional) The spacing between values (default = 1).
            dtype → (Optional) The data type (e.g., torch.float, torch.int).
        '''
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        #Applying sin (even) and cosine (odd) to extract positional encoding
        pe[:, 0::2] = torch.sin(position * division_term)
        pe[:, 1::2] = torch.cos(position * division_term)

        pe = pe.unsqueeze(0) # (1, seq_length, d_model)

        self.register_buffer('pe', pe)

    def forward(self,x):
            x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
            return self.dropout(x)

class LayerNormalisation(nn.Module):
    def __init__(self, eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps=eps
        self.alpha = nn.Parameter(torch.ones(1)) #Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) #Added

    def forward(self,x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)  
        return self.alpha * (x - mean) / (std + self.eps) + self.bias  
    
class FeedForwardBlock(nn.Module):

    '''
    Self-attention captures relationships between words (tokens), but it is still a linear operation.
    The FFN introduces non-linearity, allowing the model to learn more complex mappings.
    Without FFNs, Transformers would be just a stack of linear transformations and wouldn’t be able to learn deep representations
    The FFN expands features from 512 → 2048 (d_ff) and then projects them back to 512.
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self,x):
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))    
     


class MultiheadAttention(nn.Module):
        def __init__(self, d_model:int, h: int, dropout: float):
            super().__init__()
            assert d_model % h == 0, "d_model must be divisible by num_heads"
            self.d_model = d_model
            self.h = h
            self.d_k = d_model // h


            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)

            self.W_o = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        
        def attention(query, key, value, mask, dropout: nn.Dropout):
            d_k = query.shape[-1]
            
            attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)

            if mask is not None:
                attention_scores.masked_fill(mask == 0, -1e9)
            attention_scores = attention_scores.softmax(dim=-1)

            if dropout is not None:
                attention_scores = dropout(attention_scores)

            return (attention_scores @ value), attention_scores       

        def forward(self, q, k, v, mask):
        # Linear transformations
            query = self.W_q(q)  # (batch, seq_len, d_model)
            key = self.W_k(k)  
            value = self.W_v(v)  

            query.view(query.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2)
            key.view(key.shape[0],key.shape[1], self.h, self.d_k).transpose(1,2)
            value.view(value.shape[0],value.shape[1], self.h, self.d_k).transpose(1,2)

        
            x, self.attention_scores = MultiheadAttention.attention(query, key, value, mask, self.dropout)

            x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)  # (batch, seq_len, d_model)

            return self.W_o(x)
        

class ResidualConnection(nn.Module):
        def __init__(self, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.normalisation = LayerNormalisation()

        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.normalisation(x)))
        

class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiheadAttention, feed_forward: FeedForwardBlock, dropout: float) -> None:
            super().__init__()    
            self.self_attention=self_attention
            self.feed_forward=feed_forward
            self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self,x,source_mask):
         x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, source_mask))
         x = self.residual_connection[1](x, self.feed_forward)
         return x
    
class Encoder(nn.Module):
     def __init__(self, layers: nn.ModuleList) -> None:
          super().__init__()    
          self.layers = layers
          self.normalisation = LayerNormalisation()

     def forward(self, x, source_mask):
          for layer in self.layers:
               x= layer(x, source_mask)
          return self.normalisation(x)


class DecoderBlock(nn.Module):
     def __init__(self, self_attention: MultiheadAttention, feed_forward:FeedForwardBlock, cross_attention: MultiheadAttention, dropout: float):
          super().__init__()   
          self.self_attention = self_attention
          self.feed_forward = feed_forward
          self.cross_attention = cross_attention
          self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

     def forward(self,x,encoder_ouput,source_mask,target_mask):
          x = self.residual_connection[0](x, lambda x: self.self_attention(x,x,x, target_mask))
          x = self.residual_connection[1](x, lambda x: self.cross_attention(x,encoder_ouput,encoder_ouput,source_mask))
          x = self.residual_connection[2](x, self.feed_forward)
          return x
     
class Decoder(nn.Module):
     def __init__(self, layers:nn.ModuleList):
          super().__init__()     
          self.layers = layers
          self.normalisation = LayerNormalisation()

     def forward(self, x, encoder_output, source_mask, target_mask):
          for layer in self.layers:
               x = layer(x, encoder_output, source_mask, target_mask)
          return self.normalisation(x)    


class ProjectionLayer(nn.Module):
     def __init__(self, d_model: int, vocab_size: int):
          super().__init__()
          self.projection_layer = nn.Linear(d_model, vocab_size)

     def forward(self, x):
          #(batch, seq_length, d_model) -> (batch, seq_length, vocab_size)  
          return torch.log_softmax(self.projection_layer(x), dim=-1)


class Transformer(nn.Module):
     def __init__(self, encoder: Encoder, decoder: Decoder, source_embedding: InputEmbeddings, target_embedding: InputEmbeddings, source_position: PositionalEncoding, target_postion: PositionalEncoding, projection_layer: ProjectionLayer):
          super().__init__()    
          self.encoder = encoder
          self.decoder = decoder
          self.source_embedding = source_embedding
          self.target_embedding = target_embedding
          self.source_position = source_position
          self.target_postion = target_postion
          self.projection_layer = projection_layer

     def encode(self, source, source_mask):
          source = self.source_embedding(source)
          source = self.source_position(source)
          return self.encoder(source, source_mask)
     
     def decode(self, encoder_output, source_mask, target, target_mask):
          target = self.target_embedding(target)
          target = self.target_postion(target)
          return self.decoder(target, encoder_output, source_mask, target_mask)
     
     def project(self, x):
          return self.projection_layer(x)
     
def build_transformer(source_vocab_size: int, target_vocab_size: int, source_seq_length: int, target_seq_length: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
     #Creating embedding layer
     source_embedding = InputEmbeddings(d_model, source_vocab_size)
     target_embedding = InputEmbeddings(d_model, target_vocab_size)

     #Creating positional encoding
     source_position = PositionalEncoding(d_model, source_seq_length, dropout)
     target_position = PositionalEncoding(d_model, target_seq_length, dropout)

     #create encoder block
     encoder_blocks = []
     for _ in range(N):
          encoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
          feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
          encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
          encoder_blocks.append(encoder_block)

     decoder_blocks = []
     for _ in range(N):
          decoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
          decoder_cross_attention_block = MultiheadAttention(d_model, h, dropout)
          feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
          decoder_block = DecoderBlock(decoder_self_attention_block, feed_forward_block, decoder_cross_attention_block, dropout)
          decoder_blocks.append(decoder_block)     

     encoder = Encoder(nn.ModuleList(encoder_blocks))
     decoder = Decoder(nn.ModuleList(decoder_blocks))  


     projection_layer = ProjectionLayer(d_model, target_vocab_size)

     transformer = Transformer(encoder, decoder, source_embedding, target_embedding, source_position, target_position, projection_layer)

     for p in transformer.parameters():
          if p.dim()>1:
               nn.init.xavier_uniform_(p)   

     return transformer          






          



        
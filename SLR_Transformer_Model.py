import torch
from torch import nn
import math

'''
This is a tranformer model that is based on the research paper "Attention is all you need"
'''
# Input size aka d_model -> usually the default is 512

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        '''
        Initialises a Multi-Head Attention Block.
        
        args:
            d_model: Size of the input. Aka. d_model
            num_heads: number of heads for Q, K, V    
        '''
        # Input must be dividible by the number of heads
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.values_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear = nn.Linear(d_model, d_model, bias=False)
        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        
    def scaled_dot_product_attention(self, v, k, q, mask=None):
        '''
        Returns the Scaled Dot Product Attention from the v, k, q 
        (Value, Key, Query). Masking can be added optionally 
        (useful for decoders).
        
        Args:
            v: Value Tensor
            k: Key Tensor
            q: Query Tensor
            mask: The masking value (for decoder)
        '''
        
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # mask for decoding
        if mask != None:
            score *= mask
            # score = score.masked_fill(mask == 0, -1e9)
        
        # Using the formula from the paper
        softmax_score = torch.softmax(score, -1)
        attention = torch.matmul(softmax_score, v)
        
        return attention
    
    # Shape [batch_size, Sequence_length, Input] -> [batch_size, Sequence_length, Num_heads, Head_dim]
    def split_to_heads(self, x):
        '''
        Splits the input Tensor between the heads and returns it. 
        (shape: [batch_size, Sequence_length, Num_heads, Head_dim])
        
        Args:
            x: Input Tensor (shape: [batch_size, Sequence_length, Input])
        '''
        
        
        
        batch_size, sequence_length, _ = x.size()
        return x.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
    
    # [batch_size, Sequence_length, Num_heads, Head_dim] -> [batch_size, Sequence_length, Input]
    def combine_heads(self, x):
        '''
        Combines the split input Tensor  returns it. 
        (shape: [batch_size, Sequence_length, Input])
        
        Args:
            x: Input Tensor (shape: [batch_size, Sequence_length, Num_heads, Head_dim])
        '''
        
        batch_size, sequence_length, _, _ = x.size()
        return x.reshape(batch_size, sequence_length, self.d_model)
    
    
    def forward(self, q, k, v, mask=None):
        '''
        Computing the attention.
        
        Args:
            v: Value Tensor
            k: Key Tensor
            q: Query Tensor
            mask: The masking value (for decoder)
        '''
        
        q = self.split_to_heads(self.query_linear(q))
        k = self.split_to_heads(self.key_linear(k))
        v = self.split_to_heads(self.values_linear(v))
        
        attention = self.scaled_dot_product_attention(v, k, q, mask)
        return self.output_linear(self.combine_heads(attention))
        
# Hidden layer size. AKA. d_ff
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_size):
        super(FeedForward, self).__init__()
        '''
        Creates a Feed Forward block.
        
        Args:
            d_model: Size of the input. Aka d_model
        '''
        
        self.linear1 = nn.Linear(d_model, hidden_size)
        self.linear2 = nn.Linear(hidden_size, d_model)
        self.relu_activation = nn.ReLU()
        
    # Keep an eye on this for later tweaking
    def forward(self, x):
        '''
        Feed-forward calculation
        
        Args:
            x: Input Tensor
        '''
        
        relu_linear = self.relu_activation(self.linear1(x))
        
        return self.linear2(relu_linear)
    
# Positional Encoding -> for both the encoder and decoder
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence, dropout):
        super(PositionalEncoding, self).__init__()
        '''
        Positional encding for the inputs. Makes sure that the position of the 
        inputs 

        Args:
            d_model: The size of the input. Aka. d_model
            max_sequence: The maximum sequence length
            dropout: The dropout value
        '''
    
        self.droupout = nn.Dropout(dropout)
        
        even_seq = torch.arange(0, d_model, 2).float()
        
        # positions from 0 to max_sequence
        position = torch.arange(0, max_sequence, dtype=torch.float).unsqueeze(1)
        denom  = torch.exp(even_seq * -(math.log(10000.0) / d_model)) # This method tends to be more efficient than using pow.8

        pe = torch.zeros(max_sequence, d_model)

        # Sin for even values, Cos for odd values
        pe[:, 0::2] = torch.sin(position * denom)
        pe[:, 1::2] = torch.cos(position * denom)

        pe = pe.unsqueeze(0)

        # Some papers add an extra dimension and transpose
        # For simplicity, I just kept it as is

        # Save as state instead of model parameter
        self.register_buffer('pe', pe) 



    def forward(self, x):
        '''
        Adds positional encoding to the inputs

        Args:
            x: Input Tensor [batch, sequence_len, d_model]
        '''
        
        import numpy as np
        output = x + self.pe[:, :x.size(1), :x.size(2)]

        return self.droupout(output)
        
# Encoder Layer
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, hidden_size, dropout):
        super(Encoder, self).__init__()
        '''
        Encoder block for the Transformer model.

        Args:
            d_model: Size of the input. Aka. d_model
            num_heads: number of heads for Multi-Head Attention block
            hidden_size: Hidden size of inputs. Aka. d_ff
            dropout: The dropout value
        '''



        # Dropout is added to Positional encoding
        self.mutli_head_attention_block = MultiHeadAttention(d_model, num_heads)
        self.feed_forward_block = FeedForward(d_model, hidden_size)
        self.normalisation_layer1 = nn.LayerNorm(d_model)
        self.normalisation_layer2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        Encodes the input for the deoder block.

        Args:
            d_model: The size of the input. Aka. d_model
            num_heads: The number of heads
            hidden_size: The size of the hidden layer. Aka. d_ff.
        '''

        attention = self.dropout(self.mutli_head_attention_block(x, x, x))
        normalised_attiention = self.normalisation_layer1(x + attention)

        feed_forward = self.dropout(self.feed_forward_block(normalised_attiention))
        output = self.normalisation_layer2(feed_forward + normalised_attiention)

        return output


# Original decoder Layer
# class Decoder(nn.Module):
#     def __init__(self, d_model, num_heads, hidden_size, dropout):
#         super(Decoder, self).__init__()
#         '''
#         Decoder block for the Transformer model.

#         Args:
#             d_model: Size of the input. Aka. d_model
#             num_heads: number of heads for Multi-Head Attention block
#             hidden_size: Hidden size of inputs. Aka. d_ff
#             dropout: The dropout value
#         '''

#         self.masked_mutli_head_attention_block = MultiHeadAttention(d_model, num_heads)
#         self.mutli_head_attention_block = MultiHeadAttention(d_model, num_heads)
#         self.feed_forward_block = FeedForward(d_model, hidden_size)

#         self.normalisation_layer1 = nn.LayerNorm(d_model)
#         self.normalisation_layer2 = nn.LayerNorm(d_model)
#         self.normalisation_layer3 = nn.LayerNorm(d_model)

#         self.dropout = nn.Dropout(dropout)



#     def forward(self, x, encoder_output, mask):
#         '''
#         Decodes the input from the Encoder for the output.

#         Args:
#             d_model: The size of the input. Aka. d_model
#             num_heads: The number of heads
#             hidden_size: The size of the hidden layer. Aka. d_ff.
#         '''

#         masked_attention = self.dropout(self.masked_mutli_head_attention_block(x, x, x, mask))
#         normalised_masked_attention = self.normalisation_layer1(x + masked_attention)

#         attention = self.dropout(self.mutli_head_attention_block(encoder_output, encoder_output, x))
#         normalised_attention = self.normalisation_layer2(normalised_masked_attention + attention)

#         feed_forward = self.dropout(self.feed_forward_block(normalised_attention))
#         output = self.normalisation_layer3(normalised_attention + feed_forward)

#         return output

# Modified Decoder Layer
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, hidden_size, dropout):
        super(Decoder, self).__init__()
        '''
        Decoder block for the Transformer model. Based on the SPOTER model.

        Args:
            d_model: Size of the input. Aka. d_model
            num_heads: number of heads for Multi-Head Attention block
            hidden_size: Hidden size of inputs. Aka. d_ff
            dropout: The dropout value
        '''
        
        # Masked Multi-Head attention is redundand, since decoder input will only have one word

        self.mutli_head_attention_block = MultiHeadAttention(d_model, num_heads)
        self.feed_forward_block = FeedForward(d_model, hidden_size)

        self.normalisation_layer1 = nn.LayerNorm(d_model)
        self.normalisation_layer2 = nn.LayerNorm(d_model)
        self.normalisation_layer3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)



    def forward(self, x, encoder_output):
        '''
        Decodes the input from the Encoder for the output.

        Args:
            d_model: The size of the input. Aka. d_model
            num_heads: The number of heads
            hidden_size: The size of the hidden layer. Aka. d_ff.
        '''
        

        x_droupout = self.dropout(x)
        x_droupout = self.normalisation_layer1(x + x_droupout)

        attention = self.dropout(self.mutli_head_attention_block(x, encoder_output, encoder_output))
        normalised_attention = self.normalisation_layer2(x_droupout + attention)

        feed_forward = self.dropout(self.feed_forward_block(normalised_attention))
        output = self.normalisation_layer3(normalised_attention + feed_forward)

        return output

        
# Transformer model
class SLRTransformer(nn.Module):
    def __init__(self,
                 d_model, 
                 feedforward_hidden_size,
                 num_layers,
                 max_sequence,
                 num_classes, 
                 num_heads, 
                 dropout
                 ):
        super(SLRTransformer, self).__init__()
        '''
        SLR Transformer that has been tweaked slightly to allow for the classification if signs.
        The output will not be fead through a softmax layer, since it will make use of 
        cross-entropy loss (which automatically implements this). The source vocab size and the
        tarket vocab size are kept separate to allow for flexibility. Heavily inspired by SPOTER model.

        Args:
            d_model: the hidden size of the transformer's hidden dimension. Aka. hidden_dim in SPOTER
            feedforward_hidden_size: The size of the feed forward hidden layer. Aka. d_ff.
            num_layers: Number of encoder & decoder layers
            max_sequence: The maximum sequence length
            input_vocab_size: Vocabulary size of the input
            output_vocab_size: Vocabulary size of the output
            num_heads: The number of heads
            dropout: The dropout value
        '''
        
        
        self.row_embed = nn.Parameter(torch.rand(50, d_model))
        self.pos = nn.Parameter(torch.cat([self.row_embed[0].unsqueeze(0).repeat(1, 1, 1)], dim=-1).flatten(0, 1).unsqueeze(0))
        
        

        self.max_sequence = max_sequence

        self.class_query = nn.Parameter(torch.rand(1, d_model))

        self.encoder_block = nn.ModuleList([Encoder(d_model, num_heads, feedforward_hidden_size, dropout) for _ in range(num_layers)])
        self.decoder_block = nn.ModuleList([Decoder(d_model, num_heads, feedforward_hidden_size, dropout) for _ in range(num_layers)])

        self.linear_output = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        '''
        Passing the input through encoder and decoder layers for sign language recognition.
        
        Args:
            source: Input to the transformer. Aka. encoder input
        '''

        positional_encoder_input = x + self.pos
        
        # Makes sure that the initial iteration passes the encoded input
        encoder_output = positional_encoder_input
        for encoder_head in self.encoder_block:
            
            # For this implementation, the encoder does not have any mask
            encoder_output = encoder_head(encoder_output)
        
        
        # Makes sure that the initial iteration passes the encoded input
        decoder_output = self.class_query.unsqueeze(0)
        for decoder_head in self.decoder_block:
            
            decoder_output = decoder_head(decoder_output, encoder_output)
            
        output = self.linear_output(decoder_output.transpose(0, 1))
        
        return output
    
'''
Basic GPT4Rec
'''

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import GPT2Model, GPT2Config


class GPT4RecommendationBaseModel(nn.Module):
    '''
        The base class for collaborative GPT model, i.e.,
        the GPT model with extra user/item embeddings
    '''
    def __init__(self, config, gpt2model):
        super(GPT4RecommendationBaseModel, self).__init__()
        # Obtain the number of users, items
        # and the size of the original vocabulary
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.vocab_size = config.vocab_size
        self.config = config

        # Create new token embeddings for user/item tokens
        self.user_embeddings = nn.Embedding(self.num_users, config.n_embd)
        self.item_embeddings = nn.Embedding(self.num_items, config.n_embd)

        # Randomly initialize the new token embeddings
        self.user_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.item_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        
        # The pretrained gpt2 model
        self.gpt2model = gpt2model
        
    def embed(self, input_ids):
        # input_ids is a tensor of shape (batch_size, seq_length)
        vocab_mask = (input_ids < self.vocab_size).long() 
        user_mask = ((input_ids >= self.vocab_size) & (input_ids < self.vocab_size + self.num_users)).long() 
        item_mask = (input_ids >= self.vocab_size + self.num_users).long()
        
        # IDs outside of vocab range are set to 0
        vocab_ids = (input_ids * vocab_mask).clamp_(0, self.vocab_size-1)  
        vocab_embeddings = self.gpt2model.wte(vocab_ids)
        vocab_embeddings = vocab_embeddings * vocab_mask.unsqueeze(-1)
        
        # IDs outside of user range are set to 0
        user_ids = ((input_ids - self.vocab_size) * user_mask).clamp_(0, self.num_users-1)
        user_embeddings = self.user_embeddings(user_ids)
        user_embeddings = user_embeddings * user_mask.unsqueeze(-1)
        
        # IDs outside of item range are set to 0
        item_ids = ((input_ids - self.vocab_size - self.num_users) * item_mask).clamp_(0, self.num_items-1)
        item_embeddings = self.item_embeddings(item_ids)
        item_embeddings = item_embeddings * item_mask.unsqueeze(-1)

        # Sum up the embeddings as the input embeddings
        input_embeddings = vocab_embeddings + user_embeddings + item_embeddings
        return input_embeddings
        
    def forward(self, input_ids=None, **kwargs):
        # Obtain the embeddings of the input id sequence
        input_embeddings = self.embed(input_ids)
        # The input_embeds will be summed up with the pos_embed
        # And then forward into the transformer to get the results
        return self.gpt2model(inputs_embeds=input_embeddings, **kwargs)
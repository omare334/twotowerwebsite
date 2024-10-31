# %% Import necessary libraries
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import psycopg2
from tqdm import tqdm
import torch.nn.functional as F

# Set device to CPU explicitly
device = torch.device("cpu")

# Database connection function
def get_db_connection():
    conn = psycopg2.connect(
        host='postgres',  # Update with your actual host
        database='passages',
        user='omarewis',
        password='ommmaha260801'
    )
    return conn

# Load results from PostgreSQL
def load_results():
    conn = get_db_connection()
    query = 'SELECT query, passage_text, negative_sample FROM results;'  # Adjusted to load all required columns
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

class TowerOneRNN(nn.Module):
    def __init__(self):
        super(TowerOneRNN, self).__init__()
        self.rnn = nn.RNN(input_size=64, hidden_size=512, num_layers=3, batch_first=True)

    def forward(self, x, lengths):
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x[torch.arange(x.size(0)), lengths - 1]

class TowerTwoRNN(nn.Module):
    def __init__(self):
        super(TowerTwoRNN, self).__init__()
        self.rnn = nn.RNN(input_size=64, hidden_size=512, num_layers=3, batch_first=True)

    def forward(self, x, lengths):
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x[torch.arange(x.size(0)), lengths - 1]

# Load results from PostgreSQL
results = load_results()  # Use the function to load results
results.reset_index(drop=True, inplace=True)  # Reset index if necessary

# Load tower two outputs
tower_two_outputs = torch.load('tower_two_outputs.pt', map_location=device)

# Load TowerOne model
tower_one = TowerOneRNN()
checkpoint = torch.load("model_2_checkpoint_epoch_6.pth", map_location=device)
tower_one.load_state_dict(checkpoint['tower_one_state_dict'])
tower_one.eval()

# Define the SkipGramFoo model
class SkipGramFoo(nn.Module):
    def __init__(self, voc, emb, ctx):
        super(SkipGramFoo, self).__init__()
        self.ctx = ctx
        self.emb = nn.Embedding(num_embeddings=voc, embedding_dim=emb)
        self.ffw = nn.Linear(in_features=emb, out_features=voc, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, inpt, trgs, rand):
        emb = self.emb(inpt)
        batch_size = inpt.size(0)
        rand = rand[:batch_size]
        
        ctx = self.ffw.weight[trgs.to(inpt.device)]
        rnd = self.ffw.weight[rand.to(inpt.device)]
        
        out = torch.bmm(ctx.view(batch_size, 1, -1), emb.unsqueeze(2)).squeeze(2)
        rnd = torch.bmm(rnd.view(batch_size, 1, -1), emb.unsqueeze(2)).squeeze(2)

        out = self.sig(out).clamp(min=1e-7, max=1 - 1e-7)
        rnd = self.sig(rnd).clamp(min=1e-7, max=1 - 1e-7)

        pst = -out.log().mean()
        ngt = -(1 - rnd).log().mean()
        
        return pst + ngt

# Load updated vocabulary
with open("updated_vocab_dict.pkl", "rb") as f:
    updated_vocab = pickle.load(f)

# Function to create a reverse vocabulary
def create_reverse_vocab(vocab):
    return {word: index for index, word in vocab.items()}

reverse_vocab = create_reverse_vocab(updated_vocab)

# Function to tokenize a single text using the reverse vocabulary
def tokenize_text(text, reverse_vocab):
    words = text.lower().split()  # Convert the text to lowercase
    tokenized = [reverse_vocab[word] for word in words if word in reverse_vocab]
    return tokenized

# Initialize and load the SkipGram model
embedding_dim = 64
model_path = "finetuned_skipgram_model.pth"
mFoo = SkipGramFoo(len(updated_vocab), embedding_dim, 2).to(device)
mFoo.load_state_dict(torch.load(model_path, map_location=device), strict=False)
mFoo.eval()

# Function to get embedding for a single token
def get_embedding_for_single_token(tokens, model):
    with torch.no_grad():
        if len(tokens) > 0:
            token_tensor = torch.LongTensor(tokens).to(device)
            token_embedding = model.emb(token_tensor)  # Shape: [num_tokens, embedding_dim]
            return token_embedding.cpu().numpy()  # Return the full sequence embeddings
        else:
            return torch.zeros((1, embedding_dim)).cpu().numpy()  # Zero vector for empty sequences

# Function to get top-k passages
def get_top_k_passages(query_text, top_k=10):
    tokenized_query = tokenize_text(query_text, reverse_vocab)
    query_embedding = get_embedding_for_single_token(tokenized_query, mFoo)
    query_embedding_tensor = torch.tensor(query_embedding).unsqueeze(0)
    length = torch.tensor([len(query_embedding)])

    with torch.no_grad():
        tower_one_output = tower_one(query_embedding_tensor, length)
        similarities = F.cosine_similarity(tower_one_output, tower_two_outputs.to(device), dim=1)

    top_k_indices = torch.topk(similarities, top_k).indices.numpy()
    return results['passage_text'].iloc[top_k_indices].tolist(), similarities[top_k_indices].tolist()

# Streamlit UI
st.title("Passage Similarity Finder")
query_text = st.text_input("Enter a query", "cooking chicken")

if st.button("Find Similar Passages"):
    passages, scores = get_top_k_passages(query_text)
    st.write(f"Top {len(passages)} similar passages:")
    for i, (passage, score) in enumerate(zip(passages, scores), start=1):
        st.write(f"{i}. **Similarity:** {score:.4f}")
        st.write(f"**Passage:** {passage}")
        st.write("---")

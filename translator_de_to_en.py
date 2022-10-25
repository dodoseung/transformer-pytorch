# Refer to https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice
from transformer import Transformer

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
from torchtext.data.metrics import bleu_score

import spacy
import time
import math

# Parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
N_EPOCHS = 12

# Pytorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tokenization
spacy_en = spacy.load('en') # English
spacy_de = spacy.load('de') # Deutsch

# Set the field
SRC = Field(tokenize = 'spacy', tokenizer_language='en', init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)
TRG = Field(tokenize = 'spacy', tokenizer_language='de', init_token="<sos>", eos_token="<eos>", lower=True, batch_first=True)

# Data split
train_dataset, valid_dataset, test_dataset = Multi30k.splits(exts=(".de", ".en"), fields=(SRC, TRG), root='nlp-01-transformer-pytorch/data')

# Build vocab
SRC.build_vocab(train_dataset, min_freq=2)
TRG.build_vocab(train_dataset, min_freq=2)

# Split the dataset
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, valid_dataset, test_dataset), batch_size=BATCH_SIZE, device=device)

# Set the padding token
src_pad_idx = SRC.vocab.stoi[SRC.pad_token]
trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]

# Set the vocab length
src_vocab_len = len(SRC.vocab)
trg_vocab_len = len(TRG.vocab)

# Transformer
model = Transformer(num_encoder_layer=3, num_decoder_layer=3,
                 d_model=256, num_heads=8, d_ff=512, dropout_rate=0.1, 
                 src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx,
                 src_vocab_size=src_vocab_len, trg_vocab_size=trg_vocab_len,
                 max_seq_len=100, device=device).to(device)

# Set the optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Set the criterion as the cross entropy loss
criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx)

# Training
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0

    for _, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        output = model(src, trg[:,:-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)

        optimizer.zero_grad()
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Evaluation
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    for _, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        output = model(src, trg[:,:-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)

        loss = criterion(output, trg)

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Translation
def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50, logging=True):
    model.eval()

    # Set the de source
    tokens = [token.text.lower() for token in spacy_de(sentence)]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.padding_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    # Set the en target
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    # Inference the character step by step
    for _ in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.look_ahead_mask(trg_tensor)

        with torch.no_grad():
            output = model.predict(trg_tensor, trg_mask, enc_src, src_mask)
        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:-1]

# BLEU score
def show_bleu(data, src_field, trg_field, model, device, max_len=50):
    trgs = []
    pred_trgs = []
    index = 0

    for datum in data:
        src = vars(datum)['src']
        trg = vars(datum)['trg']

        pred_trg = translate_sentence(src, src_field, trg_field, model, device, max_len, logging=False)

        pred_trgs.append(pred_trg)
        trgs.append([trg])

        index += 1
        if (index + 1) % 100 == 0:
            print(f"[{index + 1}/{len(data)}]")
            print(f"TRG: {trg}")
            print(f"Model: {pred_trg}")   

    bleu = bleu_score(pred_trgs, trgs, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
    print(f'Total BLEU Score = {bleu*100:.2f}')

    s1 = bleu_score(pred_trgs, trgs, max_n=4, weights=[1, 0, 0, 0])
    s2 = bleu_score(pred_trgs, trgs, max_n=4, weights=[0, 1, 0, 0])
    s3 = bleu_score(pred_trgs, trgs, max_n=4, weights=[0, 0, 1, 0])
    s4 = bleu_score(pred_trgs, trgs, max_n=4, weights=[0, 0, 0, 1])
    print(f'Individual BLEU1, 2, 3, 4 scores = {s1*100:.2f}, {s2*100:.2f}, {s3*100:.2f}, {s4*100:.2f},') 

    c1 = bleu_score(pred_trgs, trgs, max_n=4, weights=[1, 0, 0, 0])
    c2 = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/2, 1/2, 0, 0])
    c3 = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/3, 1/3, 1/3, 0])
    c4 = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/4, 1/4, 1/4, 1/4])
    print(f'Cumulative BLEU1, 2, 3, 4 score = {c1*100:.2f}, {c2*100:.2f}, {c3*100:.2f}, {c4*100:.2f}') 

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'nlp-01-transformer-pytorch/transformer_german_to_english.pt')

    print(f'\nEpoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s | Train loss: {train_loss:.3f} | Valid loss: {valid_loss:.3f}')
    
    example_idx = 1

    src = vars(test_dataset.examples[example_idx])['src']
    trg = vars(test_dataset.examples[example_idx])['trg']
    translation = translate_sentence(src, SRC, TRG, model, device, logging=True)

    print(f'SRC: {src}')
    print(f'TRG: {trg}')
    print("Model:", " ".join(translation))

show_bleu(test_dataset, SRC, TRG, model, device)
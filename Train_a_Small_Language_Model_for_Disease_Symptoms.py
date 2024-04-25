"""
Author : Janarddan Sarkar
File_name = Train_a_Small_Language_Model_for_Disease_Symptoms.py
Date : 25-04-2024
Description :
Huggingface model - https://huggingface.co/distilbert/distilgpt2
Huggingface Dataset - https://huggingface.co/datasets/QuyenAnhDE/Diseases_Symptoms

Code original sourse - https://github.com/AIAnytime/Training-Small-Language-Model
"""

from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
import ast
import datasets
from tqdm import tqdm
import time

# Load data
data_sample = load_dataset("QuyenAnhDE/Diseases_Symptoms")
print(data_sample)

updated_data = [{'Name':item['Name'], 'Symptoms':item['Symptoms']} for item in data_sample['train']]
df = pd.DataFrame(updated_data)
print(df.head())

# Just extract the symptoms
df['Symptoms'] = df['Symptoms'].apply(lambda x: ', '.join(x.split(', ')))

print(df.head())

#
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    # try:
    #     device = torch.device('mps')
    # except Exception:
    #     device = torch.device('cpu')

print("Device: ", device)

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2').to(device)
print("Model: ", model)
"""
1. wte:   Embedding layer for token embeddings. It maps token indices to their corresponding vector representations.
2. wpe:   Embedding layer for position embeddings. It maps positions in the input sequence to their corresponding vector 
          representations.
3. drop:  Dropout layer
4. h:     A list of GPT2Blocks. Each GPT2Block represents a transformer block in the GPT-2 model. 
          The model seems to have 6 transformer blocks based on the naming convention (0-5).
          
   4.1. ln_1:  Layer normalization applied before the self-attention mechanism within the transformer block.
   4.2. attn: The self-attention mechanism (GPT2Attention) within the transformer block. It calculates attention scores 
              between each input token and attends to relevant tokens.
              -> c_attn:        Convolutional layer for attention computation.
              -> c_proj:        Convolutional layer for projecting attention outputs.
              -> attn_dropout:  Dropout applied to attention scores.
              -> resid_dropout: Dropout applied to the residual connection.
    4.3 ln_2:  Layer normalization applied after the self-attention mechanism within the transformer block.
    4.4 mlp:   Multi-layer perceptron (MLP) layer ('GPT2MLP') within the transformer block.
               -> c_fc: Convolutional layer for the feedforward neural network.
               -> c_proj: Convolutional layer for projecting MLP outputs.
               -> act: Activation function. It appears to be a custom activation function called NewGELUActivation().
               -> dropout: Dropout applied to the MLP outputs.
               
5. ln_f:    Layer normalization applied after the last transformer block.
6. lm_head: Linear layer projecting the transformer output to a vocabulary-sized output. 
            It is responsible for predicting the next token in the sequence.
"""
"""
---------------------------------------------------------------------------------------------------------------------
In a transformer model like GPT-2, c_attn and c_proj are components used in the attention mechanism.

************ c_attn (Attention Computation): *********************
1. It's like a mechanism that decides which words or tokens in a sentence are important to focus on.
2. Imagine you're reading a sentence and trying to understand it. You might focus more on certain words based on their 
   relevance to the overall meaning. That's what c_attn does.
3. This component computes the attention scores, which represent how much each token should attend to other tokens in
   the sequence.
4. It's like assigning importance weights to each word in a sentence based on how relevant they are to each other.

***************** c_proj (Projection) ******************************
1. Once we have the attention scores, we need to use them to combine information from different words or tokens.
2. Think of it as taking all the attention-weighted words and putting them together to get a representation of what's 
  important in the sentence.
3. c_proj takes these attended representations and projects them to a different space (usually with a different
   dimensionality), making them ready for the next step in the neural network.
4. It's like transforming the weighted words into a new representation that captures the most important aspects 
   of the sentence.

In short, c_attn decides what to focus on in the input sequence, and c_proj combines this focused information into
a new representation that the model can use for further processing. Together, they play a crucial role in the attention 
mechanism of transformer models like GPT-2, helping the model understand and generate text effectively.
"""

BATCH_SIZE = 8
print(df.describe())

# Dataset Prep
class LanguageDataset(Dataset):
    """
    An extension of the Dataset object to:
      - Make training loop cleaner
      - Make ingestion easier from pandas df's
    """
    def __init__(self, df, tokenizer):
        self.labels = df.columns
        self.data = df.to_dict(orient='records')
        self.tokenizer = tokenizer
        x = self.fittest_max_length(df)  # Fix here
        self.max_length = x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][self.labels[0]]
        y = self.data[idx][self.labels[1]]
        text = f"{x} | {y}"
        tokens = self.tokenizer.encode_plus(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        return tokens

    def fittest_max_length(self, df):  # Fix here
        """
        Smallest power of two larger than the longest term in the data set.
        Important to set up max length to speed training time.
        """
        max_length = max(len(max(df[self.labels[0]], key=len)), len(max(df[self.labels[1]], key=len)))
        x = 2
        while x < max_length: x = x * 2
        return x

# Cast the Huggingface data set as a LanguageDataset we defined above
data_sample = LanguageDataset(df, tokenizer)

print(data_sample)

# Create train, valid
train_size = int(0.8 * len(data_sample))
valid_size = len(data_sample) - train_size
train_data, valid_data = random_split(data_sample, [train_size, valid_size])

# Make the iterators
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)

# Set the number of epochs
num_epochs = 2

# Training parameters
batch_size = BATCH_SIZE
model_name = 'distilgpt2'
gpu = 0

# Set the learning rate and loss function
## CrossEntropyLoss measures how close answers to the truth.
## More punishing for high confidence wrong answers
criterion = nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)
# The ignore_index parameter is set to tokenizer.pad_token_id. This means that during the loss calculation,
# any predictions corresponding to the padding token (used for padding sequences to the same length) will be ignored.
# This is helpful in tasks where sequences are padded to the maximum length.
optimizer = optim.Adam(model.parameters(), lr=5e-4)
tokenizer.pad_token = tokenizer.eos_token
print('EOS token: ', tokenizer.eos_token)
print('pad token: ', tokenizer.pad_token)

# Init a results dataframe
results = pd.DataFrame(columns=['epoch', 'transformer', 'batch_size', 'gpu', 'training_loss', 'validation_loss',
                                'epoch_duration_sec'])

# The training loop
for epoch in range(num_epochs):
    start_time = time.time()  # Start the timer for the epoch

    # Training
    # This line tells the model we're in 'learning mode' i.e. in training mode
    model.train()
    epoch_training_loss = 0
    train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs} Batch Size: {batch_size}, Transformer: {model_name}")
    for batch in train_iterator:
        optimizer.zero_grad()
        inputs = batch['input_ids'].squeeze(1).to(device)
        targets = inputs.clone()
        outputs = model(input_ids=inputs, labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_iterator.set_postfix({'Training Loss': loss.item()})
        epoch_training_loss += loss.item()
    avg_epoch_training_loss = epoch_training_loss / len(train_iterator)

    # Validation
    # This line below tells the model to 'stop learning' i.e., we are in evaluation mode now
    model.eval()
    epoch_validation_loss = 0
    total_loss = 0
    valid_iterator = tqdm(valid_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}")
    with torch.no_grad():
        for batch in valid_iterator:
            inputs = batch['input_ids'].squeeze(1).to(device)
            targets = inputs.clone()
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss
            total_loss += loss
            valid_iterator.set_postfix({'Validation Loss': loss.item()})
            epoch_validation_loss += loss.item()

    avg_epoch_validation_loss = epoch_validation_loss / len(valid_loader)

    end_time = time.time()  # End the timer for the epoch
    epoch_duration_sec = end_time - start_time  # Calculate the duration in seconds

    new_row = {'transformer': model_name,
               'batch_size': batch_size,
               'gpu': gpu,
               'epoch': epoch+1,
               'training_loss': avg_epoch_training_loss,
               'validation_loss': avg_epoch_validation_loss,
               'epoch_duration_sec': epoch_duration_sec}  # Add epoch_duration to the dataframe

    results.loc[len(results)] = new_row
    print(f"Epoch: {epoch+1}, Validation Loss: {total_loss/len(valid_loader)}")

# Test the code
input_str = "Kidney Failure"
input_ids = tokenizer.encode(input_str, return_tensors='pt').to(device)

output = model.generate(
    input_ids,
    max_length=20,
    num_return_sequences=1,
    do_sample=True,
    top_k=8,
    top_p=0.95,
    temperature=0.5,
    repetition_penalty=1.2
)

decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)

torch.save(model, 'SmallDisease_LM.pt')



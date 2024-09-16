import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import re

#importing the dataset the bot is trained on
dataset = "human_chat.txt"

with open(dataset, 'r', encoding='utf-8') as f:
    data = f.read()

conversations = re.split(r'(?i)(Human \d+: Hi[!.]*)', data)

conversations_2 = []
for i in range(1, len(conversations), 2):
    conv = conversations[i] + conversations[i+1]
    conversations_2.append(conv.strip())

conversations_3 = []
for conversation in conversations_2:
    conversations_3.append(conversation.split('\n'))

#make final_conversations, a list of lists where each list is a complete conversation.
final_conversations = []
for conversation in conversations_3:
    lines = []
    #remove the "Human 1 / Human 2" which comes before the dialogue
    for line in conversation:
        final_line = re.sub(r'(?i)(Human \d+: )', '', line)
        lines.append(final_line)
    final_conversations.append(lines)

input_texts = []
target_texts = []

# flatenning conversations
for conv in final_conversations:
    for i in range(1, len(conv), 2):
        input_texts.append(conv[i-1])
        target_texts.append(conv[i])

# print(final_conversations)
# print(input_texts)
# print()
# print(target_texts)

# create a Tokenizer and fit on all texts
tokenizer = Tokenizer(filters='', lower=True, oov_token='<OOV>')
tokenizer.fit_on_texts(input_texts + target_texts)

# Convert text to sequences
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

# Pad sequences to ensure uniform input length
max_input_length = max(len(seq) for seq in input_sequences)
max_target_length = max(len(seq) for seq in target_sequences)

input_sequences = pad_sequences(input_sequences, maxlen=max_input_length, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_target_length, padding='post')

# convert target sequences to one-hot encoding for categorical prediction
vocab_size = len(tokenizer.word_index) + 1

encoder_input_data = input_sequences
encoder_input_data = tf.keras.utils.to_categorical(input_sequences, num_classes=vocab_size)
# target_sequences = tf.keras.utils.to_categorical(target_sequences, num_classes=vocab_size) 

# preparing data for seq2seq model
decoder_input_data = target_sequences[:, :-1]
decoder_target_data = target_sequences[:, 1:]
decoder_input_data = tf.keras.utils.to_categorical(decoder_input_data, num_classes=vocab_size)
decoder_target_data = tf.keras.utils.to_categorical(decoder_target_data, num_classes=vocab_size)

# additional variables for model training
num_encoder_tokens = vocab_size # number of unique tokens for the encoder
num_decoder_tokens = vocab_size # number of unique tokens for the decoder
max_encoder_seq_length = max_input_length
max_decoder_seq_length = max_target_length

# print(f'Encoder Input Data Shape: {encoder_input_data.shape}')
# print(f'Adjusted Decoder Input Data Shape: {decoder_input_data.shape}')
# print(f'Adjusted Decoder Target Data Shape: {decoder_target_data.shape}')
print(f'Vocab size: {vocab_size}')
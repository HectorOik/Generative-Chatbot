import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing_functions import(
    pad_or_truncate_conversations,
    raw_data_to_clean_conversation_pairs
)

#importing the dataset the bot is trained on
dataset = "human_chat.txt"
# dataset = "test_dataset.txt"

with open(dataset, 'r', encoding='utf-8') as f:
    data = f.read()

conversations = raw_data_to_clean_conversation_pairs(data)

#

pairs = []
for conversation in conversations:
    for i in range(len(conversation) - 1):
        input_text = conversation[i][1]
        target_text = conversation[i+1][0]
        pairs.append((input_text, target_text))

flattened_inputs, flattened_targets = zip(*pairs)

# print(flattened_inputs[:5])
# print()
# print(flattened_targets[:5])


tokenizer = Tokenizer(filters='', lower=True, oov_token='<OOV>')
tokenizer.fit_on_texts(flattened_inputs + flattened_targets)

input_sequences = tokenizer.texts_to_sequences(flattened_inputs)
target_sequences = tokenizer.texts_to_sequences(flattened_targets)

MAX_SEQ_LENGTH = max(max(len(seq) for seq in input_sequences), max(len(seq) for seq in target_sequences))

padded_inputs = pad_sequences(input_sequences, maxlen=MAX_SEQ_LENGTH, padding='post')
padded_targets = pad_sequences(target_sequences, maxlen=MAX_SEQ_LENGTH, padding='post')

inputs = np.array(padded_inputs)
targets = np.array(padded_targets)

print(inputs.shape, targets.shape)

print(f'Tokenizer word index {tokenizer.word_index}')
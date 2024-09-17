import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Masking, Embedding

from preprocessing import (
    tokenizer, inputs, targets, MAX_SEQ_LENGTH
    )

VOCAB_SIZE = len(tokenizer.word_index) + 1
EMBEDDING_DIM = 100

model = tf.keras.Sequential([
    Masking(mask_value=0, input_shape=(MAX_SEQ_LENGTH,)),
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=inputs.shape[1]),
    LSTM(128, return_sequences=True),
    Dense(VOCAB_SIZE, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

epochs=10
batch_size=32

history = model.fit(inputs, targets, epochs=epochs, validation_split=0.2, batch_size=batch_size)

test_loss, test_acc = model.evaluate(inputs, targets)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')

# # Hyperparameters
# dimensionality = 128
# batch_size = 16
# epochs = 5

# # Encoder
# encoder_inputs = Input(shape=(None, num_encoder_tokens))
# # print(f'Encoder Inputs: {encoder_inputs.shape}')
# encoder_lstm = LSTM(dimensionality, return_state=True)
# encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
# encoder_states = [state_hidden, state_cell] # encoder stastes to pass into the decoder

# # Decoder
# decoder_inputs = Input(shape=(None, num_decoder_tokens))
# # print(f'Decoder Inputs: {decoder_inputs.shape}')
# decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
# decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
# # print(f'Decoder Outputs: {decoder_outputs.shape}')
# decoder_dense = Dense(num_decoder_tokens, activation='softmax')
# decoder_outputs = decoder_dense(decoder_outputs) # final softmax output

# # Define the model
# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# # Compile the model
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# # Print model summary
# model.summary()

# # training the model
# model.fit(
#     [encoder_input_data, decoder_input_data],
#     decoder_target_data,
#     epochs=epochs, 
#     batch_size=batch_size, 
#     validation_split=0.2
#     )

model.save('chatbot_seq2seq_model.h5')
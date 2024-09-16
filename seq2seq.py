from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

from preprocessing import (
    encoder_input_data, decoder_input_data, decoder_target_data,
    num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length,
    max_decoder_seq_length
    )

# Hyperparameters
dimensionality = 256
batch_size = 64
epochs = 100

# Encoder
encoder_inputs = Input(shape=(None, num_encoder_tokens))
# print(f'Encoder Inputs: {encoder_inputs.shape}')
encoder_lstm = LSTM(dimensionality, return_state=True)
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
encoder_states = [state_hidden, state_cell] # encoder stastes to pass into the decoder

# Decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# print(f'Decoder Inputs: {decoder_inputs.shape}')
decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
# print(f'Decoder Outputs: {decoder_outputs.shape}')
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs) # final softmax output

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
# model.summary()

# training the model
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    epochs=epochs, 
    batch_size=batch_size, 
    validation_split=0.2
    )
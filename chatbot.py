import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer

from preprocessing import(
    tokenizer, MAX_SEQ_LENGTH
)
# from seq2seq import encoder_lstm, decoder_lstm

class Chatbot:

    negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")
    exit_commands = ("quit", "pause", "exit", "bye", "goodbye", "later", "stop")

    def __init__(self):
        # load model trained in seq2seq.py
        self.model = tf.keras.models.load_model('chatbot_seq2seq_model.h5')

        # be careful because the tokenizer has also been fit in preprocessing
        # self.tokenizer = Tokenizer(filters='', lower=True, oov_token='<OOV>')
        self.tokenizer = tokenizer
        self.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH

    def start_chat(self):
        user_response = input("Hi! I am a chatbot trained on human dialogue. Would you like to chat with me?\n")

        if user_response in self.negative_responses:
            print("Ok, have a great day!")
            return
        
        self.chat(user_response)

    def chat(self, reply):
        while not self.make_exit(reply):
            reply = input(self.generate_response(reply))

    def generate_response(self, user_input):
        preprocessed_input = self.preprocess_input(user_input)
        # sequence = self.generate_response(preprocessed_input)
        # return 'END OF FUNCTION'
        prediction_probs = self.model.predict(preprocessed_input)
        print(prediction_probs[0, :, :10])
        prediction_probs = prediction_probs / np.sum(prediction_probs, axis=-1, keepdims=True)
        print(f'Prediction Probabilities: {prediction_probs}')
        response = self.decode_prediction(prediction_probs)
        return response

        # # print(f'Input seqs: {input_seq.shape}')

        # decoder_input_seq = np.zeros((1, self.max_target_length, self.vocab_size))
        # decoder_input_seq[0, 0, self.tokenizer.word_index.get('<start>', 0)] = 1

        # # states_value = self.model.layers[2].get_weights()[:2]

        # decoded_sentence = ''
        # stop_condition = False
        # decoder_token_index = 0

        # while not stop_condition:
        #     output_tokens = self.model.predict([input_seq, decoder_input_seq])

        #     print(f'Output tokens shape: {output_tokens.shape}')
        #     print(f'Output tokens: {output_tokens}')

        #     # Sample a token
        #     sampled_token_index = np.argmax(output_tokens[0, decoder_token_index, :])
        #     sampled_word = tokenizer.index_word.get(sampled_token_index, '')

        #     print(f'Sampled token index: {sampled_token_index}')
        #     print(f'Sampled word: {sampled_word}')

        #     # Exit condition: either hit max length or find <end>
        #     if sampled_token_index == 0 or len(decoded_sentence) > self.max_target_length:
        #         # stop_condition = True
        #         stop_condition = True
        #     else:
        #         decoded_sentence += ' ' + sampled_word

        #         # Update the target sequence and states
        #         target_seq = np.zeros((1, self.max_target_length, self.vocab_size))
        #         target_seq[0, i, sampled_token_index] = 1
        #         # states_value = [h, c]
        #         if decoder_token_index >= self.max_target_length:
        #             stop_condition = True

        # return decoded_sentence.strip() + '\n'
    
    def preprocess_input(self, user_input):
        if isinstance(user_input, str):
            print(f"Original Input: {user_input}")
            sequences = self.tokenizer.texts_to_sequences([user_input])
            print(f"Sequences: {sequences}")
            if all(len(seq) == 0 or all(token == 0 for token in seq) for seq in sequences):
                print("Warning: The input text does not match any known tokens.")
            padded_sequence = pad_sequences(sequences, maxlen=self.MAX_SEQ_LENGTH, padding='post')
            print(f'Padded Sequence: {padded_sequence}')
            return padded_sequence
        else:
            raise ValueError("Input must be a string")
        # sequences = self.tokenizer.texts_to_sequences([user_input])
        # padded_sequences = pad_sequences(sequences, maxlen=self.MAX_SEQ_LENGTH, padding='post')
        # return padded_sequences
    
    # def generate_sequence(self, input_sequence):
    #     sequence = input_sequence
    #     print(sequence)


    
    def decode_prediction(self, prediction_probs):
        print(prediction_probs.shape)
        predicted_token_ids = np.argmax(prediction_probs, axis=-1)
        print(f'Predicted Token IDs: {predicted_token_ids}')
        predicted_words = tokenizer.sequences_to_texts(predicted_token_ids)
        # print(f'Predicted Words: {predicted_words}')
        return predicted_words[0]

    def make_exit(self, reply):
        for exit_command in self.exit_commands:
            if exit_command in reply:
                print("Ok, have a great day!")
                return True
            
        return False
    
chatbot = Chatbot()
chatbot.start_chat()

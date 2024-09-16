import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing import tokenizer, max_input_length, max_target_length
from seq2seq import encoder_lstm, decoder_lstm

class Chatbot:

    negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")

    exit_commands = ("quit", "pause", "exit", "bye", "goodbye", "later", "stop")

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
        # Tokenize and pad input
        input_seq = tokenizer.texts_to_sequences([user_input])
        input_seq = pad_sequences(input_seq, maxlen=max_input_length, padding='post')

        # Encode Input
        states_value = encoder_lstm.predict(input_seq)

        # Generate first token
        target_seq = np.zeros((1, max_target_length))
        target_seq[0, 0] = tokenizer.word_index.get('<start>', 0)

        stop_condition = False
        decoded_sentence = ''
        
        while not stop_condition:
            output_tokens, h, c = decoder_lstm.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = tokenizer.index_word.get(sampled_token_index, '')

            # Exit condition: either hit max length or find <end>
            if sampled_word == '<end>' or len(decoded_sentence) > max_target_length:
                stop_condition = True
            else:
                decoded_sentence += ' ' + sampled_word

            # Update the target sequence and states
            target_seq = np.zeros((1, max_target_length))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]

        return decoded_sentence.strip() + '\n'

    def make_exit(self, reply):
        for exit_command in self.exit_commands:
            if exit_command in reply:
                print("Ok, have a great day!")
                return True
            
        return False
    
chatbot = Chatbot()
chatbot.start_chat()

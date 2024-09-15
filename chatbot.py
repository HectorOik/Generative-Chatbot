class Chatbot:

    negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")

    exit_commands = ("quit", "pause", "exit", "bye", "goodbye", "later", "stop")

    def start_chat(self):
        user_response = input("Hi! I am a chatbot trained on human dialogue. Would you like to chat with me? ")

        if user_response in self.negative_responses:
            print("Ok, have a great day!")
            return
        
        self.chat(user_response)

    def chat(self, reply):
        while not self.make_exit(reply):
            reply = input(self.generate_response(reply))

    def generate_response(self, user_input):
        return "Cool\n"
    
    def make_exit(self, reply):
        for exit_command in self.exit_commands:
            if exit_command in reply:
                print("Ok, have a great day!")
                return True
            
        return False
    
chatbot = Chatbot()
chatbot.start_chat()

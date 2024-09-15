import re

#importing the dataset the bot is trained on
dataset = "test_dataset.txt"

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

print(final_conversations)
import re

# used to pad / truncate conversations and achieve unifrom conversation length
def pad_or_truncate_conversations(conversations, max_length, pad_token):
    padded_conversations = []

    for convo in conversations:
        if len(convo) > max_length:
            padded_conversations.append(convo[:max_length])
        else:
            padding = [pad_token] * (max_length - len(convo))
            padded_conversations.append(convo + padding)

    return padded_conversations

def raw_data_to_clean_conversation_pairs(data):
    conversations = split_data_to_conversations(data)
    final_conversations = conversations_to_conversation_pairs(conversations)
    return final_conversations

def split_data_to_conversations(data):
    raw_conversations = re.findall(r'(?i)(Human 1: Hi[!.]*)(.*?)(?=(Human 1: Hi|$))', data, re.DOTALL)

    raw_conversations_2 = []
    for hi_part, conversation_part, _ in raw_conversations:
        conversation = (hi_part + conversation_part).strip()
        raw_conversations_2.append(conversation)

    return raw_conversations_2

def conversations_to_conversation_pairs(conversations):
    final_conversations = []

    for convo in conversations:
        convo_lines = convo.splitlines()
        
        convo_lines = [line.strip() for line in convo_lines if line.strip()]

        cleaned_lines = [re.sub(r'Human \d+: ', '', line) for line in convo_lines]
        # print(cleaned_lines)
        pairs = []
        # print()
        for i in range(0, len(cleaned_lines) - 1, 1):
            part_1 = cleaned_lines[i]
            part_2 = cleaned_lines[i+1]
            pairs.append((part_1, part_2))
        
        if pairs:
            final_conversations.append(pairs)

    return final_conversations
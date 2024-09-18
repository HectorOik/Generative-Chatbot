# Generative-Chatbot

## THE MODEL IS NOT FUNCTIONING YET

This is the link to the training set
https://www.kaggle.com/datasets/projjal1/human-conversation-training-data


Make sure the necessary dependencies with tensorflow / keras are installed and available by using the correct Python interpreter in your IDE.
The easiest way to achieve this is to create a vritual environment 
    python -m venv myenv
    myenv/Scripts/activate (#source myenv/bin/activate on non-Windows)
    pip install tensorflow keras

First run the preprocessing.py to preprocess the data
Second, pass the data to the seq2seq model to create a model and train it on the data.
Lastly, use chatbot.py to initiate dialogue.

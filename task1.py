import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F
from typing import List, Dict
from tqdm import tqdm

from utils.utils import set_seed
from task1.core import preprocess_dataset, encode, calculate_loss

import argparse

# Parameters (Do Not Adjust)
n_epochs = 2
learning_rate = 1e-4
model_name = "gpt2"
batch_size = 4

# Check whether the GPU is available or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a GPT2 tokenizer and Add special token <SEP> and <PAD>
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"sep_token": "<SEP>", "pad_token": "<PAD>"})

# Load a pretrained GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)

# Load Dataset
dialogs = [json.loads(line) for line in open("dailydialog.json")]

# Transform and split dataset
batched_dialog = preprocess_dataset(dialogs, 4, tokenizer)

set_seed(42)
    
def selfcheck():
    print("\n\n=============== YOUR BATCH ===============")
    your_answer = f"BATCH 0: \n\
    Instance 0:  {batched_dialog[0][0]} \n\
    Instance 1:  {batched_dialog[0][1]} \n\
    Instance 2:  {batched_dialog[0][2]} \n\
    Instance 3:  {batched_dialog[0][3]}"
    print(your_answer)

    # The output of this cell should be the same as the reference answer:
    print("\n=============== REFERENCE BATCH ===============")
    reference = "BATCH 0: \n\
    Instance 0:  So Dick , how about getting some coffee for tonight ?<SEP>Coffee ? I don ’ t honestly like that kind of stuff .<SEP>Come on , you can at least try a little , besides your cigarette .<SEP>What ’ s wrong with that ? Cigarette is the thing I go crazy for .<SEP>Not for me , Dick .<SEP> \n\
    Instance 1:  The kitchen stinks .<SEP>I'll throw out the garbage .<SEP> \n\
    Instance 2:  Are things still going badly with your houseguest ?<SEP>Getting worse . Now he ’ s eating me out of house and home . I ’ Ve tried talking to him but it all goes in one ear and out the other . He makes himself at home , which is fine . But what really gets me is that yesterday he walked into the living room in the raw and I had company over ! That was the last straw .<SEP>Leo , I really think you ’ re beating around the bush with this guy . I know he used to be your best friend in college , but I really think it ’ s time to lay down the law .<SEP>You ’ re right . Everything is probably going to come to a head tonight . I ’ ll keep you informed .<SEP> \n\
    Instance 3:  Would you mind waiting a while ?<SEP>Well , how long will it be ?<SEP>I'm not sure . But I'll get a table ready as fast as I can .<SEP>OK . We'll wait .<SEP>"
    print(reference)


    print('\n\n------------- PREPROCESS_DATASET FUNCTION -------------')
    if your_answer == reference:
        print("\t\t\tSuccess!!!")
    else:
        print("\t\t\tFailed.")
    print('-------------------------------------------------------')


def encode_dialogues():
    # Encode the Dialogues
    train_data = encode(batched_dialog, tokenizer)
    your_answer = train_data[0]['input_ids'][0]
    reference = torch.tensor([ 2396, 11740,   837,   703,   546,  1972,   617,  6891,   329,  9975,
            5633, 50257,    34,  2364,  1453,  5633,   314,   836,   564,   247,
            256, 12698,   588,   326,  1611,   286,  3404,   764, 50257, 16773,
            319,   837,   345,   460,   379,  1551,  1949,   257,  1310,   837,
            13769,   534, 17779,   764, 50257,  2061,   564,   247,   264,  2642,
            351,   326,  5633, 32616, 14758,   318,   262,  1517,   314,   467,
            7165,   329,   764, 50257,  3673,   329,   502,   837, 11740,   764,
            50257, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258,
            50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258,
            50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258,
            50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258,
            50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258,
            50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258,
            50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258,
            50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258,
            50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258, 50258])


    print('\n =============== YOUR INPUT IDS ===============')
    print(your_answer)

    # The output of this cell should be the same as the reference answer:
    print('\n =============== REFENCE INPUT IDS ===============')
    print(reference)

    # Unindent to Compare original dialog vs decoded dialog
    print('\nORIGINAL DIALOG:')
    for utterance in dialogs[0]['utterances']:
        print(utterance)
    print('\nDECODED BATCHED DIALOG:')
    for token in tokenizer.batch_decode(train_data[0]['input_ids'][0], skip_special_tokens=True):
        if token == '<SEP>':  print();  continue;
        if token == '<PAD>':  continue;
        print(token, end='')
    print()

    print('\n\n-------------- ENCODE FUNCTION --------------')
    if torch.all(your_answer == reference):
        print("\t\t  Success!!!")
    else:
        print("\t\t    Failed.")
    print('---------------------------------------------')

def prediction():
    # Make prediction and Calculate loss
    train_data = encode(batched_dialog, tokenizer)
    batch = train_data[0]
    logits = model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device)).logits

    your_answer = calculate_loss(logits, batch['input_ids']).item()
    reference = 51.139068603515625
    print('\n =============== YOUR LOSS ===============')
    print(your_answer)


    # The output of calculate_loss function should be the same as the reference answer:
    # The reference loss is calculated based on GPT-2 without any fine-tuning
    print('\n =============== REFENCE LOSS ===============')
    print(reference)



    print('\n\n-------------- CALCULATE_LOSS FUNCTION --------------')
    if your_answer == reference or (reference-your_answer < 1e-4 and your_answer-reference < 1e-4):
        print("\t\t      Success!!!")
    else:
        print("\t\t      Failed.")
    print('-----------------------------------------------------')

def test():
    # Test Case
    print("\n\n=============== PREPROCESS_DATASET TEST CASE 1 ===============")
    test_case_batched_dialog = preprocess_dataset(dialogs, 4, tokenizer)
    print()
    print(f"BATCH 0: \n\
    Instance 0:  {test_case_batched_dialog[30][0]} \n\
    Instance 1:  {test_case_batched_dialog[30][1]} \n\
    Instance 2:  {test_case_batched_dialog[30][2]} \n\
    Instance 3:  {test_case_batched_dialog[30][3]}")


    print("\n\n=============== PREPROCESS_DATASET TEST CASE 2 ===============")
    test_case_dialog = []
    batch_size = 4
    test_case_batched_dialog = preprocess_dataset(test_case_dialog, batch_size, tokenizer)
    print()
    print(test_case_batched_dialog)


    print("\n\n=============== PREPROCESS_DATASET TEST CASE 3 ===============")
    test_case_dialog = [
                {'utterances':['Hi', 'Hello', 'How are you ?', 'I am fine thank you . And you ?']},
                {'utterances':['I love NLP .', 'Why ?', 'Reading texts helps me sleep well .']},
                {'utterances':['May I sit here ?', 'Absolutely No .']}
            ]
    batch_size = 2
    test_case_batched_dialog = preprocess_dataset(test_case_dialog, batch_size, tokenizer)
    print()
    print(f"BATCH 0: \n\
    Instance 0:  {test_case_batched_dialog[0][0]} \n\
    Instance 1:  {test_case_batched_dialog[0][1]} \n\
    BATCH 1: \n\
    Instance 0:  {test_case_batched_dialog[1][0]} \n")
    
    print('\n\n=============== ENCODE TESET CASE 1 ===============')
    train_data = encode(batched_dialog, tokenizer)
    print(train_data[30]['input_ids'][0])
    print(train_data[30]['input_ids'][1])
    print(train_data[30]['input_ids'][2])
    print(train_data[30]['input_ids'][3])

    print('\nORIGINAL DIALOG:')
    for utterance in batched_dialog[30]:
        print(utterance)
    print('\nDECODED BATCHED DIALOG:')
    for seq in train_data[30]['input_ids']:
        print(tokenizer.decode(seq, skip_special_tokens=True))
    print()

    print("\n\n=============== ENCODE TEST CASE 2 ===============")
    test_case_dialog = []
    batch_size = 4
    test_case_batched_dialog = preprocess_dataset(test_case_dialog, batch_size, tokenizer)
    test_case_train_data = encode(test_case_batched_dialog, tokenizer)
    print(test_case_train_data)


    print("\n\n=============== ENCODE TEST CASE 3 ===============")
    test_case_dialog = [
                {'utterances':['Hi', 'Hello', 'How are you ?', 'I am fine thank you . And you ?']},
                {'utterances':['I love NLP .', 'Why ?', 'Reading texts helps me sleep well .']},
                {'utterances':['May I sit here ?', 'Absolutely No .']}
            ]
    batch_size = 2
    test_case_batched_dialog = preprocess_dataset(test_case_dialog, batch_size, tokenizer)
    test_case_train_data = encode(test_case_batched_dialog, tokenizer)

    print('\nORIGINAL DIALOG:')
    print(test_case_batched_dialog[0][0])
    print(test_case_batched_dialog[0][1])
    print(test_case_batched_dialog[1][0])


    print('\nDECODED BATCHED DIALOG:')
    for seq in test_case_train_data:
        print(tokenizer.batch_decode(seq['input_ids'], skip_special_tokens=False))
    print()
    
def train():
    # Load Dataset
    dialogs = [json.loads(line) for line in open("dailydialog.json")]

    # Transform and split dataset
    batched_dialog = preprocess_dataset(dialogs, 4, tokenizer)

    # Encode the Dialogues
    train_data = encode(batched_dialog, tokenizer)
    
    # Set Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Train the model
    model.train()
    losses = []
    for epoch in range(n_epochs):
        pbar = tqdm(enumerate(train_data), total=len(train_data), desc=f"Epoch {epoch + 1}")

        for i, batch in pbar:
            optimizer.zero_grad()
            # Prepare input and attention mask
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask, labels=None).logits

            # Calculate loss
            loss = calculate_loss(logits=logits, input_ids=input_ids)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix({'loss': sum(losses) / len(losses)})

        # Save the model checkpoint
        torch.save(model.state_dict(), f"trained_model_checkpoint.pt")

    # Test Case
    print('\n=============== CALCULATE_LOSS TEST CASE ===============')
    for i in range(5):
        idxes = [8, 1838, 3058, 5873, 6000]
        idx = idxes[i]
        print(f'\n=============== CALCULATE_LOSS TEST CASE {i} ===============')
        print(losses[idx])
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type = str, default='all')
    args = parser.parse_args()
    if args.run == 'all':
        selfcheck()
        encode_dialogues()
        prediction()
        test()
    elif args.run == 'selfcheck':
        selfcheck()
    elif args.run == 'encode':
        encode_dialogues()
    elif args.run == 'prediction':
        prediction()
    elif args.run == 'test':
        test()
    elif args.run == 'train':
        train()
    else:
        print("Invalid Argument")
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import List, Dict
from tqdm import tqdm

from utils.utils import set_seed
import argparse

from task2.core import greedy_decoding, beam_search

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"sep_token": "<SEP>", "pad_token": "<PAD>"})
print(tokenizer.encode('<SEP>'))
print(tokenizer.encode('<PAD>'))

# Initialize a model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)

# Load a model checkpoint
state_dict = torch.load("model_checkpoint.pt", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

set_seed(42)
    
def selfcheck():
    # The output of this cell should be as follows:
    # I had a couple of scrambled eggs and bacon.

    input_text = "What did you have for breakfast today?"
    max_length = 20

    print('\n =============== GREEDY_DECODING OUTPUT ===============')
    output_text = greedy_decoding(model, tokenizer, input_text, max_length)
    print(output_text)


    # The output of  this cell should be as the reference answer:
    print('\n =============== REFENCE OUTPUT ===============')
    reference = 'I had a couple of scrambled eggs and bacon .'
    print(reference)


    print('\n\n-------------- GREEDY_DECODING FUNCTION --------------')
    if output_text == reference:
        print("\t\t      Success!!!")
    else:
        print("\t\t      Failed.")
    print('------------------------------------------------------')
    
    # The output of this cell should be as follows:
    # ['I had a couple of scrambled eggs and bacon.']

    input_text = "What did you have for breakfast today?"
    num_beams = 1
    max_length = 20
    length_penalty = 1.0


    print('\n =============== BEAM_SEARCH OUTPUT ===============')
    final_candidate = beam_search(model, tokenizer, input_text, num_beams, max_length, length_penalty)
    print(final_candidate)

    # The output of  this cell should be as the reference answer:
    print('\n =============== REFENCE OUTPUT ===============')
    reference = ['I had a couple of scrambled eggs and bacon .']
    print(reference)

    print('\n\n-------------- BEAM_SEARCH FUNCTION --------------')
    if final_candidate == reference:
        print("\t\t   Success!!!")
    else:
        print("\t\t      Failed.")
    print('--------------------------------------------------')

    # The output of this cell should be as follows:
    # ['I have a couple of scrambled eggs and a cup of coffee.', 'I have a couple of scrambled eggs and bacon.']

    input_text = "What did you have for breakfast today?"
    num_beams = 2
    max_length = 20
    length_penalty = 1.5

    print('\n =============== BEAM_SEARCH OUTPUT ===============')
    final_candidate = beam_search(model, tokenizer, input_text, num_beams, max_length, length_penalty)
    print(final_candidate)

    # The output of  this cell should be as the reference answer:
    print('\n =============== REFENCE OUTPUT ===============')
    reference = ['I have a couple of scrambled eggs and a cup of coffee .', 'I have a couple of scrambled eggs and bacon .']
    print(reference)

    print('\n\n-------------- BEAM_SEARCH FUNCTION --------------')
    if final_candidate == reference:
        print("\t\t   Success!!!")
    else:
        print("\t\t      Failed.")
    print('--------------------------------------------------')
    
    # The output of this cell should be as follows:
    # ['I have a couple of scrambled eggs and bacon.', 'I have a couple of scrambled eggs and a cup of coffee.']

    input_text = "What did you have for breakfast today?"
    num_beams = 2
    max_length = 20
    length_penalty = -1.0

    print('\n =============== BEAM_SEARCH OUTPUT ===============')
    final_candidate = beam_search(model, tokenizer, input_text, num_beams, max_length, length_penalty)
    print(final_candidate)

    # The output of  this cell should be as the reference answer:
    print('\n =============== REFENCE OUTPUT ===============')
    reference = ['I have a couple of scrambled eggs and bacon .', 'I have a couple of scrambled eggs and a cup of coffee .']
    print(reference)

    print('\n\n-------------- BEAM_SEARCH FUNCTION --------------')
    if final_candidate == reference:
        print("\t\t   Success!!!")
    else:
        print("\t\t      Failed.")
    print('--------------------------------------------------')
    
    # The output of this cell should be as follows:
    # ['I had scrambled eggs, bacon, toast and coffee.', 'I had scrambled eggs, bacon and coffee.', 'I had scrambled scrambled eggs, bacon, bacon and coffee.', 'I had scrambled scrambled eggs, bacon, bacon and coffee beans.']

    input_text = "What did you have for breakfast today?"
    num_beams = 4
    max_length = 20
    length_penalty = 1.5

    print('\n =============== BEAM_SEARCH OUTPUT ===============')
    final_candidate = beam_search(model, tokenizer, input_text, num_beams, max_length, length_penalty)
    print(final_candidate)

    # The output of  this cell should be as the reference answer:
    print('\n =============== REFENCE OUTPUT ===============')
    reference = ['I had scrambled eggs , bacon , toast and coffee .', 'I had scrambled scrambled eggs , bacon , bacon and coffee beans .', 'I had scrambled scrambled eggs , bacon , bacon and coffee .', 'I had scrambled eggs , bacon and coffee .']
    print(reference)


    print('\n\n-------------- BEAM_SEARCH FUNCTION --------------')
    if final_candidate == reference:
        print("\t\t   Success!!!")
    else:
        print("\t\t      Failed.")
    print('--------------------------------------------------')


def test():
    # Test Case 1
    input_text = "What are your hobbies?"
    max_length = 20

    print('\n =============== GREEDY_DECODING TEST CASE 1 ===============')
    final_candidate = greedy_decoding(model, tokenizer, input_text, max_length)
    print(final_candidate)

    # Test Case 2
    input_text = "Do you know what the weather will be like today?"
    max_length = 20

    print('\n =============== GREEDY_DECODING TEST CASE 2 ===============')
    final_candidate = greedy_decoding(model, tokenizer, input_text, max_length)
    print(final_candidate)
    
    # Test Case 1
    input_text = "What are your hobbies?"
    num_beams = 2
    max_length = 20
    length_penalty = 1.5

    print('\n =============== BEAM_SEARCH TEST CASE 1 ===============')
    final_candidate = beam_search(model, tokenizer, input_text, num_beams, max_length, length_penalty)
    print(final_candidate)

    # Test Case 2
    input_text = "Do you know what the weather will be like today?"
    num_beams = 4
    max_length = 20
    length_penalty = 1.5

    print('\n =============== BEAM_SEARCH TEST CASE 2 ===============')
    final_candidate = beam_search(model, tokenizer, input_text, num_beams, max_length, length_penalty)
    print(final_candidate)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type = str, default='all')
    args = parser.parse_args()
    if args.run == 'all' or 'selfcheck':
        selfcheck()

    else:
        print("Invalid Argument")
    
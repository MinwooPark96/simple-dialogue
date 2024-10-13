import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import List, Dict
from tqdm import tqdm

from utils.utils import set_seed
import argparse

from task2.core import greedy_decoding, beam_search
from task3.core import calculate_rouge_n

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
    # Set parameters to calculate ROUGE-N
    # The Reference ROUGE Score is ROUGE-2 F1 score
    n_gram_size = 2
    max_length = 20

    # Input and Reference Text
    input_text = "What did you have for breakfast today?"
    reference_text = "I had scramble eggs and coffee."

    # Get model's prediction for each decoding strategy
    greedy_output = greedy_decoding(model, tokenizer, input_text, max_length)

    # Evalute each sentence with ROUGE
    greedy_rouge_n = calculate_rouge_n(greedy_output, reference_text, N=n_gram_size)


    print('Reference Text: \n\t', reference_text)
    print('Greedy Decoding Output: \n\t', greedy_output)

    print('\n=============== GREEDY ROUGE OUTPUT ===============')
    print(greedy_rouge_n)

    # The output of  this cell should be as the reference answer:
    print('=============== GREEDY REFENCE OUTPUT ===============')
    greedy_reference = 0.3076923076923077
    print(greedy_reference)

    print('\n\n-------------- ROUGE FUNCTION --------------')
    if greedy_rouge_n == greedy_reference:
        print("\t\t  Success!!!")
    else:
        print("\t\t   Failed.")
    print('--------------------------------------------')

    # Set parameters to calculate ROUGE-N
    # The Reference ROUGE Score is ROUGE-2 F1 score
    n_gram_size = 2
    num_beams = 2
    max_length = 20
    length_penalty = 1.0

    # Input and Reference Text
    input_text = "What did you have for breakfast today?"
    reference_text = "I had scramble eggs and coffee."

    # Get model's prediction for each decoding strategy
    beam_search_ouput = beam_search(model, tokenizer, input_text, num_beams, max_length, length_penalty)[0]

    # Evalute each sentence with ROUGE
    beam_search_rouge_n = calculate_rouge_n(beam_search_ouput, reference_text, N=n_gram_size)

    print('Reference Text: \n\t', reference_text)
    print('Beam Search Decoding Output: \n\t', beam_search_ouput)

    print('\n=============== BEAM ROUGE OUTPUT ===============')
    print(beam_search_rouge_n)

    # The output of  this cell should be as the reference answer:
    print('=============== BEAM REFENCE OUTPUT ===============')
    beam_reference = 0.12500000000000003
    print(beam_reference)

    print('\n\n-------------- ROUGE FUNCTION --------------')
    if beam_search_rouge_n == beam_reference:
        print("\t\t  Success!!!")
    else:
        print("\t\t   Failed.")
    print('--------------------------------------------')
    
    
def test():
    # Test Case
    n_gram_size = 2

    print('\n=============== ROUGE TEST CASE 1 ===============')
    input_txt = "The quick brown fox jumps over the lazy dog."
    reference_txt = "The quick brown fox jumps over the lazy dog."
    print(calculate_rouge_n(input_txt, reference_txt, N=n_gram_size))

    print('\n=============== ROUGE TEST CASE 2 ===============')
    input_txt = "The quick brown fox."
    reference_txt = "The quick brown fox jumps over the lazy dog."
    print(calculate_rouge_n(input_txt, reference_txt, N=n_gram_size))


    print('\n=============== ROUGE TEST CASE 3 ===============')
    input_txt = "The cat is sleeping."
    reference_txt = "The quick brown fox jumps over the lazy dog."
    print(calculate_rouge_n(input_txt, reference_txt, N=n_gram_size))

    print('\n=============== ROUGE TEST CASE 4 ===============')
    input_txt = ""
    reference_txt = "The quick brown fox jumps over the lazy dog."
    print(calculate_rouge_n(input_txt, reference_txt, N=n_gram_size))

    print('\n=============== ROUGE TEST CASE 5 ===============')
    input_txt = "fox jumps"
    reference_txt = "The quick brown fox jumps over the lazy dog."
    print(calculate_rouge_n(input_txt, reference_txt, N=n_gram_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type = str, default='all')
    args = parser.parse_args()
    if args.run == 'all' :
        selfcheck()
        test()
    elif args.run == 'selfcheck':
        selfcheck()
    elif args.run == 'test':
        test()
    else:
        print("Invalid Argument")
    
import torch
from typing import List, Dict

def preprocess_dataset(dialogs: List[Dict[str, List]],
                       batch_size: int,
                       tokenizer
                       ):
    """
    In this function, your task is to split the entire dataset into smaller batches.
    !!! Important !!! The order of the data should be preserved.

    Here's how to do it step by step:
        1. Transform a Dialogue into a Single String:
            - Take each list of utterances (individual pieces of dialogue) from the dataset.
            - Combine the utterances in each dialogue into a single string, where each utterance is separated by a special token <SEP>.
            Ex)
                [
                    ["Hello", "Hi, how are you?", "Great!"],
                    ['The kitchen stinks .', "I'll throw out the garbage ."],
                    ["I'm exhausted .", "Okay , let's go home ."],
                    ...
                 ]
            --> [
                    "Hello<SEP>Hi, how are you?<SEP>Great!<SEP>",
                    "The kitchen stinks .<SEP>I'll throw out the garbage .<SEP>",
                    "I'm exhausted .<SEP>Okay , let's go home .<SEP>",
                    ...
                ]

        2. Split the Dialogues into Batches:
            - After transforming each dialogue into a single string, split the resulting dataset into batches.
            - The size of each batch should be determined by the variable batch_size.

            Ex) Example of batch size = 2
                [
                    "Hello<SEP>Hi, how are you?<SEP>Great!<SEP>",
                    "The kitchen stinks .<SEP>I'll throw out the garbage .<SEP>",
                    "I'm exhausted .<SEP>Okay , let's go home .<SEP>",
                    ...
                ]
            --> [
                    [
                        "Hello<SEP>Hi, how are you?<SEP>Great!<SEP>",
                        "The kitchen stinks .<SEP>I'll throw out the garbage .<SEP>",
                    ],
                    [
                        "I'm exhausted .<SEP>Okay , let's go home .<SEP>",
                        ...
                    ],
                    ...
                ]


    Args:
        dialogs (List[Dict[List]]): The dataset of dialogues to be split. Each dialogue is a list of utterances.
        batch_size (int): The size of each batch.
        tokenizer: The tokenizer for Language Model.

    Returns:
        list_of_batches: List of batches, where each batch is a list of strings.
    """
    # !!! Important !!! The order of the data should be preserved

    text_dataset = []
    list_of_batches = []
    ##############################################################
    # TODO
    sep = tokenizer.sep_token
    
    for i, line in enumerate(dialogs):
      sen = line['utterances']
      
      processed = ""
      for utterance in sen:
        processed += utterance + sep
      
      text_dataset.append(processed)

    for i in range(0, len(text_dataset), batch_size):
        list_of_batches.append(text_dataset[i:i+batch_size])
    
    ##############################################################

    return list_of_batches

def encode(list_of_batches: List[str], tokenizer:torch.nn.Module):
    """
    In this function, your task is to encode batches of dialogues using the provided tokenizer.

    Here's how to do it step by step:
        1. Encode the Dialogues:
            - Use the tokenizer to convert each batch of dialogues (provided as a list of strings) into tokenized sequences.
            - Apply padding and truncation to ensure that all sequences in a batch have the same length.
            - Return the encoded dialogues as PyTorch tensors.

    Args:
        list_of_batches (list of str): Batched dialog converted to strings.
        tokenizer: The tokenizer for Language Model.

    Returns:
        List of encoded dialogues, where each item is a dictionary containing input tensors.
    """
    encoded_dialogs = []
    ##############################################################
    # TODO
    
    for batch in list_of_batches:
      encoded_batch = tokenizer(
          batch,
          padding = True,             
          truncation = True,          
          return_tensors = 'pt'
      )

      encoded_dialogs.append(encoded_batch)
    ##############################################################

    return encoded_dialogs

def calculate_loss(logits:torch.Tensor, 
                   input_ids:torch.Tensor,
                   device:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    This function calculates the loss between the model's predicted logits and the ground truth labels (input_ids)
    using CrossEntropyLoss, which is commonly used for classification tasks like language modeling.

    Steps:
        1. Set the loss function to CrossEntropyLoss.

        2. Align the logits and input_ids for teacher forcing:
           - Teacher forcing: Providing the model with the correct output tokens during training rather than
             letting it rely solely on its own predictions from previous time steps.
           - The 'logits' tensor contains the predictions for each token in the sequence.
           - The 'input_ids' contain the true token ids. Set the ground truth 'labels' from 'input_ids'.

        3. Compute the loss with proper dimensions to correctly compare the tokens:
           - Reshape the logits and labels so that the 'logits' have the dimensions necessary to compare each token's prediction to the correct token.
           - Compute the average loss across all tokens in the batch by comparing the 'logits' and 'labels'.

    Args:
        logits (torch.Tensor): The model's predicted logits with shape (batch_size, sequence_length, vocab_size).
                               Each element contains the predicted scores for each token in the vocabulary.
        input_ids (torch.Tensor): Ground truth token ids for the sequence with shape (batch_size, sequence_length).
                                  Each element contains the actual token id from the vocabulary.

    Returns:
        torch.Tensor: The computed loss, which is a scalar value representing how far off the model's predictions were
                      from the ground truth.
    """

    loss_fn = None
    labels = None
    loss = None

    ############################################
    # TODO

    loss_fn = torch.nn.CrossEntropyLoss()
    logits = logits[:, :-1, :]
    logits = logits.reshape(-1, logits.shape[2]).to(device) 
    labels = input_ids[:, 1:].reshape(-1).to(device)             
    
    loss = loss_fn(logits, labels)
    ############################################

    return loss
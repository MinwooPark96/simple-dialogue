import torch
import torch.nn.functional as F


def greedy_decoding(model:torch.nn.Module, tokenizer:torch.nn.Module, input_text:str, max_length:int):
    """
    Perform greedy decoding using a trained language model.

    Args:
        model: The pretrained language model.
        tokenizer: The tokenizer associated with the model.
        input_text: A single input text.
        max_length: The maximum length of the generated sequence.

    Returns:
        generated_text (str): The decoded text of the sequence. Only the generated part (excluding the input_text) should be returned.
    """

    sep_token_id = None
    input_ids = None
    output_ids = None
    generated_text = None

    ###################################################
    
    # TODO
    # Encode input_text
    sep_token_id = tokenizer.sep_token_id
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    output_ids = torch.cat([input_ids, torch.tensor(sep_token_id).view(-1, 1).to(model.device)], dim=-1)   # Add sep token at the end.

    # Loop until the sequence reach max_length
    for i in range(max_length):
        # Get the model output for this current step to compute the logits for following decoding step
        logits = model(output_ids).logits

        # Perform greedy decoding: select the appropriate token for this greedy decoding step
        next_token_id = logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)

        # If the generated token is the <EOS> or <SEP> token, break
        if (next_token_id.item() == sep_token_id) or (next_token_id.item() == tokenizer.eos_token_id):
          break

        # Append the predicted token to the output sequence
        output_ids = torch.cat([output_ids, next_token_id], dim=-1)

    # Convert the list of output token IDs to a human-readable text
    generated_text = tokenizer.decode(output_ids[0][len(input_ids[0])+1:], skip_special_tokens=True)  # Decode starting from the first position of the generated tokens.
    ####################################################


    return generated_text


def beam_search(model, tokenizer, input_text, num_beams, max_length, length_penalty):
    """
    Implement beam search according to the following instructions.

    Args:
        model: The pretrained language model.
        tokenizer: The tokenizer associated with the model.
        input_text: The input text.
        num_beams: Number of beams (top sequences) to keep at each step.
        max_length: The maximum length of the generated sequence.
        length_penalty: The length penalty to apply to outputs. length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences. This parameter will not impact the beam search paths, but only influence the choice of sequences in the end towards longer or shorter sequences.

    Returns:
        final_candidates (list): A list of the top k (k=num_beams) completed sentences, where the next token with the highest log probability is <SEP>, sorted in descending order based on scores with the length penalty applied.

    Implementation Notes:
	    1) Calculate the cumulative sum of log probabilities as the score for each sequence. (Use F.log_softmax() to compute log probabilities.)
	    2) If any sequence encounters the next token with the highest log probability being <SEP> token (indicating the sentence has ended), add that sequence to the final_candidates list and exclude it from the beams list.
         ⚠ To maintain k beams, activate the next highest-ranked beam (k+1th beam at that point) to replace the ended beam.
      3) Update the scores of the candidates in the final_candidates list using the formula 'updated_score = score / (output_length ** length_penalty)', and sort them in descending order based on the updated score.

    Recommended Resources for Reference:
        ⚠ The following materials are for reference only. (The generated results may vary.) Please implement according to the instructions above.
        https://huggingface.co/spaces/m-ric/beam_search_visualizer
        https://m.blog.naver.com/PostView.naver?blogId=sooftware&logNo=221809101199&proxyReferer=https:%2F%2Fwww.google.com%2F&trackingCode=external
    """

    final_candidates = []
    # TODO: Implement Here
    ###################################################################
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    sep_token_id = tokenizer.sep_token_id
    sep_token_tensor = torch.tensor([[sep_token_id]], device=model.device)

    input_ids = torch.cat([input_ids, sep_token_tensor], dim=-1)


    # Initialize the beams
    beams = [(input_ids, 0.0)]  # (sequence, score)
    candidates = []

    # Beam search loop
    for _ in range(max_length):
        new_beams = []  # Temporary storage for newly expanded beams

        # Expand each beam
        for seq, score in beams:
            # Get model output (logits) for the current sequence
            logits = model(seq).logits[:, -1, :]  # Get logits for the last token in the sequence (shape: [1, vocab_size])
            log_probs = F.log_softmax(logits, dim=-1)  # Convert logits to log-probabilities

            # Get the top `num_beams` next tokens with highest log probabilities
            top_log_probs, top_ids = log_probs.topk(num_beams*2, dim=-1)

            # For each top token, create a new beam (expand the sequence)
            for i in range(num_beams):
                next_token_id = top_ids[0, i].unsqueeze(0).unsqueeze(0)  # (1, 1) shape
                new_seq = torch.cat([seq, next_token_id], dim=-1)  # Append the new token to the sequence
                new_score = score + top_log_probs[0, i].item()  # Update cumulative score

                if next_token_id == sep_token_id:
                  candidates.append((new_seq, new_score / ((len(new_seq[0])-len(input_ids[0])) ** length_penalty)))
                  print(len(new_seq[0])-len(input_ids[0]))
                else:
                  new_beams.append((new_seq, new_score))

            n = 0
            while len(new_beams) < num_beams:
                next_token_id = top_ids[0, num_beams+n].unsqueeze(0).unsqueeze(0)
                if next_token_id != sep_token_id:
                  new_seq = torch.cat([seq, next_token_id], dim=-1)  # Append the new token to the sequence
                  new_score = score + top_log_probs[0, num_beams+n].item()  # Update cumulative score
                  new_beams.append((new_seq, new_score))
                n += 1

        # Sort beams by cumulative score and keep the top `num_beams`
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:num_beams]
        print(len(beams))

        if len(candidates) >= num_beams:
          break

    # Sort the final candidates by their scores
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    # Decode the top candidates (exclude input text and <SEP> from the output)
    final_candidates = [
        tokenizer.decode(candidate[0][0][len(input_ids[0]):], skip_special_tokens=True)
        for candidate in candidates[:num_beams]
    ]

    ###################################################################

    return final_candidates
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
    sep_token_id = tokenizer.sep_token_id
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    output_ids = torch.cat([input_ids, torch.tensor(sep_token_id).view(-1, 1).to(model.device)], dim=-1)  

    eos_token_id = tokenizer.eos_token_id
    break_cond = {sep_token_id, eos_token_id}
    
    for _ in range(max_length):
        next_id = model(output_ids).logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
        if (next_id.item() in break_cond):
          break

        output_ids = torch.cat([output_ids, next_id], dim=-1)

    generated_text = tokenizer.decode(output_ids[0][len(input_ids[0])+1:], skip_special_tokens=True) 
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
      3) Update the scores of the considerations in the final_candidates list using the formula 'updated_score = score / (output_length ** length_penalty)', and sort them in descending order based on the updated score.

    Recommended Resources for Reference:
        ⚠ The following materials are for reference only. (The generated results may vary.) Please implement according to the instructions above.
        https://huggingface.co/spaces/m-ric/beam_search_visualizer
        https://m.blog.naver.com/PostView.naver?blogId=sooftware&logNo=221809101199&proxyReferer=https:%2F%2Fwww.google.com%2F&trackingCode=external
    """

    final_candidates = []
    # TODO: Implement Here
    ###################################################################
    device = model.device
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    sep_token_id = tokenizer.sep_token_id
    sep_token_tensor = torch.tensor([[sep_token_id]], device = device)
    input_ids = torch.cat([input_ids, sep_token_tensor], dim=-1)
    activate = F.log_softmax
    
    beams = [(input_ids, 0.0)]  
    considerations = []

    for _ in range(max_length):
        temp_beams = []  

        for seq, score in beams:
            logits = model(seq).logits[:, -1, :]  
            logsomax = activate(logits, dim=-1)  

            top_logsomax, top_ids = logsomax.topk(num_beams*2, dim=-1)

            for i in range(num_beams):
                next_token_id = top_ids[0, i].unsqueeze(0).unsqueeze(0)  
                new_seq = torch.cat([seq, next_token_id], dim=-1)  
                new_score = score + top_logsomax[0, i].item()  

                if next_token_id == sep_token_id:
                  considerations.append((new_seq, new_score / ((len(new_seq[0])-len(input_ids[0])) ** length_penalty)))
                else:
                  temp_beams.append((new_seq, new_score))

            n = 0
            while len(temp_beams) < num_beams:
                next_token_id = top_ids[0, num_beams+n].unsqueeze(0).unsqueeze(0)
                if next_token_id != sep_token_id:
                  new_seq = torch.cat([seq, next_token_id], dim=-1)  
                  new_score = score + top_logsomax[0, num_beams+n].item()  
                  temp_beams.append((new_seq, new_score))
                n += 1

        beams = sorted(temp_beams, key=lambda x: x[1], reverse=True)[:num_beams]

        if len(considerations) >= num_beams:
          break

    considerations = sorted(considerations, key=lambda x: x[1], reverse=True)
    final_candidates = [
        tokenizer.decode(candidate[0][0][len(input_ids[0]):], skip_special_tokens=True)
        for candidate in considerations[:num_beams]
    ]

    ###################################################################

    return final_candidates
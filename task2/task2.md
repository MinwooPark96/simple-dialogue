# Task 2: Decode with difference decoding strategy

### Objective:
Your task is to decode the outputs of your trained Language Model using two different decoding strategies (greedy decoding and beam search). This will help you understand how different decoding strategies affect the quality of generated text. \


### Subtasks:
1. **Implement the ```greedy_decoding``` function.**
    - This function should generate the most probable token at each step, iteratively, until the entire sequence is decoded. Greedy decoding selects the highest probability token at each step, producing a straightforward output sequence. Implement this decoding algorithm in ```greedy_decoding``` function.
2. **Implement the ```beam_search``` function.**
    - The function should generate text by selecting the top n sequences (defined by `num_beams`) based on their probability scores. The process continues until the sequence reaches the maximum length (`max_length`) or the top sequence at the step ends with the <SEP> token. Implement this within the provided ```beam_search``` function. 


### Instruction:
1.   Run the cell in the '**Load Pretrained Model**' section to load the pretrained model.
2.   In the '**Greedy Decdoing**' section, review the comments provided for each part and complete the code where `# TODO: Implement Here` is indicated.
3.   In the '**Beam Search Decoding**' section, review the comments provided for each part and complete the code where `# TODO: Implement Here` is indicated.
4. Check if the function you implemented works properly by running the cells in ✅ **Self-Check** section.

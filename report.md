# Report

## Task setup
### k
I chose k=3 because 900x900=810,000 potential examples seems to be a reasonable dataset size. 

### Model
I use a basic causal lm from the Transformers library.

### Tokenizer 
For this task we only need to tokenize digits, operands, and special tokens (bos, eos, pad) so we want to keep tokenization simple. This is why I choose to do character-level tokenization, this way I can keep the vocab small and avoid added complexity from more language suited tokenization strategies like BPE. 

## Data Design


## Training details


## Evaluation Suite


## Results


## Interpretation


## Discussion

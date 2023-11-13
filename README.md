# Attention is all you need

Predecessor: Recurrent neural network
- Sequential processing
- Slow, vanishing/exploding gradients, difficulty in accessing information from distant past

Solution: The tranformer!

## Encoder
### Input embedding
- Transform sentence into tokens (words, ngrams, characters)
- Map tokens into numbers (position in vocabulary, fixed values)
- Embedding: map numbers into vectors (vector size is 512, no fixed values)
- d_model: 512 (embedding size)

### Positional encoding
We want each token to carry some information about its position in the sentence. We want the model to treat words that appear close to each other as "close", and words that are distant as "distant". This positional encoding should represent a pattern that we can learn.

- We take input embedding and **add** position embedding
- Position embedding is fixed, size 512, represents position of token in the sentence, **reused** for every sentence
- PE(pos, 2i) = sin... , and PE(pos, 2i+1) = cos...

Why trigonometric functions? Periodic, pattern, hope model will learn

### Self-attention
Allows the model to relate words to each other.
- Take, e.g., seq = 6, and d_model = 512
- Q, K, V are input (all (6, 512))
- Follow formula:
    - multiply Q by transposed input (K), divide by sqrt(512) -> (6, 6) matrix
    - Softmax makes each row sum to 1
    - Multiply by V -> (6, 512) matrix
- Each row in this matrix captures not only the meaning (given by the embedding) or the position in the sentence (positional encodings), but also each word's interaction with other words.
- Self-attention
    - is **permutation invariant**!
    - does not require parameters
    - we expect values along the diagonal to be the highest
    - if we dont want some positions to interact, we can always set theit values to negative infinity before applying the softmax 

### Multi-Head attention
- Input (seq, d_model)
- Make 4 copies of it
- 3 of which: Q, K, V
- Then, for each we multiply by parameter matrices (W^Q, W^K, W^V, all (d_model, d_model))
- Q',K',V' (seq, d_model)
- Split in smaller matrices
    - by seq dim or d_model dim, in multihead always by d_model dim
    - so, every head will see the full sentence but a smaller part of the embedding of each word
    - d_k = d_model / num_heads (h)
    - we get e.g., Q_1, Q_2, Q_3, Q_4, K_1, K_2, K_3, K_4, V_1, V_2, V_3, V_4
    - we can calculate the attention between these smaller matrices
    - head_i = Attention(QW_i^Q, KW_i^K, VW_i^V), (SEQ, d_v) (d_v = d_k)
    - Concat heads -> (SEQ, h * d_v) = H
    - H multiply by W^O -> (SEQ, d_model) = MH-A
    - So, each head is watching the sentence but different aspect of the embedding of each word
    - Why do we want this? So we have a different representation of the sentence for each head, and we can learn different things from each representation
    
### Add & Norm
layer normalization
- e.g., batch of 3 items
- each item has 512 features
- for each item, get mean and variance
- normalize each item
- all values are in range of 0 - 1
- this we multiply by gamma and add beta (introduce flucaution)
- bach norm: feature-wise normalization
- layer norm: item-wise normalization

## Decoder
### Masked Multi-Head Attention
Goal, make model causal: output at certain position can only depend on the words on the previous positions. Basically, model cant see future. How? We put -infinity before softmax at places where we dont want the model to look.

## Inference and training
### Training
TIME STEP 1
e.g., I love you very much -> Ik hou heel erg veel van jou
<SOS> I love you very much <EOS> -> Encoder input preparing
-> Encoder -> (Seq, d_model)
Input decoder: <SOS> Ik hou heel erg veel van jou
Add padding, embed, add positional encoding
- Decoder output -> (Seq, d_model)
- Linear layer -> (Seq, vocab_size) (logits)
- Softmax -> (Seq, vocab_size)
- target: <EOS>
- loss: cross entropy
TIME STEP 1 end

### Inference
TIME STEP 1
<SOS> I love you very much <EOS> -> Encoder input preparing
-> Encoder -> (Seq, d_model)
Input decoder: <SOS>
Add padding, embed, add positional encoding
- Decoder output -> (Seq, d_model)
- Linear layer -> (Seq, vocab_size) (logits)
- Softmax -> (Seq, vocab_size)
We select a token from the vocab corresponding to the position of the token with the maximum value
e.g., "Ik"
TIME STEP 1 end
TIME STEP 2
No need for recomputing encoder output
Input decoder: <SOS> Ik
and so on...

### Inference strategy
- We selected, at every step, the word with the maximum softmax value. This is greedy, and usually does not perform well.
- A better strategy is to select at each step the top B words and evaluate all the possible next words for each of them and at each step, keeping the top B most probable sequencees. This is called Beam search.


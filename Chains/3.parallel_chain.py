from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel


load_dotenv()

model1= ChatOpenAI()

model2= ChatOpenAI()

prompt1 = PromptTemplate(
    template="generate short and simple notes on {text}",
    input_variables=["text"]
)      

prompt2 = PromptTemplate(
    template='Generate a 5 short question and answer pairs from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='merge the provided notes and Q&A pairs into a single Document \n Notes: {notes} \n Q&A: {qna}',
    input_variables=['notes','qna']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'qna': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """Self - Attention in NLP
Last Updated : 23 Aug, 2025
In Transformer models, self-attention allows the model to look at all words in a sentence at once but it doesn’t naturally understand the order of those words. This is a problem because word order matters in language. To solve this Transformers use positional embeddings extra information added to each word that tells the model where it appears in the sentence. This helps the model understand both the meaning of each word and its position so it can process sentences more effectively.

Attention in NLP
The goal of self attention mechanism is to improve performance of traditional models such as encoder decoder models used in RNNs (Recurrent Neural Networks).
In traditional encoder decoder models input sequence is compressed into a single fixed-length vector which is then used to generate the output.
This works well for short sequences but struggles with long ones because important information can be lost when compressed into a single vector.
To overcome this problem self attention mechanism was introduced.
Encoder Decoder Model
An encoder decoder model is used in machine learning tasks that involve sequences like translating sentences, generating text or creating captions for images. Here's how it works:

Encoder: It takes the input sequence like sentences and processes them. It converts input into a fixed size summary called a latent vector or context vector. This vector holds all the important information from the input sequence.
Decoder: It then uses this summary to generate an output sequence such as a translated sentence. It tries to reconstruct the desired output based on the encoded information.
frame_3053
Encoder Decoder Model
Attention Layer in Transformer
Input Embedding: Input text like a sentences are first converted into embeddings. These are vector representations of words in a continuous space.
Positional Encoding: Since Transformer doesn’t process words in a sequence like RNNs positional encodings are added to the input embeddings and these encode the position of each word in the sentence.
Multi Head Attention: In this multiple attention heads are applied in parallel to process different part of sequences simultaneously. Each head finds the attention scores based on queries (Q), keys (K) and values (V) and adds information from different parts of input.
Add and Norm: This layer helps in residual connections and layer normalization. This helps to avoid vanishing gradient problems and ensures stable training.
Feed Forward: After attention output is passed through a feed forward neural network for further transformation.
Masked Multi Head Attention for the Decoder: This is used in the decoder and ensures that each word can only attend to previous words in the sequence not future ones.
Output Embedding: Finally transformed output is mapped to a final output space and processed by softmax function to generate output probabilities.
Self Attention Mechanism
This mechanism captures long range dependencies by calculating attention between all words in the sequence and helping the model to look at the entire sequence at once. Unlike traditional models that process words one by one it helps the model to find which words are most relevant to each other helpful for tasks like translation or text generation.

Here’s how the self attention mechanism works:

Input Vectors and Weight Matrices: Each encoder input vector is multiplied by three trained weight matrices (
W
(
Q
)
W(Q), 
W
(
K
)
W(K), 
W
(
V
)
W(V)) to generate the key, query and value vectors.
Query Key Interaction: Multiply the query vector of the current input by the key vectors from all other inputs to calculate the attention scores.
Scaling Scores: Attention scores are divided by the square root of the key vector's dimension (
d
k
dk) usually 64 to prevent the values from becoming too large and making calculations unstable.
Softmax Function: Apply the softmax function to the calculated attention scores to normalize them into probabilities.
Weighted Value Vectors: Multiply the softmax scores by the corresponding value vectors.
Summing Weighted Vectors: Sum the weighted value vectors to produce the self attention output for the input.
Above procedure is applied to all the input sequences. Mathematically self attention matrix for input matrices (
Q
,
K
,
V
Q,K,V) is calculated as:

A
t
t
e
n
t
i
o
n
(
Q
,
K
,
V
)
=
s
o
f
t
m
a
x
(
Q
K
T
d
k
)
V
Attention(Q,K,V)=softmax( 
d 
k

 
QK 

 )V

where
Q
,
K
,
V
Q,K,V are the concatenation of query, key and value vectors

Multi Head Attention
In multi headed attention mechanism, multiple attention heads are used in parallel which allows the model to focus on different parts of the input sequence simultaneously. This approach increases model's ability to capture various relationships between words in the sequence.

M
u
l
t
i
H
e
a
d
(
Q
,
K
,
V
)
=
c
o
n
c
a
t
(
h
e
a
d
1
h
e
a
d
2
.
.
.
h
e
a
d
n
)
W
O
MultiHead(Q,K,V)=concat(head 
1
​
 head 
2
​
 ...head 
n
​
 )W 
O
​
 

Here’s a step by step breakdown of how multi headed attention works:

MHA
Multi-headed-attention
Generate Embeddings: For each word in the input sentence it generate its embedding representation.
Create Multiple Attention Heads: Create 
h
h(e.g
h
=
8
h=8) attention heads and each with its own weight matrices 
W
(
Q
)
,
W
(
K
)
,
W
(
V
)
W(Q),W(K),W(V).
Matrix Multiplication: Multiply the input matrix by each of the weight matrices 
W
(
Q
)
,
W
(
K
)
,
W
(
V
)
W(Q),W(K),W(V) for each attention head to produce key, query and value matrices.
Apply Attention: Apply attention mechanism to the key, query and value matrices for each attention head which helps in generating an output matrix from each head.
Concatenate and Transform: Concatenate the output matrices from all attention heads and apply a dot product with weight 
W
O
W 
O
​
  to generate the final output of the multi-headed attention layer.
Use in Transformer Architecture
Encoder Decoder Attention: In this layer queries come from the previous decoder layer while the keys and values come from the encoder’s output. This allows each position in the decoder to focus on all positions in the input sequence.
Encoder Self Attention: This layer receives queries, keys and values from the output of the previous encoder layer. Each position in the encoder looks at all positions from the previous layer to calculate attention scores.
encoderselfattention
Encoder Self-Attention
Decoder Self Attention: Similar to the encoder's self attention but here the queries, keys and values come from the previous decoder layer. Each position can attend to the current and previous positions but future positions are masked to prevent the model from looking ahead when generating the output and this is called masked self attention.
"""

result = chain.invoke({"text": text})
print(result)

chain.get_graph().print_ascii()

    
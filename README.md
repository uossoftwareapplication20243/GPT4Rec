### * 2.1.2. 부분을 참고하여 GPT4Rec 모델을 활용하면 될 것 같습니다!!

# CLLM4Rec: Collaborative LLM for Recommender Systems

These codes are associated with the following paper [[pdf]](https://arxiv.org/abs/2311.01343):
>Collaborative Large Language Model for Recommender Systems  
>**Yaochen Zhu**, Liang Wu, Qi Guo, Liangjie Hong, Jundong Li,   
>The ACM Web Conference (WWW) 2024.
  
## 1. Introduction(for CLLM4Rec)
The proposed CLLM4Rec combines the ID-based paradigm and LLM-based paradigm recommedation system. 

## 2. Structure of Codes

### 2.1. Libary from...
Based on the Hugging Face🤗 [transformer](https://github.com/huggingface/transformers) library.

#### 2.1.1. Tokenizer Class:
**TokenizerWithUserItemIDTokens** breaks down the word sequence into tokens.  
User_i, Item_j를 각각 토큰취급하여 처리

**Demo:**
```
-----Show the encoding process:-----
Hello, user_1! Have you seen item_2?
['Hello', ',', 'user_1', '!', 'ĠHave', 'Ġyou', 'Ġseen', 'item_2', '?']
[15496, 11, 50258, 0, 8192, 345, 1775, 50269, 30]
```
#### 2.1.2. GPT4Rec Base Model Class:
**GPT4RecommendationBaseModel** is the base class.  
This class extends the vocabulary of the original GPT2 with the user/item ID tokens. In our implementation, we randomly initialize the user/item ID embeddings. In the training time, we freeze the token embeddings for the original vocabulary and the transformer weights and only user/item ID embeddings can be updated.

**Demo:**

```
input_ids:
tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])

-----Calculated Masks-----
vocab_mask:
tensor([[1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]])

user_mask:
tensor([[0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]])

item_mask:
tensor([[0, 0, 0],
        [0, 0, 0],
        [1, 1, 1]])

-----Embed Vocabulary Tokens-----
vocab_ids:
tensor([[0, 1, 2],
        [0, 0, 0],
        [0, 0, 0]])

vocab_embeddings:
tensor([[[ 1.4444,  0.0186],
         [-0.3905,  1.5463],
         [-0.2093, -1.3653]],

        [[ 0.0000,  0.0000],
         [ 0.0000,  0.0000],
         [ 0.0000,  0.0000]],

        [[ 0.0000,  0.0000],
         [ 0.0000,  0.0000],
         [ 0.0000,  0.0000]]], grad_fn=<MulBackward0>)

-----Embed User Tokens-----
user_ids:
tensor([[0, 0, 0],
        [0, 1, 2],
        [0, 0, 0]])

user_embeds:
tensor([[[-0.0000,  0.0000],
         [-0.0000,  0.0000],
         [-0.0000,  0.0000]],

        [[-0.1392,  1.1265],
         [-0.7857,  1.4319],
         [ 0.4087, -0.0928]],

        [[-0.0000,  0.0000],
         [-0.0000,  0.0000],
         [-0.0000,  0.0000]]], grad_fn=<MulBackward0>)

-----Embed Item Tokens-----
item_ids:
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 1, 2]])

item_embeds:
tensor([[[-0.0000,  0.0000],
         [-0.0000,  0.0000],
         [-0.0000,  0.0000]],

        [[-0.0000,  0.0000],
         [-0.0000,  0.0000],
         [-0.0000,  0.0000]],

        [[-0.3141,  0.6641],
         [-1.4622, -0.5424],
         [ 0.6969, -0.6390]]], grad_fn=<MulBackward0>)

-----The Whole Embeddings-----
input_embeddings:
tensor([[[ 1.4444,  0.0186],
         [-0.3905,  1.5463],
         [-0.2093, -1.3653]],

        [[-0.1392,  1.1265],
         [-0.7857,  1.4319],
         [ 0.4087, -0.0928]],

        [[-0.3141,  0.6641],
         [-1.4622, -0.5424],
         [ 0.6969, -0.6390]]], grad_fn=<AddBackward0>)
```

## 만약 저 논문과 같이 Collborative GPT4Rec으로 테스트하길 원하신다면
[https://github.com/yaochenzhu/LLM4Rec] 를 참고하시면 됩니다!

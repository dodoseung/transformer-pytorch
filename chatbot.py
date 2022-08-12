from transformer import Transformer

transformer = Transformer(num_encoder_layer=6,
                          num_decoder_layer=6,
                          d_model=512,
                          num_heads=8,
                          d_ff=2048,
                          vocab_size=10000)

from torchtext.data import Field
import spacy 

SRC = Field(tokenize = 'spacy',
            tokenizer_language='en',
            init_token = '<sos>',
            pad_token = '<pad>',
            eos_token = '<eos>',
            unk_token = '<unk>',
            lower=True
            )

TRG = Field(tokenize = 'spacy',
            tokenizer_language='de',
            init_token = '<sos>',
            pad_token = '<pad>',
            eos_token = '<eos>',
            unk_token = '<unk>',
            lower=True
            )  
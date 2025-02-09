import pickle

import torch
import torchtext.vocab

torchtext.disable_torchtext_deprecation_warning()

from typing import List
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer

import itertools

# TextTensorizer provides utilities for building tensors from text,
# and provides an easy way to create torchtext vocab objects.
class TextTensorizer:

    tokenizer = get_tokenizer("spacy", "en_core_web_sm")

    # Accepts a torchtext vocabulary object for token lookups, and a string 
    # to be parsed to tensor (using vocab)
    @classmethod
    def text_to_tensor(cls, lang_vocab,
                       doc: str | List[str], 
                       max_tokens: int=None,
                       reverse_tokens: bool=False,
                       tokenize: bool=True,
                       remove_unknown_tokens: bool=True) -> torch.Tensor: 
        
        tokens = doc if not tokenize else cls.tokenizer(doc)
        
        if max_tokens is not None:
            tokens = tokens[ : max_tokens]

        # Optionally reverse input sequence (trusting the paper)
        if reverse_tokens:
            tokens.reverse()
        
        text_tensor = [lang_vocab[token] for token in tokens]
        if remove_unknown_tokens:
            default_idx = lang_vocab.get_default_index()
            tokens = [i for i in text_tensor if i != default_idx]

        text_tensor = torch.tensor(text_tensor, dtype=torch.long)

        return text_tensor
    
    @classmethod
    def build_vocab(cls, corpus: List[str], 
                    specials: List[str],
                    default_index_token: str ="<UNK>",  
                    min_freq: int=5,
                    itos_save_path: str = "all_vocab_itos.pickle", 
                    stoi_save_path: str = "all_vocab_stoi.pickle",
                    save:bool=True):
        
        # Apply tokenizer to each entry in corpus
        tokenized_entries = iter(map(cls.tokenizer, corpus))
        tokenized_entries = list(itertools.chain.from_iterable(tokenized_entries))
        
        # Get token frequencies then sort lo-to-hi
        counter = Counter(tokenized_entries)
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        ordered_dict = OrderedDict(sorted_by_freq_tuples)    
        vocab_object = vocab(ordered_dict, min_freq=min_freq, specials=specials)

        default_idx = vocab_object[default_index_token]

        # Sets the default index for lookups (redundant...?)
        vocab_object.set_default_index(default_idx)
        
        # Return itos and stoi mappings separately so colab doesn't yell at us
        # for using torchtext
        vocab_itos = vocab_object.get_itos()
        vocab_stoi = vocab_object.get_stoi()
        
        if save and (itos_save_path and stoi_save_path):
            cls.save_vocab(vocab_itos, itos_save_path)
            cls.save_vocab(vocab_stoi, stoi_save_path)
        
        return vocab_itos, vocab_stoi
    
    @classmethod
    def save_vocab(cls, lang_vocab, filename: str):

        with open(filename, "wb+") as f:
            pickle.dump(lang_vocab, f)
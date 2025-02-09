import requests
import threading

import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader

from datasets import load_dataset

from PIL import Image
import numpy as np
import itertools
import pickle

from DatasetHelper.dataset_helper import run_threads

from typing import List
import math

from text_tensor_builder import TextTensorBuilder

# Creates batches for torch's DataLoader class
def collate_fn(data_batch):

    caption_batch, image_batch = [], []
    for caption, image in data_batch:

        # Input indices need to be long for embeddings
        caption_batch.append(torch.tensor(caption, dtype=torch.long))

        # Scale RGB values b/w 0-1 by dividing by 255
        image_batch.append(torch.tensor(image, dtype=torch.float64) / 255.) 
    # I probably won't fix this
    try:
        caption_batch = pad_sequence(caption_batch, padding_value=0, batch_first=True)
        image_batch = pad_sequence(image_batch, padding_value=0, batch_first=True)
    except:
        return None, None
    return caption_batch, image_batch

def download_images_coroutine(dataset_batch) -> None:
    
    global processed_data

    # Attempt to download image from each URL, adding caption + image to 
    # the output if success
    for caption, url in dataset_batch:
        try:
            img = np.array(Image.open(
                requests.get(url, timeout=0.5, stream=True).raw).resize((100,100)))
        except:
            continue
        processed_data.append((caption, img))   


if __name__ == "__main__":

    ds = load_dataset("google-research-datasets/conceptual_captions", "unlabeled")
    # Create language vocab with flattened all  _data
    max_seq_len = 10
    # Training data
    train_captions = ds["train"]["caption"][:20000]
    train_image_urls = ds["train"]["image_url"][:20000]

    # Validation data
    valid_captions = ds["validation"]["caption"][:1000]
    valid_image_urls = ds["validation"]["image_url"][:1000]

    # Convert all captions (both train & valid) to lowercase
    train_captions = [caption.lower() for caption in train_captions]
    valid_captions = [caption.lower() for caption in valid_captions]


    
    # Pad out captions to be of length 10 (if they aren't already)

    # Create vocabulary using concattenated train and valid captions
    vocab_itos, vocab_stoi = TextTensorBuilder.build_vocab(
        corpus=train_captions + valid_captions, 
        specials=["<PAD>", "<UNK>"], 
        default_index_token="<UNK>", min_freq=0, 
        save=True)
    
    train_vocab_captions = [[vocab_stoi[tok] if tok in vocab_stoi else 1
                       for tok in caption.lower().split()[:max_seq_len]] 
                       for caption in train_captions]
    valid_vocab_captions = [[vocab_stoi[tok] if tok in vocab_stoi else 1
                       for tok in caption.lower().split()[:max_seq_len]] 
                       for caption in valid_captions]
    
    for i, caption in enumerate(train_vocab_captions):
        if len(caption) < max_seq_len:
            train_vocab_captions[i] = caption + [0] * (max_seq_len - len(caption))

    for i, caption in enumerate(valid_vocab_captions):
        if len(caption) < max_seq_len:
            valid_vocab_captions[i] = caption + [0] * (max_seq_len - len(caption))

    train_queries=list(zip(train_vocab_captions, train_image_urls))
    valid_queries=list(zip(valid_vocab_captions, valid_image_urls))
    
    
    processed_data = list()

    run_threads(num_threads=12, dataset=train_queries, coroutine=download_images_coroutine)
    

    train_dl = DataLoader(processed_data,
                          batch_size=64,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=collate_fn)
    with open("train_dl1.pickle","wb+") as f:
        pickle.dump(train_dl, f) 

    valid_dl = DataLoader(processed_data,
                          batch_size=64,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=collate_fn)

    # Write dataloaders to files
    with open("valid_dl1.pickle","wb+") as f:
        pickle.dump(valid_dl, f)
  #  with open("valid_dl.pickle","wb+") as f:
   #     pickle.dump(valid_dl,f)

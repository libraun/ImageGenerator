import requests

import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader

from datasets import load_dataset

from PIL import Image
import numpy as np
import pickle
import math

from modules.DatasetHelper.dataset_helper import run_threads
from modules.TextTensorizer.text_tensorizer import TextTensorizer

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

def pad_captions(captions, max_seq_length: int=10):
    for i, caption in enumerate(captions):
        if len(caption) < max_seq_length:
            captions[i] = caption + [0] * (max_seq_length - len(caption))
    return captions


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

    MAX_SEQ_LENGTH = 10
    TRAIN_SPLIT_RATIO = 0.9

    OUTPUT_DIR_PREFIX = "./storage/"

    TRAIN_OUTPUT_PATH = f"{OUTPUT_DIR_PREFIX}train_dataloader.pickle"
    VALID_OUTPUT_PATH = f"{OUTPUT_DIR_PREFIX}valid_dataloader.pickle"

    ds = load_dataset("google-research-datasets/conceptual_captions", "unlabeled")

    # Global buffer that holds all processed (image, caption) tuples
    processed_data = list()

    # Training data
    train_captions = ds["train"]["caption"][:20000]
    train_image_urls = ds["train"]["image_url"][:20000]

    # Validation data
    valid_captions = ds["validation"]["caption"][:1000]
    valid_image_urls = ds["validation"]["image_url"][:1000]

    # Convert all captions (both train & valid) to lowercase
    train_captions = [caption.lower() for caption in train_captions]
    valid_captions = [caption.lower() for caption in valid_captions]

    # Create vocabulary using concatenated train and valid captions
    vocab_itos, vocab_stoi = TextTensorizer.build_vocab(
        corpus=train_captions + valid_captions, 
        specials=["<PAD>", "<UNK>"], 
        default_token="<UNK>", 
        save_directory=OUTPUT_DIR_PREFIX)
    
    train_vocab_captions = [[vocab_stoi[tok] if tok in vocab_stoi else 1
                       for tok in caption.lower().split()[:MAX_SEQ_LENGTH]] 
                       for caption in train_captions]
    valid_vocab_captions = [[vocab_stoi[tok] if tok in vocab_stoi else 1
                       for tok in caption.lower().split()[:MAX_SEQ_LENGTH]] 
                       for caption in valid_captions]
    
    
    # Pad out captions to be of length 10 (if they aren't already)
    train_vocab_captions = pad_captions(
        train_vocab_captions, max_seq_length=MAX_SEQ_LENGTH)
    valid_vocab_captions = pad_captions(
        valid_vocab_captions, max_seq_length=MAX_SEQ_LENGTH)

    train_queries=list(zip(train_vocab_captions, train_image_urls))
    valid_queries=list(zip(valid_vocab_captions, valid_image_urls))
    train_end_idx = math.floor(len(train_vocab_captions) * TRAIN_SPLIT_RATIO)

    # Use threads to download image data from training set URLs
    run_threads(
        num_threads=12, 
        coroutine=download_images_coroutine, 
        dataset=train_queries )
    train_dl = DataLoader(processed_data,
                          batch_size=64,
                          shuffle=True,
                          collate_fn=collate_fn)
    
    # Clear global buffer before processing validation dataset
    processed_data.clear()

    # Collect data from validation set
    run_threads(
        num_threads=12, 
        coroutine=download_images_coroutine, 
        dataset=valid_queries )

    valid_dl = DataLoader(processed_data,
                          batch_size=64,
                          shuffle=True,
                          collate_fn=collate_fn)

    # Write dataloaders to files
    with open(TRAIN_OUTPUT_PATH,"wb+") as f:
        pickle.dump(train_dl, f) 

    with open(VALID_OUTPUT_PATH,"wb+") as f:
        pickle.dump(valid_dl, f)

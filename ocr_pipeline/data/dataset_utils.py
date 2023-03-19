import os

import pandas as pd
from data.extraction_methods import *
from tqdm import tqdm
import statistics


def get_padded_labels(idxs, words, labels, lengths):
    batch_labels = []
    batch_lengths = []
    batch_words = []
    maxlen = 0

    for idx in idxs:
        batch_labels.append(labels[idx])
        batch_words.append(words[idx])
        batch_lengths.append(lengths[idx])
        maxlen = max(lengths[idx], maxlen)

    for i in range(len(batch_labels)):
        # Pad with 1 (grapheme_dict['<pad>'])
        batch_labels[i] = batch_labels[i] + [1]*(maxlen-len(batch_labels[i]))

    return batch_words, batch_labels, batch_lengths



def decode_prediction(preds, inv_grapheme_dict):
    decoded_string = []
    decoded_label = []

    #print(preds)

    # if(not gt):
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != 1 and (not (i > 0 and preds[i - 1] == preds[i])):
            decoded_string.append(inv_grapheme_dict.get(preds[i]))
            decoded_label.append(preds[i])

    # Decode ground truth
    # else:
    #     for i in range(len(preds)):
    #         if preds[i] != 0 and preds[i] != 1:
    #             decoded_string.append(inv_grapheme_dict.get(preds[i]))
    #             decoded_label.append(preds[i])
                
    ##################Cases that hold None types####################
    # #print(decoded_label)
    # if(len(decoded_label) != 0):
    #     return decoded_label, ''.join(decoded_string)
    # else:
    #     return None

    #print(decoded_label)

    return decoded_label, ''.join(decoded_string)


def encode_mjsynth_data(root_path, labels_file, inv_grapheme_dict=None):
    
    if inv_grapheme_dict is None:
        grapheme_dict = {}
        # 0 is reserved for ctc blank
        grapheme_dict['<pad>'] = 1  # pad
        grapheme_dict['<unk>'] = 2  # OOV
        count = 3 # valid graphemes start from 3
    else:
        grapheme_dict = {v: k for k, v in inv_grapheme_dict.items()}
    
    img_paths = []
    words = []
    labels = []
    lengths = []
    
    with open(os.path.join(root_path, labels_file), 'r') as file:
        mappings = [line.strip() for line in file]

    print("\nPreprocessing mjsynth data:")
    for i, mapping in enumerate(tqdm(mappings)):
        filepath = mapping.split(' ', 1)[0]
        # Get labels from labels dict
        curr_word = filepath.rsplit('/', 1)[-1].split('_')[1].lower()        
        # curr_word = normalize_word(curr_word)
        # print(curr_word)
        img_paths.append(os.path.join(root_path, filepath[2:]))
        curr_label = []
        words.append(curr_word)
        
        graphemes = list(curr_word)
        
        for grapheme in graphemes:
            if grapheme not in grapheme_dict:
                if inv_grapheme_dict is None:
                    grapheme_dict[grapheme] = count
                    curr_label.append(count)
                    count += 1
                else:
                    curr_label.append(grapheme_dict['<unk>']) 
            else:
                curr_label.append(grapheme_dict[grapheme])
        lengths.append(len(curr_label))
        labels.append(curr_label)
    
    if inv_grapheme_dict is None:
        inv_grapheme_dict = {v: k for k, v in grapheme_dict.items()}
        
    return inv_grapheme_dict, img_paths, words, labels, lengths


def encode_synth_data(labels_file, inv_grapheme_dict=None, representation='ads'):
    
    if inv_grapheme_dict is None:
        grapheme_dict = {}
        # 0 is reserved for ctc blank
        grapheme_dict['<pad>'] = 1  # pad
        grapheme_dict['<unk>'] = 2  # OOV
        count = 3 # valid graphemes start from 3
    else:
        grapheme_dict = {v: k for k, v in inv_grapheme_dict.items()}
    
    labels = []
    words = []
    lengths = []
    
    with open(labels_file, 'r') as file:
        mappings = [line.strip() for line in file]
    
    labels_dict = {filename: label for filename, label in (mapping.split(' ', 1) for mapping in mappings)}
    filenames = sorted(labels_dict.keys(), key=lambda x: int(x.split('.')[0]))
    
    if representation == 'ads':
        extract_graphemes = ads_grapheme_extraction
    elif representation == 'vds': 
        extract_graphemes = vds_grapheme_extraction
    elif representation == 'naive':
        extract_graphemes = naive_grapheme_extraction
    else:
        raise ValueError("Invalid chracter representation method. Must be one of ads, vds or naive.")

    print("\nPreprocessing synthetic data:")
    for i, name in enumerate(tqdm(filenames)):
        
        # Get labels from labels dict
        curr_word = labels_dict[name]
        curr_word = normalize_word(curr_word)
        # print(curr_word)
        curr_label = []
        words.append(curr_word)
        
        graphemes = extract_graphemes(curr_word)
        
        for grapheme in graphemes:
            if grapheme not in grapheme_dict:
                if inv_grapheme_dict is None:
                    grapheme_dict[grapheme] = count
                    curr_label.append(count)
                    count += 1
                else:
                    curr_label.append(grapheme_dict['<unk>'])
            else:
                curr_label.append(grapheme_dict[grapheme])
        lengths.append(len(curr_label))
        labels.append(curr_label)
    
    if inv_grapheme_dict is None:
        inv_grapheme_dict = {v: k for k, v in grapheme_dict.items()}
    
    # print(statistics.mean(lengths), words[lengths.index(max(lengths))], statistics.quantiles(lengths, n=4))
    return inv_grapheme_dict, words, labels, lengths


def encode_bnhtrd(csv_path, inv_grapheme_dict=None, representation='ads'):

    labels = {}
    words = {}
    lengths = {}
    
    labels_df = pd.read_csv(csv_path)
    
    if inv_grapheme_dict is None:
        grapheme_dict = {}
        # 0 is reserved for ctc blank
        grapheme_dict['<pad>'] = 1  # pad
        grapheme_dict['<unk>'] = 2  # OOV
        count = 3 # valid graphemes start from 3
    else:
        grapheme_dict = {v: k for k, v in inv_grapheme_dict.items()}
    
    
    if representation == 'ads':
        extract_graphemes = ads_grapheme_extraction
    elif representation == 'vds': 
        extract_graphemes = vds_grapheme_extraction
    elif representation == 'naive':
        extract_graphemes = naive_grapheme_extraction
    else:
        raise ValueError("Invalid chracter representation method. Must be one of ads, vds or naive.")
    
    print("\nPreprocessing data:")
    for i, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        curr_word = str(row['Word'])
        curr_word = curr_word.strip()
        curr_word = normalize_word(curr_word)
        aid = row['id']
        
        curr_label = []
        words[aid] = curr_word
        graphemes = extract_graphemes(curr_word)
        
        for grapheme in graphemes:
            if grapheme not in grapheme_dict:
                if inv_grapheme_dict is None:
                    grapheme_dict[grapheme] = count
                    curr_label.append(count)
                    count += 1
                else:
                    curr_label.append(grapheme_dict['<unk>'])
            else:
                curr_label.append(grapheme_dict[grapheme])
                    
        lengths[aid] = len(curr_label)
        labels[aid] = curr_label
    
    if inv_grapheme_dict is None:
        inv_grapheme_dict = {v: k for k, v in grapheme_dict.items()}
    
    return inv_grapheme_dict, words, labels, lengths



def encode_banglawriting_data(labels_file, inv_grapheme_dict=None, representation='ads'):
    
    if inv_grapheme_dict is None:
        grapheme_dict = {}
        # 0 is reserved for ctc blank
        grapheme_dict['<pad>'] = 1  # pad
        grapheme_dict['<unk>'] = 2  # OOV
        count = 3 # valid graphemes start from 3
    else:
        grapheme_dict = {v: k for k, v in inv_grapheme_dict.items()}
    
    labels = []
    words = []
    lengths = []
    
    with open(labels_file, 'r') as file:
        mappings = [line.strip() for line in file]
    
    labels_dict = {filename: label for filename, label in (mapping.split(maxsplit=1) for mapping in mappings)}
    filenames = sorted(labels_dict.keys(), key=lambda x: int(x.split('.')[0]))
    
    if representation == 'ads':
        extract_graphemes = ads_grapheme_extraction
    elif representation == 'vds': 
        extract_graphemes = vds_grapheme_extraction
    elif representation == 'naive':
        extract_graphemes = naive_grapheme_extraction
    else:
        raise ValueError("Invalid chracter representation method. Must be one of ads, vds or naive.")

    print("\nPreprocessing BanglaWriting data:")
    for i, name in enumerate(tqdm(filenames)):
        
        # Get labels from labels dict
        curr_word = labels_dict[name]
        curr_word = normalize_word(curr_word)
        # print(curr_word)
        curr_label = []
        words.append(curr_word)
        
        graphemes = extract_graphemes(curr_word)
        
        for grapheme in graphemes:
            if grapheme not in grapheme_dict:
                if inv_grapheme_dict is None:
                    grapheme_dict[grapheme] = count
                    curr_label.append(count)
                    count += 1
                else:
                    curr_label.append(grapheme_dict['<unk>']) 
            else:
                curr_label.append(grapheme_dict[grapheme])
        lengths.append(len(curr_label))
        labels.append(curr_label)
    
    if inv_grapheme_dict is None:
        inv_grapheme_dict = {v: k for k, v in grapheme_dict.items()}
        
    return inv_grapheme_dict, words, labels, lengths

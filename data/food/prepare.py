import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset ,Dataset # huggingface datasets
import re 
import json
# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8
# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc
enc = tiktoken.get_encoding("gpt2")



if __name__ == '__main__':
    # load the dataset
    # downllad the csv from https://recipenlg.cs.put.poznan.pl/dataset
    dataset = load_dataset('csv', data_files={'train': ['data/food/full_dataset.csv']},num_proc=num_proc_load_dataset)
    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    # function to generate the prompt for the model
    def gen_prompt(row):
        ingredients = json.loads(row['ingredients'])
        directions = json.loads(row['directions'])
        recipe_name = row['title']
        ingredients_string = ""
        directions_string = ""
        for i in ingredients:
            ingredients_string += "-"+ i + "\n"
        for d in directions:
            directions_string += "-"+ d + "\n"
        prompt = f""" [USER]You are a Michelin-starred chef. Using the below ingridents suggest a recipe.\nIngredients:\n{ingredients_string}[/USER]\n[INST] You can make {recipe_name} by following these steps:\n{directions_string}[/INST]"""
        return prompt
    
    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        prompt = gen_prompt(example)
        ids = enc.encode_ordinary(prompt) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out    

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['Unnamed: 0', 'title', 'ingredients', 'directions', 'link', 'source', 'NER'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
#
# chefGPT
![chefGPT](assets/chefgpt.jpeg)
## FYI 
A lot of code is borrowed from https://github.com/karpathy/nanoGPT
## Motivation behind this project
Everyday I use to spend a lot of time thinking about what should I make for dinner with avaliable ingridents. So I thought why not just train a GPT for suggesting me recipies.😝😝 Just joking!!

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## quick start

Manually download data fom this site and move it to food directory.[Download](https://recipenlg.cs.put.poznan.pl/dataset)

**Prepare dataset for fine tuning GPT2**

```
$ python data/food/prepare.py
```
This creates a train.bin and val.bin in that data directory. Now it is time to train your GPT. The size of it very much depends on the computational resources of your system:

**Now lets train GPT2**

```
$ python train.py
```
Adjust the configs as per GPU avaliable at your disposal


You have your personal chef read
```
$ python sample.py
```

## TODO
- [x] From strach write GPT and train it.
- [x] Implement Flash attention.
- [x] Implement KV cache Pagged attention.
- [x] Convert the model to run on mobile device.
- [x] Make it multimodal so I don't have to type ingridents everytime.

## Running the Model on Mobile Devices

To run the model on mobile devices, follow these steps:

1. Ensure your mobile device supports running PyTorch models.
2. Use the `optimize_for_mobile` method provided in the `model.py` to convert the model.
3. Export the optimized model using TorchScript for compatibility.
4. Deploy the TorchScript model on your mobile device using PyTorch Mobile.

For detailed instructions and compatibility checks, refer to the PyTorch Mobile documentation.

## Providing Multimodal Input for Ingredients

The model supports multimodal input, allowing users to input ingredients in both text and image formats. To use this feature:

1. For text input, simply type the ingredients as you would normally.
2. For image input, capture or upload an image of the ingredients.
3. The model will automatically detect and process the input type.
4. Ensure the `data/food/prepare.py` script is updated to handle image preprocessing.

This feature enhances the user experience by offering flexibility in how ingredients are inputted into the model.

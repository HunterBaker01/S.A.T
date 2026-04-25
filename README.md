# Show, Attend and Tell — Image Captioning

A PyTorch implementation of the image captioning model from [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044) (Xu et al., 2015).

> **Status:** The current implementation is a "Show and Tell" baseline (encoder-decoder without attention). The attention mechanism is planned for a future update.

## Architecture

```
Image → VGG16 Encoder → Spatial Features → Mean Pool → LSTM Decoder → Caption
```

- **Encoder:** Pretrained VGG16 (frozen convolutional layers) extracts 512-dim spatial feature maps from images.
- **Decoder:** Single-layer LSTM with word embeddings generates captions token-by-token.
- **Training:** Teacher forcing — ground-truth tokens are fed as input at each time step.
- **Inference:** Autoregressive greedy decoding (no beam search).

## Project Structure

```
├── main.py          # Training entry point and hyperparameters
├── models.py        # VGG16 encoder and LSTM decoder
├── eval.py          # Training and validation loops
├── get_data.py      # Vocabulary, dataset, and data loaders
├── create_h5.py     # Precompute VGG16 features → HDF5
├── requirements.txt # Python dependencies
└── data/            # (not tracked) images and captions
    ├── Images/      # Flickr8k JPEG images
    ├── captions.txt # CSV: image_filename, caption
    └── features.h5  # Precomputed VGG16 features (generated)
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the Flickr8k dataset

Download the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) and place the files in `data/`:

- `data/Images/` — directory of JPEG images
- `data/captions.txt` — CSV file with header `image,caption`

### 3. Train

```bash
python main.py
```

This will:
1. Build a vocabulary from all captions (words appearing >= 2 times).
2. Extract VGG16 features for every image and cache them in `data/features.h5`.
3. Train the LSTM decoder for 5 epochs, saving the best checkpoint by validation loss.

### Output files

| File | Description |
|---|---|
| `vocab.pkl` | Pickled Vocabulary object |
| `data/features.h5` | Precomputed VGG16 image features |
| `best_checkpoint.pth` | Best model checkpoint (lowest val loss) |
| `final_model.pth` | Model weights after final epoch |

## Hyperparameters

| Parameter | Value |
|---|---|
| Embedding dim | 256 |
| LSTM hidden dim | 512 |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 5 |
| Min word frequency | 2 |

## How It Works

1. **Feature extraction** (`create_h5.py`): Each image is resized/cropped to 224x224 and passed through VGG16's convolutional layers, producing a 7x7x512 feature map. This is reshaped to 49x512 (49 spatial regions, each with a 512-dim descriptor) and saved to HDF5.

2. **Data loading** (`get_data.py`): At training time, features are loaded from HDF5 (not recomputed), mean-pooled to a single 512-dim vector, and paired with tokenized captions wrapped in SOS/EOS tokens.

3. **Training** (`eval.py`): The decoder receives the image feature vector as the first "token", followed by embedded ground-truth caption tokens (teacher forcing). The model predicts the next token at each step. Loss is computed with CrossEntropyLoss, ignoring PAD tokens.

4. **Inference** (`models.py:generate`): Given an image feature vector, the decoder generates tokens autoregressively — each predicted token is fed back as input for the next step until EOS is produced or max length is reached.

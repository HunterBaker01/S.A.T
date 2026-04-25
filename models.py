"""
Neural network modules for the Show, Attend and Tell image captioning model.

Contains:
    - MyVGG16: Pretrained VGG16 encoder that extracts spatial feature maps.
    - Attention: Soft attention module that computes a context vector as a
      weighted sum over spatial encoder features at each decoding time step.
    - AttentionDecoder: LSTM decoder with attention that generates captions
      word-by-word, attending to different image regions at each step.

Reference:
    Xu et al., "Show, Attend and Tell: Neural Image Caption Generation
    with Visual Attention", ICML 2015.
"""

import torch
from torch import nn
from torchvision.models import vgg16


class MyVGG16(nn.Module):
    """
    Pretrained VGG16 feature extractor.

    Extracts convolutional feature maps from input images using the VGG16
    feature layers (no classifier). For a 224x224 input the output is
    (batch, 512, 7, 7), which is reshaped to (batch, 49, 512) — one
    512-dim vector for each of the 49 spatial regions.

    Args:
        freeze_features: If True, freeze all VGG16 feature parameters
            so they are not updated during backpropagation.
    """

    def __init__(self, freeze_features=True):
        super().__init__()
        model = vgg16(weights="DEFAULT")
        self.feature = nn.Sequential(*list(model.features.children()))

        if freeze_features:
            for param in self.feature.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Extract feature maps from an input image tensor.

        Args:
            x: Image tensor of shape (batch, 3, 224, 224).

        Returns:
            Feature maps of shape (batch, 512, 7, 7) for 224x224 input.
        """
        x = self.feature(x)
        return x


class Attention(nn.Module):
    """
    Soft attention module (Bahdanau-style).

    At each decoding time step, computes a probability distribution over the
    encoder's spatial regions conditioned on the decoder's previous hidden
    state, then returns the weighted-sum context vector.

    The energy function is:
        e_t,i = w^T tanh(W_enc * enc_i  +  W_dec * h_{t-1})
        alpha_t = softmax(e_t)
        context_t = sum_i(alpha_t,i * enc_i)

    Args:
        encoder_dim: Dimensionality of encoder feature vectors (512 for VGG16).
        hidden_dim: Dimensionality of the decoder LSTM hidden state.
        attention_dim: Internal dimensionality of the attention MLP.
    """

    def __init__(self, encoder_dim, hidden_dim, attention_dim):
        super().__init__()
        self.enc_att = nn.Linear(encoder_dim, attention_dim)   # transform encoder features
        self.dec_att = nn.Linear(hidden_dim, attention_dim)    # transform decoder hidden state
        self.full_att = nn.Linear(attention_dim, 1)            # compute scalar energy per region
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Compute attention-weighted context vector.

        Args:
            encoder_out: Spatial encoder features, shape (batch, num_pixels, encoder_dim).
            decoder_hidden: Decoder hidden state, shape (batch, hidden_dim).

        Returns:
            context: Weighted context vector, shape (batch, encoder_dim).
            alpha: Attention weights, shape (batch, num_pixels).
        """
        # (batch, num_pixels, attention_dim)
        att_enc = self.enc_att(encoder_out)
        # (batch, attention_dim)  ->  (batch, 1, attention_dim) for broadcasting
        att_dec = self.dec_att(decoder_hidden).unsqueeze(1)
        # (batch, num_pixels, 1)  ->  (batch, num_pixels)
        energy = self.full_att(self.relu(att_enc + att_dec)).squeeze(2)
        alpha = self.softmax(energy)
        # weighted sum: (batch, encoder_dim)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class AttentionDecoder(nn.Module):
    """
    LSTM decoder with soft attention for image captioning.

    At each time step the decoder:
        1. Computes attention over the encoder's spatial features using
           the previous hidden state.
        2. Concatenates the attention context vector with the word embedding
           and feeds this into the LSTM cell.
        3. Projects the LSTM output to vocabulary logits.

    The model also includes:
        - ``init_h`` / ``init_c``: linear layers that initialise the LSTM
          hidden and cell states from the mean-pooled encoder features.
        - ``f_beta``: a sigmoid gate that modulates how much the context
          vector influences each time step (from the SAT paper).

    Args:
        embedding_dim: Word embedding dimensionality.
        hidden_dim: LSTM hidden state dimensionality.
        vocab_size: Number of words in the vocabulary.
        encoder_dim: Encoder feature dimensionality (512 for VGG16).
        attention_dim: Internal dimensionality of the attention MLP.
        dropout: Dropout probability applied before the output projection.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size,
                 encoder_dim=512, attention_dim=256, dropout=0.5):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.attention = Attention(encoder_dim, hidden_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # LSTM cell input = word embedding + context vector
        self.lstm_cell = nn.LSTMCell(embedding_dim + encoder_dim, hidden_dim)

        # Initialise LSTM states from mean encoder features
        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)

        # Sigmoid gate on the context vector (Section 4.2.1 of the paper)
        self.f_beta = nn.Linear(hidden_dim, encoder_dim)

        # Output projection
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def _init_hidden(self, encoder_out):
        """
        Initialise LSTM hidden / cell states from mean-pooled encoder features.

        Args:
            encoder_out: (batch, num_pixels, encoder_dim)

        Returns:
            h0: (batch, hidden_dim)
            c0: (batch, hidden_dim)
        """
        mean_features = encoder_out.mean(dim=1)          # (batch, encoder_dim)
        h0 = self.init_h(mean_features)                  # (batch, hidden_dim)
        c0 = self.init_c(mean_features)                  # (batch, hidden_dim)
        return h0, c0

    def forward(self, encoder_out, captions, lengths):
        """
        Teacher-forced forward pass.

        Args:
            encoder_out: Spatial features of shape (batch, num_pixels, encoder_dim).
            captions: Ground-truth token indices of shape (batch, max_len).
            lengths: List of actual caption lengths (including SOS, excluding padding).

        Returns:
            predictions: Logits of shape (batch, max_decode_len, vocab_size).
            alphas: Attention weights of shape (batch, max_decode_len, num_pixels).
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # We decode for (max_caption_length - 1) steps; we don't predict SOS.
        max_decode_len = max(lengths) - 1

        embeddings = self.embedding(captions)             # (B, max_len, embed_dim)
        h, c = self._init_hidden(encoder_out)

        predictions = torch.zeros(batch_size, max_decode_len, self.vocab_size,
                                  device=encoder_out.device)
        alphas = torch.zeros(batch_size, max_decode_len, num_pixels,
                             device=encoder_out.device)

        for t in range(max_decode_len):
            # At step t, feed embedding of caption token t (starts with SOS at t=0)
            context, alpha = self.attention(encoder_out, h)

            # Gating scalar (sigmoid) applied element-wise to context
            gate = torch.sigmoid(self.f_beta(h))          # (B, encoder_dim)
            context = gate * context

            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            preds = self.fc(self.dropout(h))              # (B, vocab_size)
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha

        return predictions, alphas

    def generate(self, encoder_out, max_len=20, start_idx=1, end_idx=2):
        """
        Autoregressively generate captions without teacher forcing.

        Args:
            encoder_out: Spatial features of shape (batch, num_pixels, encoder_dim).
            max_len: Maximum number of tokens to generate.
            start_idx: Token index for SOS.
            end_idx: Token index for EOS.

        Returns:
            captions: Predicted token indices of shape (batch, generated_len).
            all_alphas: Attention weights of shape (batch, generated_len, num_pixels).
        """
        batch_size = encoder_out.size(0)
        device = encoder_out.device

        h, c = self._init_hidden(encoder_out)

        # First input is the SOS token
        inputs = self.embedding(
            torch.full((batch_size,), start_idx, dtype=torch.long, device=device)
        )

        captions = []
        all_alphas = []

        for _ in range(max_len):
            context, alpha = self.attention(encoder_out, h)
            gate = torch.sigmoid(self.f_beta(h))
            context = gate * context

            lstm_input = torch.cat([inputs, context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            logits = self.fc(h)
            predicted = logits.argmax(dim=1)

            captions.append(predicted)
            all_alphas.append(alpha)

            if (predicted == end_idx).all():
                break

            inputs = self.embedding(predicted)

        captions = torch.stack(captions, dim=1)
        all_alphas = torch.stack(all_alphas, dim=1)
        return captions, all_alphas

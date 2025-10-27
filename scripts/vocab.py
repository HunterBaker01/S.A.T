import re
from collections import Counter
import csv

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.toks = {0: "pad", 1: "sos", 2: "eos", 3: "unk"}
        self.ind = {v: k for k, v in self.toks.items()}
        self.index = 4

    def __len__(self):
        return len(self.toks)
    def tockenizer(self, text):
        "Standardize the data"
        text = text.lower()
        tockens = re.findall(r"\w+", text)
        return tockens

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            tokens = self.tockenizer(sentence)
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.ind[word] = self.index
                self.toks = [self.index] = word
                self.index += 1

    def numerical(self, text):
        tokens = self.tockenizer(text)
        numerical = []
        for token in tokens:
            if token in self.ind:
                numerical.append(self.ind[token])
            else:
                numerical.append(self.ind["unk"])
        return numerical

def parse_tokens(file):
    imgid2captions = {}
    with open(file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            img_id, caption = row[0], row[1]
            if img_id not in imgid2captions:
                imgid2captions[img_id] = []
            imgid2captions[img_id].append(caption)
    return imgid2captions

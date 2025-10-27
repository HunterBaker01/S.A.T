from vocab import parse_tokens, Vocabulary
import pickle

TOKENS_FILE = ""

imgid2captions = parse_tokens(TOKENS_FILE)
all_captions = []
for caps in imgid2captions.values():
    all_captions.extend(caps)

vocab = Vocabulary(freq_threshold=MIN_WORD_FREQ)
vocab.build_vocab(all_captions)

with open(VOCAB_PATH, "wb") as f:
    pickle.dump(vocab, f)

print("Vocabulary saved to: ", VOCAB_PATH)

vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

img_ids = list(imgid2captions.keys())
random.shuffle(img_ids)
split_idx = int(0.8 * len(img_ids))
train_ids = img_ids[:split_idx]
test_ids = img_ids[split_idx:]

train_dict = {iid: imgid2captions[iid] for iid in train_ids}
test_dict= {iid: imgid2captions[iid] for iid in test_ids}

def build_voab(caption_file):
    counter = Counter()
    with open(caption_file, 'r') as f:
        for line in f:
            caption = line.strip().split('\t')[1]
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
    vocab = {}
    vocab['<pad>'] = 0
    vocab['<start>'] = 1
    vocab['<end>'] = 2
    vocab['<unk>'] = 3
    idx = 4
    for word, count in counter.items():
        if count >= 5:
            vocab[word] = idx
            idx += 1
    return vocab

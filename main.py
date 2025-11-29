from models.encoder import MyVGG16
from models.decoder import MyLSTM
from models.combined import CombinedModel
from scripts.build_vocab import build_vocab
from scripts.loaders import get_loaders
from scripts.get_data import Flickr8kDataset

EMBED_DIM = 256
HIDDEN_DIM = 512
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 50
MIN_WORD_FREQ = 1
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
 
IMAGES_DIR = "flickr8k/Images"
TOKENS_FILE = "flickr8k/captions.txt"
 
BEST_CHECKPOINT_PATH = "best_checkpoint.pth"
FINAL_MODEL_PATH = "final_model.pth"
VOCAB_PATH = "vocab.pkl"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
 
encoder = MyVGG16()
decoder = MyLSTM(EMBED_DIM, HIDDEN_DIM, vocab_size)
model = CombinedModel(encoder, decoder).to(DEVICE)

train_dict, test_dict = build_vocab(TOKENS_FILE, MIN_WORD_FREQ, VOCAB_PATH)

train_loader, test_loader = get_loaders(train_dict, test_dict, vocab, transform)

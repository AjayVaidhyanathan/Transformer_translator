import torch
from dataset import BilingualDataset
from train import get_model, get_ds# Import your model and config
from config import get_config,  get_weights_file_path

config = get_config()
tokenizer_src, tokenizer_target = get_ds(config)

mock_ds = [{'translation': {'de': 'Guten Tag', 'en': 'Good day'}}]
dataset = BilingualDataset(mock_ds, tokenizer_src, tokenizer_target, 'de', 'en', seq_len=10)
# Initialize model (match training setup)
model = get_model(config, tokenizer_src, tokenizer_target,  vocab_target_len)

# Load checkpoint (ensure keys match)
checkpoint = torch.load('weights/tmodel_06.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
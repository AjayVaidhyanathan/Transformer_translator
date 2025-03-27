import torch
from train import get_model, get_ds# Import your model and config
from config import get_config,  get_weights_file_path

config = get_config()
tokenizer_src, tokenizer_target = get_ds(config)

# Initialize model (match training setup)
model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_target.get_vocab_size())

# Load checkpoint (ensure keys match)
checkpoint = torch.load('weights/tmodel_06.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
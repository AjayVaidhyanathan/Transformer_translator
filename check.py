import torch
from pathlib import Path

# Check file existence and size
model_path = Path("tmodel_05.pt")
assert model_path.exists(), "Model file not found!"
print(f"File size: {model_path.stat().st_size / 1e6:.2f} MB")  # Typically 100MB+ for transformers

# Quick load test
try:
    checkpoint = torch.load(model_path, map_location="cpu")
    print("Keys in checkpoint:", checkpoint.keys())  # Should contain keys like 'epoch', 'model_state_dict'
except Exception as e:
    print("Corrupted file:", str(e))


from model import build_transformer
from config import get_config

config = get_config()
model = build_transformer(
    source_vocab_size = 30000,
    target_vocab_size = 20086,
    source_seq_length = config['seq_len'],
    target_seq_length= config['seq_len'],
)

# Test parameter loading
try:
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model weights loaded successfully")
except RuntimeError as e:
    print("❌ Shape mismatch error:", str(e))


from tokenizers import Tokenizer

# Load tokenizers
tokenizer_src = Tokenizer.from_file(config['tokenizer_file'].format('de'))
tokenizer_tgt = Tokenizer.from_file(config['tokenizer_file'].format('en'))

# Test a simple known translation
test_sentence = "Das ist ein einfacher Test."
expected_translation = "This is a simple test."

model.eval()
with torch.no_grad():
    # Use your existing translate() function
    translation = translate(test_sentence, model, tokenizer_src, tokenizer_tgt)
    print(f"Input: {test_sentence}")
    print(f"Output: {translation}")
    print(f"Expected: {expected_translation}")

    # Basic assertion
    assert "simple" in translation.lower(), "Basic translation failed!"    
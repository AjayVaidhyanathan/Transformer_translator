# Test with mock data
from config import get_config
from dataset import BilingualDataset
from train import get_or_build_tokenizer

mock_ds = [{'translation': {'de': 'Guten Tag', 'en': 'Good day'}}]
config = get_config()
tokenizer_src = get_or_build_tokenizer(config, mock_ds, config['lang_src'])
tokenizer_target = get_or_build_tokenizer(config, mock_ds, config['lang_tgt'])


dataset = BilingualDataset(mock_ds, tokenizer_src, tokenizer_target, 'de', 'en', seq_len=10)
sample = dataset[0]

print("Encoder input:", sample["encoder_input"].shape, sample["encoder_input"])
print("Decoder input:", sample["decoder_input"].shape, sample["decoder_input"])
print("Label:", sample["label"].shape, sample["label"])
print("Encoder mask:", sample["encoder_mask"].shape, sample["encoder_mask"])
print("Decoder mask:", sample["decoder_mask"].shape, sample["decoder_mask"])
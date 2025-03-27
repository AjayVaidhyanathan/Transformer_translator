import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_target, source_lang, target_lang, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_target = tokenizer_target
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.seq_len = seq_len

        self.sos_token_src = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)
        self.eos_token_src = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token_src = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)

        self.sos_token_tgt = torch.tensor([tokenizer_target.token_to_id('[SOS]')], dtype = torch.int64)
        self.eos_token_tgt = torch.tensor([tokenizer_target.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token_tgt = torch.tensor([tokenizer_target.token_to_id('[PAD]')], dtype = torch.int64)


    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx) -> any:
        source_target_pair = self.ds[idx]
        source_text = source_target_pair['translation'][self.source_lang]
        target_text = source_target_pair['translation'][self.target_lang]

        enc_input_tokens = self.tokenizer_src.encode(source_text).ids
        dec_input_tokens = self.tokenizer_target.encode(target_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1


        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        
        # Add SOS and EOS to the source text
        encoder_input = torch.cat([
            self.sos_token_src,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token_src,
            torch.tensor([self.pad_token_src] * enc_num_padding_tokens, dtype=torch.int64)
        ],
        dim=0,
        )

        # Add SOS to the decoder text
        decoder_input = torch.cat([
            self.sos_token_tgt,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token_tgt] * dec_num_padding_tokens, dtype=torch.int64)  
        ],
        dim=0,
        )

        #Add EOS to the label (what we expect as output from decoder)
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token_tgt,
            torch.tensor([self.pad_token_tgt] * dec_num_padding_tokens, dtype=torch.int64)  
        ],
        dim=0,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        encoder_mask = (encoder_input != self.pad_token_src).unsqueeze(0).unsqueeze(0).int()
        decoder_mask = (decoder_input != self.pad_token_tgt).unsqueeze(0).unsqueeze(0).int() & \
                   causal_mask(decoder_input.size(0))
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "source_text": source_text,
            "target_text": target_text
        }   

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
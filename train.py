import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path
import warnings
from tqdm import tqdm
from IPython.display import display, FileLink

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config, small_batch):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_target = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    if small_batch:
        ds_raw = ds_raw.select(range(50))

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_target, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_target, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_source = 0
    max_len_target = 0

    for item in ds_raw:
        source_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        target_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        max_len_source = max(max_len_source, len(source_ids))
        max_len_target = max(max_len_target, len(target_ids))


    print(f'Max length of source sentence: {max_len_source}')  
    print(f'Max length of target sentence: {max_len_target}')  


    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_target

def get_model(config, vocab_source_len, vocab_target_len):
    model = build_transformer(vocab_source_len, vocab_target_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model


def train_model(config, small_batch=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')


    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)


    train_dataloader, val_dataloader, tokenizer_src, tokenizer_target = get_ds(config, small_batch)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_target.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
      

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)   


    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        total_loss = 0
        total_accuracy = 0
        batch_count = 0
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            print("Encoder input dtype:", encoder_input.dtype)
            print("Encoder input dtype:", decoder_input.dtype)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_target.get_vocab_size()), label.view(-1))

            #Accuracy check
            preds = torch.argmax(proj_output, dim=-1)
            correct = (preds == label).float().sum()
            accuracy = correct / label.numel()

            total_loss += loss.item()
            total_accuracy += accuracy.item()
            batch_count += 1

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.add_scalar('train/accuracy', accuracy.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Update the weights
            optimizer.step()
            

            global_step += 1

        avg_loss = total_loss / batch_count
        avg_accuracy = total_accuracy / batch_count    

        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)    

    print(f"Model saved: {model_filename}")

        # Move the model to `/kaggle/working/` for easy downloading
    kaggle_model_path = f"/kaggle/working/model_epoch_{epoch:02d}.pt"
    shutil.copy(model_filename, kaggle_model_path)

    print(f"Model moved to: {kaggle_model_path}")

        # Create a download link (only works in Jupyter-based environments like Kaggle)
    display(FileLink(kaggle_model_path))
    print(f"Download model from: {kaggle_model_path}")


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)        
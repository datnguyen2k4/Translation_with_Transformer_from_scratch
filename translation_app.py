import streamlit as st

st.set_page_config(
    page_title="ChatGPT 4o - Translator",
    layout="centered",
    initial_sidebar_state="collapsed"
)

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torchtext; 
torchtext.disable_torchtext_deprecation_warning()
import pandas as pd
import spacy
import math
from tqdm import tqdm
import torch
from torch import nn
import lightning as pl
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from typing import Iterable, List, Callable
from torch.utils.data import Dataset, DataLoader
from underthesea import sent_tokenize, text_normalize, word_tokenize
from torchmetrics.text import BLEUScore
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from transformer_from_scratch import Transformer
# import tensorboard
from lightning.pytorch.loggers import TensorBoardLogger



PAD_IDX = 0

def read_text(en_file, vi_file):
    """
    Read text pairs from files, then build a dataframe with two columns: english and vietnamese
    """
    
    with open(en_file, 'r') as f:
        en_lines = f.readlines()
        
    with open(vi_file, 'r') as f:
        vi_lines = f.readlines()
        
    data = pd.DataFrame({'english': en_lines, 'vietnamese': vi_lines})
    
    return data


def tokenize_vi(text: str) -> List[List[str]]:
    """
    Tokenize a Vietnamese text into sentences and words
    
    Args:
        text (str): the input text
        
    Returns:
        List[List[str]]: a list of sentences, each sentence is a list of tokens
    """
    # Step 1: Sentence Tokenization
    sentences = sent_tokenize(text)
    
    # Step 2: Text Normalization (assuming it's just lowercasing here)
    sentences = [text_normalize(sentence) for sentence in sentences]
        
    # Step 3: Word Tokenization
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
        
    # Flatten the list
    tokenized_sentences = [word for sentence in tokenized_sentences for word in sentence]
        
    # Lowercase all tokens
    tokenized_sentences = [word.lower() for word in tokenized_sentences]
    
    return tokenized_sentences


class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data.iloc[idx]
        src_tokens = self.src_tokenizer(src.lower())
        tgt_tokens = self.tgt_tokenizer(tgt.lower())
        
        # TODO: your code here
        # 1. Add '<sos>' and '<eos>' into the sentence
        
        src_tokens = ['<sos>'] + src_tokens + ['<eos>']
        tgt_tokens = ['<sos>'] + tgt_tokens + ['<eos>']

        #. Convert into a tensor of IDs
        
        src_tensor = torch.tensor([self.src_vocab[token] for token in src_tokens], dtype=torch.long)
        tgt_tensor = torch.tensor([self.tgt_vocab[token] for token in tgt_tokens], dtype=torch.long)
        
        return src_tensor, tgt_tensor
        
        
        
class TranslationDataModule(pl.LightningDataModule):
    def __init__(self, df, src_tokenizer, tgt_tokenizer, batch_size=32):
        super().__init__()
        self.df = df
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.batch_size = batch_size
        self.src_vocab = None
        self.tgt_vocab = None

    def setup(self, stage=None):
        if self.src_vocab is not None and self.tgt_vocab is not None:
            return
        
        # Build vocabularies
        self.src_vocab = build_vocab_from_iterator(
            self.df['english'].apply(lambda x: self.src_tokenizer(x.lower())),
            specials=['<unk>', '<pad>', '<sos>', '<eos>']
        )
    
        # TODO: your code here
        # Construct the tgt_vocab
        self.tgt_vocab = build_vocab_from_iterator(
            self.df['vietnamese'].apply(lambda x: self.tgt_tokenizer(x.lower())),
            specials=['<unk>', '<pad>', '<sos>', '<eos>']
        )
    
        self.src_vocab.set_default_index(self.src_vocab['<unk>'])
        
        # TODO: your code here
        # Set the default index for the target vocabulary
        
        self.tgt_vocab.set_default_index(self.tgt_vocab['<unk>'])

        # TODO: your code here
        # Create datasets
        
        self.tranlastion_dataset = TranslationDataset(
            self.df,
            self.src_vocab,
            self.tgt_vocab,
            self.src_tokenizer,
            self.tgt_tokenizer
        )
        
        train_size = int(0.8 * len(self.df))
        val_size = len(self.df) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.tranlastion_dataset, [train_size, val_size])
        
    def train_dataloader(self):
        # TODO: your code here
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        # TODO: your code here
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        src_batch, tgt_batch = zip(*batch)
        src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=self.src_vocab['<pad>'], batch_first=True)
        tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=self.tgt_vocab['<pad>'], batch_first=True)
        return src_batch, tgt_batch


class TranslationModel(pl.LightningModule):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, max_seq_length=80):
        super().__init__()
        self.transformer = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=nhead,
            num_layers=num_layers,
            d_ff=dim_feedforward,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        
    def forward(self, src, tgt):
        return  self.transformer(src, tgt)


    def training_step(self, batch, batch_idx):
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        output = self(src, tgt_input)
        
        loss = self.loss_fn(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: your code here
        
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        output = self(src, tgt_input)
        
        loss = self.loss_fn(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
        self.log('val_loss', loss, prog_bar=True)
        return loss
        

    def configure_optimizers(self):
        # TODO: your code here
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        # Learning rate scheduler: Reduces LR when validation loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  
            factor=0.1, 
            patience=3, 
            verbose=False,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
        
        
# Translate one sentence 
def translate_one_sentence(model, data_module, sentence, max_len=50):
    """
    Translate a single sentence from English to Vietnamese
    Args:
        model (nn.Module): The trained translation model.
        data_module: The DataModule containing vocab and dataset.
        sentence (str): The English sentence to translate.
        max_len (int): Maximum length of translation.
    Returns:
        Tuple[str, str, str]: (source, target, predicted) sentences.
    """
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    src_vocab = data_module.src_vocab
    tgt_vocab = data_module.tgt_vocab
    src_tokenizer = data_module.src_tokenizer
    tgt_tokenizer = data_module.tgt_tokenizer

    PAD_IDX = tgt_vocab['<pad>']
    SOS_IDX = tgt_vocab['<sos>']
    EOS_IDX = tgt_vocab['<eos>']
    
    # Prepare source sentence
    src_tokens = src_tokenizer(sentence.lower())
    src_tokens = ['<sos>'] + src_tokens + ['<eos>']
    src_tensor = torch.tensor([src_vocab[token] for token in src_tokens], dtype=torch.long).unsqueeze(0).to(device)  # Move to MPS
    
    # Prepare input for model
    tgt_input = torch.tensor([[SOS_IDX]], device=src_tensor.device)  # [1, 1] with <sos>
    
    model = model.to(device)  # Move model to MPS

    # Autoregressive decoding
    with torch.no_grad():
        for _ in range(max_len):
            output = model(src_tensor, tgt_input)
            next_token = output[:, -1, :].argmax(dim=-1)  # take the last token prediction
            tgt_input = torch.cat([tgt_input, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == EOS_IDX:
                break

    # Decode predicted translation
    pred_tokens = [token.item() for token in tgt_input[0]]
    pred_tokens = [token for token in pred_tokens if token not in {PAD_IDX, SOS_IDX, EOS_IDX}]
    pred_sentence = ' '.join([tgt_vocab.lookup_token(token) for token in pred_tokens])

    return sentence, pred_sentence


def run_app(model, data_module):

    # CSS tùy chỉnh
    st.markdown("""
        <style>
            body {
                background-color: #0f0f0f;
                color: white;
            }
            .stTextInput input {
                background-color: #2e2e2e;
                color: white;
                border-radius: 10px;
                padding: 10px;
                border: none;
            }
            .stMarkdown {
                background-color: #1e1e1e;
                padding: 15px;
                border-radius: 10px;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

    # Tiêu đề ứng dụng
    st.markdown("<h1 style='text-align: center;'>Translate</h1>", unsafe_allow_html=True)

    # Hộp nhập văn bản
    user_input = st.text_input("", placeholder="Nhập câu tiếng Anh cần dịch...")

    # Nếu có đầu vào, thực hiện dịch và hiển thị kết quả
    if user_input:
        src, translated_text = translate_one_sentence(model, data_module, user_input)
        translated_text = translated_text.capitalize()
        st.markdown(f"**Kết quả dịch:**")
        st.markdown(f"<div class='stMarkdown'>{translated_text}</div>", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():    
    df = read_text('Data/en_sents', 'Data/vi_sents')
    
    if not spacy.util.is_package('en_core_web_md'):
        spacy.cli.download('en_core_web_md')
        
    nlp_en = spacy.load('en_core_web_md')
    en_tokenizer = get_tokenizer('spacy', language='en_core_web_md')
    vi_tokenizer = get_tokenizer(tokenize_vi)
    
    data_module = TranslationDataModule(df, en_tokenizer, vi_tokenizer, batch_size=32)
    data_module.setup()
    
    checkpoint_path = 'checkpoints_machine_translation/transformer-best-epoch=00-v3.ckpt'
    model = TranslationModel.load_from_checkpoint(
        checkpoint_path,
        src_vocab_size=len(data_module.src_vocab),
        tgt_vocab_size=len(data_module.tgt_vocab),
        d_model=256,
        nhead=4,
        num_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_length=80
    )
    model.eval()
    return model, data_module

if __name__ == "__main__":
    model, data_module = load_model_and_data()
    run_app(model, data_module)
    
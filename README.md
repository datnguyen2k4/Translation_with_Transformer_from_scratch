# English-Vietnamese Translation with Transformer

This project implements a machine translation system from English to Vietnamese using the Transformer architecture from scratch. The implementation focuses on building a deep learning model that can effectively translate between these two languages.

## Dataset

The project uses the English-Vietnamese Translation dataset from Kaggle:
- Dataset: [English-Vietnamese Translation](https://www.kaggle.com/datasets/hungnm/englishvietnamese-translation)
- Contains parallel English-Vietnamese sentence pairs
- Dataset structure:
  - `en_sents`: English sentences
  - `vi_sents`: Vietnamese sentences

To use the dataset:
1. Download from Kaggle
2. Extract the files to the `Data` directory in the project root
3. The files should be named `en_sents` and `vi_sents`

## Project Structure

```
.
├── transformer_from_scratch/    # Core transformer implementation
├── translation_app.py          # Streamlit web application for translation
└── translation_machine.ipynb   # Jupyter notebook for model training and evaluation
```

## Features

- Custom implementation of the Transformer architecture
- English to Vietnamese translation capabilities
- Interactive web interface built with Streamlit
- Training and evaluation pipeline using PyTorch Lightning
- Vietnamese text processing using underthesea library
- English text processing using spaCy
- BLEU score evaluation

## Requirements

- Python 3.x
- PyTorch >= 2.0.0
- PyTorch Lightning >= 2.0.0
- torchtext >= 0.15.0
- pandas >= 1.3.0
- spaCy >= 3.5.0
- underthesea >= 6.0.0
- torchmetrics >= 0.11.0
- Streamlit >= 1.22.0

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Translation_EN_VI_with_Transformer_from_scrach
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Training the Model

1. Open the Jupyter notebook:
```bash
jupyter notebook translation_machine.ipynb
```

2. Follow the notebook instructions to train the model on your dataset.

### Running the Translation Web App

1. Start the Streamlit application:
```bash
streamlit run translation_app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Enter English text and get the Vietnamese translation

## Model Architecture

The project implements the Transformer architecture with the following components:

- Multi-head attention mechanism
- Position-wise feed-forward networks
- Positional encoding
- Encoder-Decoder structure
- Layer normalization
- Residual connections

## Data Processing

- English text is processed using spaCy for tokenization
- Vietnamese text is processed using underthesea library for:
  - Sentence tokenization
  - Text normalization
  - Word tokenization

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.


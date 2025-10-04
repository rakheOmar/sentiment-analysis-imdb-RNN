# RNN IMDB Movie Review Analysis 🎬

A small Streamlit app demonstrating sentiment analysis of IMDB movie reviews using a Simple RNN.

> Built while learning recurrent neural networks and sequence modeling (tutorial inspiration: [Krish Naik](https://www.youtube.com/@krishnaik06)).

## 📷 Screenshot

<img width="2560" height="1600" alt="image" src="https://github.com/user-attachments/assets/f36b0987-cc5e-4049-8f07-ea1c98edf3db" />

## 🛠️ Tech Stack

- **Python 3.13**
- **TensorFlow / Keras** - RNN model (SimpleRNN)
- **Streamlit** - Lightweight web UI for demo
- **scikit-learn** - Tokenization helpers and preprocessing utilities
- **uv** - Fast Python package manager (project sync)
- **ruff** - Fast Python linter

## �📦 Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/rakheOmar/sentiment-analysis-imdb-RNN.git
   cd "sentiment-analysis-imdb-RNN"
   ```

2. **Install dependencies using uv**

   ```bash
   uv sync
   ```

3. **Activate the virtual environment**

   ```bash
   # On Windows PowerShell
   .\rnn-project\Scripts\activate.ps1

   # On Unix/MacOS
   source rnn-project/bin/activate
   ```

## 🚀 Run the App

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`.

## 📁 Project Structure

```
├── main.py                    # Streamlit application (text input + predict)
├── models/                    # Trained RNN model
│   └── simple_rnn_imdb.keras  # Simple RNN trained on IMDB data
├── data/                      # Optional dataset or saved tokenizers
│   └── (IMDB dataset or tokenizers)
├── notebooks/                 # Jupyter notebooks
│   ├── embedding.ipynb        # Explore embeddings and tokenization
│   ├── training.ipynb         # Model training experiments
│   └── prediction.ipynb       # In-notebook inference examples
└── pyproject.toml             # Project dependencies
```

## 🔧 Development

**Lint code with ruff:**

```bash
ruff check .
```

**Format code with ruff:**

```bash
ruff format .
```

## 💡 Usage

1. Open the app and paste or type a movie review into the left panel.
2. Click **"Predict Sentiment"** (or equivalent) button.
3. See the predicted sentiment and probability on the right panel.

## 🧠 Model Details

This project demonstrates a Simple RNN-based sentiment classifier trained on IMDB movie reviews.

**Input:**

- Tokenized sequences of word indices (padded to a fixed length)
- Optional embedding layer transforms tokens into dense vectors

**Architecture:**

- Embedding layer
- SimpleRNN layer(s)
- Dense output with sigmoid for binary sentiment

**Output:**

- Sentiment probability (0.0 - 1.0)
- Class label: Positive / Negative

## 🙏 Acknowledgments

- Dataset: Keras IMDB dataset
- Tutorial and guidance: [Krish Naik](https://www.youtube.com/@krishnaik06)

---

Made with ❤️ while learning sequence models

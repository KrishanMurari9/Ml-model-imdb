# Data

The project uses the **IMDB 50K Movie Reviews** dataset.

You do not need to manually download the dataset for the default workflow.
`src/train.py` downloads it through the Hugging Face `datasets` library:

```python
load_dataset("imdb")
```

The dataset contains:
- 25,000 training reviews
- 25,000 test reviews
- Binary labels: 0 = negative, 1 = positive

If your internet is unavailable, download the IMDB dataset separately and adapt `src/train.py` to read the local CSV/TSV files.

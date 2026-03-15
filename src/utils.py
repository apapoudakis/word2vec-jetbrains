import re
import numpy as np


def sigmoid(x):
    """Numerically stable sigmoid."""
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    # For x >= 0:  σ = 1 / (1 + exp(-x))
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    # For x < 0:   σ = exp(x) / (1 + exp(x))  — avoids exp(large positive)
    ez = np.exp(x[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def load_yaml(path):
    config = {}
    with open(path, "r") as f:
        for line in f:
            line = line.split("#")[0].strip()   # strip comments
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            config[key] = val
    return config


def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    return tokens


def build_vocab(tokens, min_count=2):

    counts = {}
    for w in tokens:
        counts[w] = counts.get(w, 0) + 1

    idx2word = [w for w, c in counts.items() if c >= min_count]
    idx2word.sort()                       
    word2idx = {w: i for i, w in enumerate(idx2word)}
    freqs = np.array([counts[w] for w in idx2word], dtype=np.float64)
    
    return word2idx, idx2word, freqs



def subsample(token_ids, freqs, t=1e-5):

    total = freqs.sum()
    if total < 5000:
        return list(token_ids)

    word_freq = freqs / total

    keep_prob = np.sqrt(t / word_freq)
    keep_prob = np.minimum(keep_prob, 1.0)

    rng = np.random.default_rng()
    kept = [idx for idx in token_ids if rng.random() < keep_prob[idx]]
    return kept



def make_training_pairs(token_ids, window_size=5):

    pairs = []
    n = len(token_ids)
    rng = np.random.default_rng()
    for i in range(n):
        w = rng.integers(1, window_size + 1)
        start = max(0, i - w)
        end = min(n, i + w + 1)
        centre = token_ids[i]
        for j in range(start, end):
            if j == i:
                continue
            pairs.append((centre, token_ids[j]))
    return pairs



def build_noise_table(freqs, table_size=int(1e6)):
    """
    Create a large array of word indices where each word appears in
    proportion to freq(w)^0.75.  Drawing a negative sample is then
    just picking a random slot — O(1).
    """
    powered = np.power(freqs, 0.75)
    powered /= powered.sum()

    table = np.zeros(table_size, dtype=np.int64)
    idx = 0
    cumulative = 0.0
    for word_idx in range(len(freqs)):
        cumulative += powered[word_idx]
        fill_to = int(cumulative * table_size)
        # make sure we don't overshoot due to rounding
        fill_to = min(fill_to, table_size)
        table[idx:fill_to] = word_idx
        idx = fill_to
    # fill any remaining slots with the last word
    table[idx:] = len(freqs) - 1
    return table


def sample_negatives(noise_table, count):
    slots = np.random.randint(0, len(noise_table), size=count)
    return noise_table[slots]


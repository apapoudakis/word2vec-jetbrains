import argparse
import os
import time
import numpy as np

from src.utils import (
    load_yaml,
    tokenize, build_vocab, subsample,
    make_training_pairs, build_noise_table, sample_negatives,
)
from src.model import Word2Vec


def load_config():
    parser = argparse.ArgumentParser(description="Train word2vec (SGNS)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to a YAML config file")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    required = [
        "corpus", "min_count", "embed_dim", "epochs", "lr",
        "window", "neg_samples", "subsample_t", "batch_size",
        "seed", "save_path",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    return cfg


def epoch_save_path(base_path, epoch):
    """Derive a per-epoch filename: foo.npz -> foo_epoch3.npz"""
    root, ext = os.path.splitext(base_path)
    return f"{root}_epoch{epoch}{ext}"


def main():

    cfg = load_config()

    np.random.seed(cfg["seed"])

    with open(cfg["corpus"], "r") as f:
        text = f.read()

    tokens = tokenize(text)
    print(f"Tokens after cleaning: {len(tokens)}")

    word2idx, idx2word, freqs = build_vocab(tokens, min_count=cfg["min_count"])
    vocab_size = len(idx2word)
    print(f"Vocabulary size (min_count={cfg['min_count']}): {vocab_size}")

    token_ids = [word2idx[w] for w in tokens if w in word2idx]
    print(f"Tokens mapped to vocab: {len(token_ids)}")

    noise_table = build_noise_table(freqs)

    model = Word2Vec(vocab_size, cfg["embed_dim"])
    print(f"Model: {vocab_size} words x {cfg['embed_dim']} dims\n")

    # training loop
    total_pairs_trained = 0
    lr0 = cfg["lr"]
    num_epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    K = cfg["neg_samples"]

    # ensure the output directory exists
    os.makedirs(os.path.dirname(cfg["save_path"]) or ".", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        sampled_ids = subsample(token_ids, freqs, t=cfg["subsample_t"])
        pairs = make_training_pairs(sampled_ids, window_size=cfg["window"])
        np.random.shuffle(pairs)

        pairs = np.array(pairs)
        n_pairs = len(pairs)

        epoch_loss = 0.0
        for start in range(0, n_pairs, batch_size):
            batch = pairs[start : start + batch_size]
            B = len(batch)

            progress = total_pairs_trained / (num_epochs * n_pairs + 1)
            lr = max(lr0 * (1.0 - progress), 1e-5)

            neg_ids = sample_negatives(noise_table, B * K).reshape(B, K)

            centres  = batch[:, 0]
            contexts = batch[:, 1]

            loss = model.train_batch(centres, contexts, neg_ids, lr)
            epoch_loss += loss
            total_pairs_trained += B

            # print intra-epoch progress
            done = start + B
            pct = done / n_pairs * 100
            print(f"\r  Epoch {epoch}/{num_epochs}  "
                  f"[{done:>7}/{n_pairs}] {pct:5.1f}%  "
                  f"lr {lr:.5f}", end="", flush=True)

        print()  # newline after the progress line
        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(n_pairs, 1)
        print(f"Epoch {epoch:>3}/{num_epochs}  |  "
              f"pairs {n_pairs:>6}  |  "
              f"avg loss {avg_loss:.4f}  |  "
              f"lr {lr:.5f}  |  "
              f"{elapsed:.1f}s")

        # ── save checkpoint after every epoch ──
        ckpt_path = epoch_save_path(cfg["save_path"], epoch)
        model.save(ckpt_path)
        print(f"  -> saved checkpoint: {ckpt_path}")

    print("Training is done!")

    print("knn of example words")
    query_words = ["king", "queen", "man", "woman", "prince", "boy"]
    for w in query_words:
        if w not in word2idx:
            continue
        wid = word2idx[w]
        neighbours = model.k_most_similar(wid, idx2word, top_k=5)
        neighbour_str = ", ".join(f"{word} ({sim:.3f})" for word, sim in neighbours)
        print(f"  {w:>10}  ->  {neighbour_str}")

    # final save (same as last checkpoint, but at the canonical path)
    model.save(cfg["save_path"])
    print(f"Final model saved to: {cfg['save_path']}")


if __name__ == "__main__":
    main()
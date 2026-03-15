from src.utils import sigmoid
import numpy as np


class Word2Vec:
    def __init__(self, vocab_size, embed_dim):
        """
        Initialise embedding matrices with small random values.
        W_in (V x D) word embeddings 
        W_out (V x D) context embeddings
        """
        self.V = vocab_size
        self.D = embed_dim

        # we intialize based on the original implementation of word2vec
        # W_in from uniform(-0.5, 0.5) and scale by 1/D to keep initial dot products small
        # W_out all zeros
        
        self.W_in = np.random.uniform(-0.5, 0.5, (self.V, self.D)) / self.D
        self.W_out = np.zeros((self.V, self.D))

    def train_pair(self, centre_id, context_id, neg_ids, lr):
        """
        Train a single (centre, context) pair with K negative samples
        """
        # forward pass
        v_c = self.W_in[centre_id]              # central vector (D,)
        v_pos = self.W_out[context_id]          # positive context (D,)
        v_neg = self.W_out[neg_ids]             # negative context, K samples (K, D)

        dot_pos = np.dot(v_pos, v_c)            # scalar
        dot_neg = v_neg @ v_c                   # (K,)

        sig_pos = sigmoid(np.array([dot_pos]))[0]
        sig_neg = sigmoid(dot_neg)              # (K,)

        # loss function, two parts for positive and negative samples
        # eps to avoid log(0)
        eps = 1e-7
        loss = -np.log(sig_pos + eps) - np.sum(np.log(1.0 - sig_neg + eps))


        # basic maths to compute gradients against v_c, v_pos and v_neg
        # derivative of log(sigmoid(x)) is (1 - sigmoid(x))
        
        # v_c
        grad_v_c = -(1.0 - sig_pos) * v_pos    
        grad_v_c += (sig_neg[:, None] * v_neg).sum(axis=0) 

        # v_pos
        grad_v_pos = -(1.0 - sig_pos) * v_c

        # v_neg
        grad_v_neg = sig_neg[:, None] * v_c[None, :]  # (K, D)

        # update weights
        self.W_in[centre_id] -= lr * grad_v_c
        self.W_out[context_id] -= lr * grad_v_pos
        self.W_out[neg_ids] -= lr * grad_v_neg

        return loss


    def train_batch(self, centres, contexts, neg_ids, lr):
        """
        Train a batch of (centre, context) pairs.
        """
        B = len(centres)
        K = neg_ids.shape[1]
 
        # look up embeddings
        v_c   = self.W_in[centres]          # (B, D)
        v_pos = self.W_out[contexts]        # (B, D)
        v_neg = self.W_out[neg_ids]         # (B, K, D)
 
        # dot products
        dot_pos = np.sum(v_c * v_pos, axis=1)                  # (B,)
        dot_neg = np.einsum('bd,bkd->bk', v_c, v_neg)         # (B, K)
 
        # sigmoids
        sig_pos = sigmoid(dot_pos)          # (B,)
        sig_neg = sigmoid(dot_neg)          # (B, K)
 
        # loss
        eps = 1e-7
        loss = -np.sum(np.log(sig_pos + eps)) - np.sum(np.log(1.0 - sig_neg + eps))
 
        # gradients
        err_pos = -(1.0 - sig_pos)                             # (B,)
        err_neg = sig_neg                                       # (B, K)
 
        # grad for centre vectors: positive part + negative part
        grad_c = err_pos[:, None] * v_pos                       # (B, D)
        grad_c += np.einsum('bk,bkd->bd', err_neg, v_neg)      # (B, D)
 
        # grad for positive context vectors
        grad_pos = err_pos[:, None] * v_c                       # (B, D)
 
        # grad for negative context vectors
        grad_neg = err_neg[:, :, None] * v_c[:, None, :]        # (B, K, D)
 
        # update weights
        # np.add.at handles duplicate indices correctly (accumulates)
        np.add.at(self.W_in,  centres,  -lr * grad_c)
        np.add.at(self.W_out, contexts, -lr * grad_pos)
        np.add.at(self.W_out, neg_ids,  -lr * grad_neg)
 
        return loss


    def k_most_similar(self, word_id, idx2word, top_k=5):
        vec = self.W_in[word_id]
        norms = np.linalg.norm(self.W_in, axis=1)
        norms = np.maximum(norms, 1e-10)
        cos = (self.W_in @ vec) / (norms * np.linalg.norm(vec))
        cos[word_id] = -1.0
        top_ids = np.argsort(cos)[::-1][:top_k]
        return [(idx2word[i], cos[i]) for i in top_ids]

    def save(self, path):
        np.savez(path, W_in=self.W_in, W_out=self.W_out)

    def load(self, path):
        data = np.load(path)
        self.W_in = data["W_in"]
        self.W_out = data["W_out"]
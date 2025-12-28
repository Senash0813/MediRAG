import numpy as np

def l2_normalize(vec):
    return vec / np.linalg.norm(vec)

def fuse_embeddings(proj_query, hyde_query, alpha=0.5):
    fused = alpha * proj_query + (1 - alpha) * hyde_query
    return l2_normalize(fused)
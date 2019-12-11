import numpy as np
import pandas as pd
import random

def get_song_bool_list(songs, song_pl_df):
    """
    given a list of songs as titles, convert to a binary list of in vs not in playlist
    
    INPUT:
    songs: list or array of song titles from those found in 100_Sample_MilPlay_Spotify
    song_pl_df: Sparse matrix of which songs are in which playlist
    
    RETURNS:
    seed_play: list of 1's at indexes corresponding to given songs
    """
    seed_play = np.zeros(len(list(song_pl_df.columns)))
    song_idx = [list(song_pl_df.columns).index(s) for s in songs]
    seed_play[song_idx] = 1
    return seed_play

def get_titles(song_bools, song_pl_df):
    """
    given a list of songs as bools for in or not in playlist, convert to song titles
    
    INPUT:
    song_bools: list or array of bools corresponding to if each song is present in the "playlist"
    song_pl_df: Sparse matrix of which songs are in which playlist
    
    RETURNS:
    list of song titles
    """
    return [list(song_pl_df.columns)[i] for i, x in enumerate(song_bools) if x == 1]
  
def remove_songs(l, n):
    """
    given a list, set n (proportion) random elements to 0
    
    INPUT:
    l: list to "remove" elements from
    n: proportion of elements to "remove"
    
    RETURNS:
    new_array: list with n random elements set to 0
    """
    l = np.array(l)
    hits = np.nonzero(l)[0]

    num_keep = int((1-n)*len(hits))

    idx_keep = random.sample(list(hits), num_keep)

    new_array = np.zeros(len(l))
    new_array[idx_keep] = 1
    return new_array
  
  
# Functions from other sources
# NDCG calculation functions, source: https://gist.github.com/bwhite/3726239

def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
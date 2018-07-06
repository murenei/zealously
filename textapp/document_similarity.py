#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 20:45:08 2018

@author: richardmurray

Functions for Document Similarity:
    - converting POS tags from nltk to WordNet
    - converting tokens from a document into WordNet Synsets (groups of synonyms)
    - calculating the similarity score between two sets of Synsets
    - calculating the similarity between two documents

Can use these functions to:
    - identify paraphrasing
    - measure label accuracy

"""

# import numpy as np
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
# import pandas as pd


def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""

    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:

        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """

    # Your Code Here
    words = word_tokenize(doc)
    tags = pos_tag(words)
    synset_tags = [(w, convert_tag(t)) for w, t in tags]

    synsets_1 = []
    for w, t in synset_tags:
        try:
            synsets_1.append(wn.synsets(w, t)[0])
        except:
            continue

    return synsets_1  # Your Answer Here


def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """

    s = []
    for w1 in s1:
        scores = [ps for ps in [w1.path_similarity(w2) for w2 in s2] if ps is not None]
        if scores:
            s.append(max(scores))
    return sum(s) / len(s)


def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2

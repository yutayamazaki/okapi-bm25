import math
from typing import Callable, Dict, List, Tuple

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm: Callable = np.linalg.norm
    return float(np.dot(a, b) / (norm(a) * norm(b)))


class OkapiBM25:

    def __init__(self, tokenizer: Callable, k: float = 1.2, b: float = 0.75):
        self.tokenizer: Callable = tokenizer
        self.k: float = k
        self.b: float = b

    @staticmethod
    def _create_idf_dict(
        docs: List[str], tokenize: Callable, smooth: float = 0.5
    ) -> Dict[str, float]:
        num_docs: int = len(docs)
        token2count: Dict[str, int] = {}
        for doc in docs:
            tokens: List[str] = tokenize(doc)
            for token in set(tokens):
                if token in token2count.keys():
                    token2count[token] += 1
                else:
                    token2count[token] = 1

        token2idf: Dict[str, float] = {}
        for token, count in token2count.items():
            idf: float = math.log(num_docs - count + smooth / count + smooth)
            token2idf[token] = idf

        return token2idf

    @staticmethod
    def _get_token_frequency(tokens: List[str]) -> Dict[str, float]:
        token2freq: Dict[str, float] = {}
        token_size: int = len(tokens)
        for token in tokens:
            if token in token2freq.keys():
                token2freq[token] += 1. / token_size
            else:
                token2freq[token] = 1. / token_size
        return token2freq

    @staticmethod
    def _get_avgdl(docs: List[str], tokenizer: Callable) -> float:
        return np.mean([len(tokenizer(doc)) for doc in docs])

    def fit(self, X: List[str], y=None):
        self.token2idf_: Dict[str, float] = \
            self._create_idf_dict(X, self.tokenizer)
        self.avgdl_: float = self._get_avgdl(X, self.tokenizer)

    def calc_score(self, doc: str, query: str) -> float:
        query_tokens: List[str] = self.tokenizer(query)
        doc_tokens: List[str] = self.tokenizer(doc)

        num_tokens: int = len(doc_tokens)
        token2freq: Dict[str, float] = self._get_token_frequency(doc_tokens)

        unique_query_token: List[str] = list(set(query_tokens))
        score: float = 0.0
        k: float = self.k
        b: float = self.b
        avgdl: float = self.avgdl_
        for query_token in unique_query_token:
            idf: float = 0.0
            if query_token in self.token2idf_.keys():
                idf = self.token2idf_[query_token]

            freq: float = 0.0
            if query_token in token2freq.keys():
                freq = token2freq[query_token]

            denominator: float = freq + k * (1 - b + b * num_tokens / avgdl)
            numerator: float = idf * freq * (k + 1)
            score += numerator / denominator
        return score

import unittest

import numpy as np

from okapi_bm25 import OkapiBM25
from okapi_bm25.bm25 import cosine_similarity


def tokenize(text: str):
    return text.split(' ')


class CosineSimilarityTests(unittest.TestCase):

    def test_simple(self):
        a: np.ndarray = np.array([1, 1])
        b: np.ndarray = np.array([1, 1])
        c: np.ndarray = np.array([-1, -1])

        sim: float = cosine_similarity(a, b)
        self.assertIsInstance(sim, float)
        self.assertAlmostEqual(sim, 1.0)

        sim: float = cosine_similarity(a, c)
        self.assertIsInstance(sim, float)
        self.assertAlmostEqual(sim, -1.0)


class BM25Tests(unittest.TestCase):

    def setUp(self):
        self.docs: List[str] = ['I am a pen', 'I']

    def test_create_idf_dict(self):
        bm25 = OkapiBM25(tokenize)
        tokenized_docs: List[List[str]] = [tokenize(x) for x in self.docs]
        token2idf: Dict[str, float] = bm25._create_idf_dict(tokenized_docs)
        expected: Dict[str, float] = {
            'a': 0.6931471805599453,
            'pen': 0.6931471805599453,
            'I': -0.2876820724517809,
            'am': 0.6931471805599453
        }
        self.assertDictEqual(token2idf, expected)

    def test_token_frequency(self):
        bm25 = OkapiBM25(tokenize)
        tokens: List[str] = ['I', 'am', 'I']
        token2freq: Dict[str, float] = bm25._get_token_frequency(tokens)

        expected: Dict[str, float] = {
            'I': 0.6666666666666666, 'am': 0.3333333333333333
        }
        self.assertDictEqual(token2freq, expected)

    def test_get_avgdl(self):
        bm25 = OkapiBM25(tokenize)
        tokenized_docs: List[List[str]] = [tokenize(x) for x in self.docs]

        avgdl: float = bm25._get_avgdl(tokenized_docs)
        self.assertAlmostEqual(avgdl, 2.5)

    def test_fit(self):
        bm25 = OkapiBM25(tokenize)
        ret: OkapiBM25 = bm25.fit(self.docs)

        self.assertIsInstance(ret, OkapiBM25)
        self.assertEqual(ret.avgdl_, 2.5)
        expected: Dict[str, float] = {
            'a': 0.6931471805599453,
            'pen': 0.6931471805599453,
            'I': -0.2876820724517809,
            'am': 0.6931471805599453
        }
        self.assertDictEqual(ret.token2idf_, expected)

    def test_calc_score(self):
        bm25 = OkapiBM25(tokenize)
        bm25.fit(self.docs)

        score: float = bm25.calc_score('I am a dog', 'I have a pen')
        self.assertEqual(score, 0.11206322083391479)

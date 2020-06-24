import unittest

from okapi_bm25 import OkapiBM25


def tokenize(text: str):
    return text.split(' ')


class BM25Tests(unittest.TestCase):

    def setUp(self):
        self.docs: List[str] = ['I am a pen', 'I']

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

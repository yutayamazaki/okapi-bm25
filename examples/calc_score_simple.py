from typing import List

from okapi_bm25 import OkapiBM25


def tokenizer(text: str) -> List[str]:
    return text.split(' ')


if __name__ == '__main__':
    docs: List[str] = [
        '私 の 名前 は 田中 です',
        '君 の 名前 は 何 です か',
        '今日 は 良い 天気 です ね'
    ]
    bm25: OkapiBM25 = OkapiBM25(tokenizer)
    bm25.fit(docs)

    doc: str = '今日 は 寒い です ね'
    query1: str = '私 は 田中 でした'
    query2: str = '今日 は 良い 天気 です ね'
    q1_score: float = bm25.calc_score(doc, query1)
    q2_score: float = bm25.calc_score(doc, query2)

    print(f'doc: {doc}')
    print(f'\t score: {q1_score:.5f}, query: {query1}')
    print(f'\t score: {q2_score:.5f}, query: {query2}')

    """
    doc: 今日 は 寒い です ね
        score: -0.14738, query: 私 は 田中 でした
        score: 0.50389, query: 今日 は 良い 天気 です ね
    """

# Make sure start elasticsearch before run this
import requests
import json
import os
from typing import List
from tqdm.autonotebook import tqdm
from sentence_transformers import SentenceTransformer
from src.utils import (print_message, word_segment, bm25_tokenizer, combined_score_mm, sigmoid)
from src.service.config import URL_ELASTICSEARCH, MODEL_PATH
from elasticsearch import Elasticsearch, helpers

print_message("#> Load Encoder...!")
ENCODER = SentenceTransformer(MODEL_PATH, device="cuda")
ES = Elasticsearch(URL_ELASTICSEARCH)


def create_hybrid_index(index: str, bm25_k1: float = 0.5,
                        bm25_b: float = 0.5, dims: int = 768,
                        m_hnsw: int = 32, ef_construction: int = 128,
                        similarity: str = 'dot_product'):
    url = os.path.join(URL_ELASTICSEARCH, index)

    payload = json.dumps({
        'settings': {
            'similarity': {
                'bm25_similarity': {
                    'type': 'BM25',
                    'k1': bm25_k1,
                    'b': bm25_b
                }
            }
        },
        'mappings': {
            '_source': {
                'includes': [
                    'context',
                    'segment_ctx'
                ],
                'excludes': [
                    'bm25_text'
                    'embedding'
                ]
            },
            'properties': {
                'context': {
                    'type': 'object',
                    'enabled': 'false'
                },
                'segment_ctx': {
                    'type': 'object',
                    'enabled': 'false'
                },
                'bm25_text': {
                    'type': 'text',
                    'similarity': 'bm25_similarity'
                },
                'embedding': {
                    'type': 'dense_vector',
                    'dims': dims,
                    'index': 'true',
                    'similarity': similarity,
                    'index_options': {
                        'type': 'hnsw',
                        'm': m_hnsw,
                        'ef_construction': ef_construction
                    }

                }
            }
        }
    })
    headers = {'Content-Type': 'application/json'}

    response = requests.request("PUT", url, headers=headers, data=payload)
    return response


def add_docs_to_index(index: str, add_docs: List, chunk_size: int = 1000):
    print_message("#> Start adding...!")
    with tqdm(total=len(add_docs)) as pbar:
        idx = 0
        for start_idx in range(0, len(add_docs), chunk_size):
            end_idx = start_idx + chunk_size
            sub_docs = add_docs[start_idx:end_idx]
            segment_docs = []
            bm25_docs = []
            # process input text to bm25 and model format
            for doc in sub_docs:
                title = word_segment(doc["passage_title"])
                content = word_segment(doc["passage_content"])
                segment_docs.append({"passage_title": title, "passage_content": content})
                bm25_docs.append(bm25_tokenizer(title + " " + content))
            # encode doc to embedding
            print_message("#> Encode corpus...!")
            encode_docs = [doc["passage_title"] + " . " + doc["passage_content"] for doc in segment_docs]
            embeddings = ENCODER.encode(encode_docs, convert_to_numpy=True,
                                        normalize_embeddings=True, show_progress_bar=False)
            # add all features to index
            print_message("#> Add docs to index...!")
            bulk_data = []
            for doc, doc_segment, bm25_doc, embedding in zip(sub_docs, segment_docs, bm25_docs, embeddings):
                bulk_data.append({
                    '_id': idx,
                    'context': doc,
                    'segment_ctx': doc_segment,
                    'bm25_text': bm25_doc,
                    'embedding': embedding
                })
                idx += 1
            helpers.bulk(ES, bulk_data, index=index)
            pbar.update(chunk_size)
    ES.indices.refresh(index=index)


def remove_docs_from_index(index: str, id_docs: List):
    url = os.path.join(index, "_delete_by_query")

    payload = json.dumps({
        "query": {
            "terms": {
                "_id": id_docs
            }
        }
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.text


def hybird_search(index: str, query: str, top_k: int = 50, rank: bool = True):
    # question = word_segment(query)
    question = query
    question_embedding = ENCODER.encode(question, show_progress_bar=False, normalize_embeddings=True)
    lex_search = ES.search(index=index, size=50,
                           query={"match": {"bm25_text": bm25_tokenizer(question)}}, source=False)
    sparse_result = {hit["_id"]: hit["_score"] for hit in lex_search["hits"]["hits"]}
    sem_search = ES.search(index=index, size=top_k, knn={"field": "embedding", "query_vector": question_embedding,
                                                         "k": 50, "num_candidates": 256}, source=False)
    dense_result = {hit["_id"]: hit["_score"] for hit in sem_search["hits"]["hits"]}
    hybrid_result = combined_score_mm(sparse_result, dense_result, top_k=top_k)
    res_ids = [res[0] for res in hybrid_result]
    result = ES.search(index=index, query={"terms": {"_id": res_ids}})["hits"]["hits"]
    result_segment = []   # return result for call api hard negative mining
    result_for_rank = []  # return result for ranking stage
    result_original = {}  # return original text
    for res in result:
        doc_id = res["_id"]
        text = res["_source"]["segment_ctx"]
        text_rank = text["passage_title"] + " . " + text["passage_title"]
        result_for_rank.append((doc_id, text_rank))
        result_segment.append(text)
        result_original[str(doc_id)] = res["_source"]["context"]
    if rank is True:
        return question, result_for_rank, result_original
    return question, result_segment


def get_original_passages(doc_ids, result_original, scores=None):
    results = [result_original[str(idx)] for idx in doc_ids]
    if scores is not None:
        nor_scores = [sigmoid(float(score)) for score in scores]
        return results, nor_scores
    else:
        return results

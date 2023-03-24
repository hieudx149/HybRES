import datetime
import py_vncorenlp
import string
import unicodedata as ud
import os
import numpy as np
from typing import Dict


dir_file_path = os.path.dirname(os.path.realpath(__file__))
vncorenlp_path = os.path.join(dir_file_path, "vncorenlp")
print(vncorenlp_path)
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_path)

vi_stop_word = ["như", "làm", "là", "và", "với", "nếu", "thì", "do", "ở", "đây", "đó", "lại", "không", "nhỉ", "ta",
                "cho", "chung", "đã", "nơi", "để", "đến", "số", "một", "khác", "được", "vào", "ra", "trong", "ạ",
                "người", "loài", "từ", "nào", "bằng", "rằng", "nên", "gì", "việc", "ấy", "khi", "này", "chỉ", "về",
                "các", "còn", "trên", "những", "có", "mà", "nhưng", "nhiều", "nó", "sẽ", "chưa", "lúc", "có_thể",
                "bởi_vì", "tại_vì", "như_thế", "thế_là", "trong_khi", "thế_mà", "chẳng_hạn", "do_đó", "tuy_nhiên",
                "đôi_khi", "chỉ_là", "một_số", "chúng_nó", "rằng_là", "tôi", "</s>", "...", "–", "ơi"]

question_start_terms = ["Muốn", "Tôi muốn", "Mình muốn", "Anh muốn", "Tớ muốn", "Chị muốn", "Em muốn", "Cho hỏi",
                        "Cho hỏi", "Cho tớ hỏi", "Cho mình hỏi", "Cho tớ hỏi", "Cho em hỏi", "Cho tôi hỏi",
                        "Cho anh hỏi", "Bạn ơi cho hỏi", "AMI ơi cho mình hỏi", "Bạn ơi cho mình hỏi"]

question_end_terms = ["nhỉ", " ạ", "thế"]


def word_segment(text):
    text = normalize(text)
    return " ".join(rdrsegmenter.word_segment(text))


def bm25_tokenizer(text):
    """Pre-processing input for bm25 search"""
    tokens = text.split()
    tokens = list(map(lower_case, tokens))
    tokens = list(filter(remove_punctuation, tokens))
    tokens = list(filter(remove_stop_word, tokens))
    return " ".join(tokens)


def normalize(text: str):
    """Normalize passage text"""
    text = ud.normalize("NFC", text)
    text = " ".join(text.split())
    text = text.replace("‘", "'")
    text = text.replace("”", '"')
    text = text.replace("“", '"')
    text = text.replace("′", "'")
    return text.strip()


def question_normalize(question: str):
    """Normalize question input with start and end terms"""
    question = question[0].upper() + question[1:]
    for start_term in question_start_terms:
        if question.startswith(start_term):
            question = question.replace(start_term, "")
            break
    for end_term in question_end_terms:
        if question.endswith(end_term):
            question = question.replace(end_term, "")
            break
    return question[0].upper() + question[1:]


def lower_case(w):
    return w.lower()


def remove_stop_word(w):
    return w not in vi_stop_word


def remove_punctuation(w):
    return w not in string.punctuation


def sigmoid(x, temp=10):
    x = x / temp
    return 1/(1 + np.exp(-x))


def print_message(*s, condition=True):
    s = ' '.join([str(x) for x in s])
    msg = "[{}] {}".format(
        datetime.datetime.now().strftime("%b %d, %H:%M:%S"), s)

    if condition:
        print(msg, flush=True)

    return msg


def combined_score_mm(sparse_result: Dict, dense_result: Dict, top_k: int = 100, alpha=0.6):
    """Combined score between BM25 and Dense, this function was inspired by pyserini"""
    hybrid_result = {}
    min_dense_score = min(dense_result.values()) if len(dense_result) > 1 else 0
    max_dense_score = max(dense_result.values()) if len(dense_result) > 1 else 1
    min_sparse_score = min(sparse_result.values()) if len(sparse_result) > 1 else 0
    max_sparse_score = max(sparse_result.values()) if len(sparse_result) > 1 else 1
    for psg in set(dense_result.keys()) | set(sparse_result.keys()):
        if psg not in dense_result:
            sparse_score = sparse_result[psg]
            dense_score = min_dense_score
        elif psg not in sparse_result:
            sparse_score = min_sparse_score
            dense_score = dense_result[psg]
        else:
            sparse_score = sparse_result[psg]
            dense_score = dense_result[psg]
        sparse_score = (sparse_score - min_sparse_score) / (max_sparse_score - min_sparse_score)
        dense_score = (dense_score - min_dense_score) / (max_dense_score - min_dense_score)
        score = alpha * dense_score + (1 - alpha) * sparse_score
        hybrid_result[psg] = (score, sparse_score, dense_score)
    return sorted(hybrid_result.items(), key=lambda x: x[1][0], reverse=True)[:top_k]

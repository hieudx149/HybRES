from flask_cors import CORS
import flask
import logging
import coloredlogs
import traceback
import copy
import requests
import time
from src.service.config import SERVICE_HOST, SERVICE_PORT, URL_RERANKING
from src.elastic_apis import (hybird_search, get_original_passages,
                              add_docs_to_index, create_hybrid_index,
                              remove_docs_from_index)

coloredlogs.install(
    level="INFO", fmt="%(asctime)s %(name)s[%(process)d] %(levelname)-8s %(message)s"
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = flask.Flask(__name__)
CORS(app)
VERSION = "1.15"
RESPONSE_SEARCH = {"status": "success", "code": 200, "knowledge_retrieval": [], "version": VERSION}
# RESPONSE_RETRIEVAL_ONLY = {"status": "success", "code": 200, "knowledge_retrieval": [], "version": VERSION}
RESPONSE = {"status": "success", "code": 200, "set_variables": {}, "bot_responses": [], "version": VERSION}
# RESPONSE_UPDATE = {"code": 200, "message": "1", "status": "success", "version": VERSION}


@app.route("/health", methods=["POST", "GET"])
def health():
    """get health."""
    res = copy.deepcopy(RESPONSE)
    try:
        res["set_variables"]["api_status"] = "1"
        res["set_variables"]["version_debug"] = "2.0"
    except Exception as e:
        logger.critical(traceback.format_exc())
        res["set_variables"]["api_status"] = "0"
        res.update(
            status="Failure",
            code=500,
            reason="INTERNAL SERVER ERROR",
            message=f"{str(e)}"
        )
        return flask.jsonify(res), 500
    return flask.jsonify(res), 200


@app.route("/create_index", methods=["POST"])
def create_index():
    """get health."""
    res = copy.deepcopy(RESPONSE)
    try:
        data = flask.request.get_json(force=True)
        index = data.get("index")
        create_hybrid_index(index=index)
    except Exception as e:
        logger.critical(traceback.format_exc())
        res["set_variables"]["api_status"] = "0"
        res.update(
            status="Failure",
            code=500,
            reason="INTERNAL SERVER ERROR",
            message=f"{str(e)}"
        )
        return flask.jsonify(res), 500
    return flask.jsonify(res), 200


@app.route("/search", methods=["POST"])
def search():
    """search top-n documents relevant from wikipedia corpus"""
    res = copy.deepcopy(RESPONSE_SEARCH)
    try:
        data = flask.request.get_json(force=True)
        query = data.get("query")
        index = data.get("index")
        top_n_retrieval = data.get("top_n_retrieval")
        top_n_reranking = data.get("top_n_reranking")
        logger.info("#> Run Retrieval !!!")
        start = time.time()
        query, candidates, hybrid_res = hybird_search(index=index, query=query,
                                                      top_k=int(top_n_retrieval))
        res["time_retrieval"] = str(time.time() - start) + "s"
        input_reranking = {
            "query": query,
            "candidates": candidates,
            "top_n_reranking": str(top_n_reranking)
        }
        logger.info("#> Sending request !!!")
        start = time.time()
        request = requests.post(URL_RERANKING, json=input_reranking)
        res["time_reranking"] = str(time.time() - start) + "s"
        if request.status_code == 200:
            response = request.json()
            doc_ids = response["document_ids"]
            scores = response["score_ranking"]
            doc_texts, scores = get_original_passages(doc_ids=doc_ids, result_original=hybrid_res, scores=scores)
        else:
            doc_ids = [item[0] for item in candidates[:top_n_reranking]]
            scores = None
            doc_texts = get_original_passages(doc_ids, hybrid_res)
        if scores is not None:
            assert len(scores) == len(doc_texts)
            for score, doc in zip(scores, doc_texts):
                doc["score_ranking"] = str(score)
        res["code"] = 200
        res["status"] = "success"
        res["knowledge_retrieval"] = doc_texts
    except Exception as e:
        logger.critical(traceback.format_exc())
        res.update(
            status="INTERNAL SERVER ERROR",
            code=500,
            message=f"{str(e)}"
        )
        return flask.jsonify(res), 500
    return flask.jsonify(res), 200


@app.route("/retrieval_only", methods=["POST"])
def retrieval_only():
    """search top-n documents relevant from wikipedia corpus"""
    res = copy.deepcopy(RESPONSE_SEARCH)
    try:
        data = flask.request.get_json(force=True)
        query = data.get("query")
        index = data.get("index")
        top_n_retrieval = data.get("top_n_retrieval")
        print(top_n_retrieval)
        logger.info("#> Run Retrieval !!!")
        start = time.time()
        query, result = hybird_search(index=index, query=query, top_k=int(top_n_retrieval), rank=False)
        res["time_retrieval"] = str(time.time() - start) + "s"
        res["code"] = 200
        res["status"] = "success"
        res["query"] = query
        res["knowledge_retrieval"] = result
    except Exception as e:
        logger.critical(traceback.format_exc())
        res.update(
            status="INTERNAL SERVER ERROR",
            code=500,
            message=f"{str(e)}"
        )
        return flask.jsonify(res), 500
    return flask.jsonify(res), 200


@app.route("/add_docs", methods=["POST"])
def add_docs():
    """get health."""
    res = copy.deepcopy(RESPONSE)
    try:
        data = flask.request.get_json(force=True)
        documents = data.get("knowledge")
        index = data.get("index")
        chunk_size = data.get("chunk_size")
        add_docs_to_index(index=index, add_docs=documents, chunk_size=chunk_size)
    except Exception as e:
        logger.critical(traceback.format_exc())
        res["set_variables"]["api_status"] = "0"
        res.update(
            status="Failure",
            code=500,
            reason="INTERNAL SERVER ERROR",
            message=f"{str(e)}"
        )
        return flask.jsonify(res), 500
    return flask.jsonify(res), 200


@app.route("/remove_docs", methods=["POST"])
def remove_docs():
    """get health."""
    res = copy.deepcopy(RESPONSE)
    try:
        data = flask.request.get_json(force=True)
        doc_ids = data.get("doc_ids")
        index = data.get("index")
        remove_docs_from_index(index=index, id_docs=doc_ids)
    except Exception as e:
        logger.critical(traceback.format_exc())
        res["set_variables"]["api_status"] = "0"
        res.update(
            status="Failure",
            code=500,
            reason="INTERNAL SERVER ERROR",
            message=f"{str(e)}"
        )
        return flask.jsonify(res), 500
    return flask.jsonify(res), 200


if __name__ == "__main__":
    app.run(host=str(SERVICE_HOST), port=int(SERVICE_PORT))

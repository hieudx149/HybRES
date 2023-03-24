"""Config."""
import os
import yaml
from src import get_root_path

ROOT_PATH = get_root_path()
CONFIG_PATH = os.path.join(ROOT_PATH, "service", "config_service.yml")
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

SERVICE_PORT = os.environ.get("SERVICE_PORT") or config["SERVICE_PORT"]
SERVICE_HOST = os.environ.get("SERVICE_HOST") or config["SERVICE_HOST"]
URL_RERANKING = os.environ.get("URL_RERANKING") or config["URL_RERANKING"]
URL_ELASTICSEARCH = os.environ.get("URL_ELASTICSEARCH") or config["URL_ELASTICSEARCH"]
# GPU_MEM_LIMIT = os.environ.get("GPU_MEM_LIMIT") or config.get("GPU_MEM_LIMIT")

NO_CUDA = os.environ.get("NO_CUDA") or config["NO_CUDA"]

BM25_K1 = 0.5
BM25_B = 0.5
VNCORENLP_PATH = os.path.join(ROOT_PATH, "vncorenlp")
MODEL_PATH = os.path.join(ROOT_PATH, "checkpoint-2022-06-28_04-25-02")

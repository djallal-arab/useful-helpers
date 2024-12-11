from typing import List, Dict, Union, Optional, Any
import numpy as np
import chromadb
import torch
from sklearn.preprocessing import normalize
import logging
import asyncio
import json
import time
from functools import wraps
import yaml

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Version of the library
__version__ = "0.1.4"


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute.")
        return result

    return wrapper


class ChromaDBHelperConfig:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        self.collection_name = config['collection_name']
        self.persistence_directory = config.get('persistence_directory', './data/chroma_db')
        self.distance_metric = config.get('distance_metric', 'cosine')
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1)
        self.batch_size = config.get('batch_size', 1000)
        self.normalize_vector = config.get('normalize_vector', True)


class ChromaDBHelper:
    def __init__(self, config: ChromaDBHelperConfig):
        self.config = config
        self.client = chromadb.PersistentClient(path=self.config.persistence_directory)
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self, construction_ef=200, M=32, search_ef=20):
        for attempt in range(self.config.max_retries):
            try:
                return self.client.get_or_create_collection(
                    name=self.config.collection_name,
                    metadata={
                        "hnsw:space": self.config.distance_metric,
                        "hnsw:construction_ef": construction_ef,
                        "hnsw:M": M,
                        "hnsw:search_ef": search_ef,
                    }
                )
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed to get or create collection after {self.config.max_retries} attempts.")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed. Retrying in {self.config.retry_delay} seconds...")
                time.sleep(self.config.retry_delay)

    def _process_vector(self, vector: np.ndarray) -> np.ndarray:
        if self.config.normalize_vector:
            return normalize(vector.reshape(1, -1)).flatten()
        else:
            return vector

    @timer
    def add_record(self, record_id: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        processed_vector = self._process_vector(vector)
        metadata = metadata or {}

        try:
            self.collection.add(ids=[record_id], embeddings=[processed_vector.tolist()], metadatas=[metadata])
            logger.info(f"Added record {record_id} to the collection.")
        except Exception as e:
            logger.error(f"Failed to add record {record_id}: {str(e)}")
            raise

    @timer
    def add_record_tensor(self, record_id: str, vector: torch.Tensor, metadata: Optional[Dict[str, Any]] = None):
        processed_vector = self._process_vector(vector.cpu().numpy())
        metadata = metadata or {}

        try:
            self.collection.add(ids=[record_id], embeddings=[processed_vector.tolist()], metadatas=[metadata])
            logger.info(f"Added record {record_id} to the collection.")
        except Exception as e:
            logger.error(f"Failed to add record {record_id}: {str(e)}")
            raise

    @timer
    def add_records_batch(self, records: List[Dict[str, Union[str, np.ndarray, Dict]]]):
        ids = [record['id'] for record in records]
        vectors = [self._process_vector(record['vector']) for record in records]
        metadata = [record.get('metadata', {}) for record in records]

        for i in range(0, len(ids), self.config.batch_size):
            batch_ids = ids[i:i + self.config.batch_size]
            batch_vectors = vectors[i:i + self.config.batch_size]
            batch_metadata = metadata[i:i + self.config.batch_size]

            try:
                self.collection.add(ids=batch_ids, embeddings=[v.tolist() for v in batch_vectors],
                                    metadatas=batch_metadata)
                logger.info(f"Added batch of {len(batch_ids)} records to the collection.")
            except Exception as e:
                logger.error(f"Failed to add batch: {str(e)}")
                raise

    @timer
    def find_similar(self, query_vector: np.ndarray, n_results: int = 5, filters: Optional[Dict] = None,
                     distance_threshold: Optional[float] = None) -> List[Dict]:
        processed_query = self._process_vector(query_vector)
        results = self.collection.query(query_embeddings=[processed_query.tolist()], n_results=n_results,
                                        include= ["metadatas", "distances"],
                                        where=filters)

        output = []
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        for i in range(len(ids)):
            if (distance_threshold is not None) and (distances[i] > distance_threshold):
                continue
            element ={
                "id": ids[i],
                "distance": distances[i],
                "metadata": metadatas[i]
            }
            output.append(element)

        logger.info(f"Retrieved {len(output)} similar records.")
        return output

    @timer
    def delete_record(self, record_id: str):
        try:
            self.collection.delete(ids=[record_id])
            logger.info(f"Deleted record {record_id}.")
        except Exception as e:
            logger.error(f"Failed to delete record {record_id}: {str(e)}")
            raise

    @timer
    def get_all_records(self, include_embeddings=True) -> List[Dict]:
        try:
            if include_embeddings:
                results = self.collection.get(include=["metadatas", "embeddings"])
                return [{"id": vid, "embedding": embedding, "metadata": metadata}
                        for vid, embedding, metadata in zip(results["ids"], results["embeddings"], results["metadatas"])]
            else:
                results = self.collection.get(include=["metadatas"])
                return [{"id": vid, "metadata": metadata}
                        for vid, metadata in zip(results["ids"], results["metadatas"])]
        except Exception as e:
            logger.error(f"Failed to get all records: {str(e)}")
            raise

    @timer
    def export_collection(self, file_path: str):
        try:
            records = self.get_all_records()
            with open(file_path, 'w') as f:
                json.dump(records, f)
            logger.info(f"Exported collection to {file_path}.")
        except Exception as e:
            logger.error(f"Failed to export collection: {str(e)}")
            raise

    @timer
    def import_collection(self, file_path: str):
        try:
            with open(file_path, 'r') as f:
                records = json.load(f)
            self.add_records_batch(records)
            logger.info(f"Imported collection from {file_path}.")
        except Exception as e:
            logger.error(f"Failed to import collection: {str(e)}")
            raise

    async def add_record_async(self, record_id: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        await asyncio.to_thread(self.add_record, record_id, vector, metadata)

    async def find_similar_async(self, query_vector: np.ndarray, n_results: int = 5, filters: Optional[Dict] = None,
                                 similarity_threshold: Optional[float] = None) -> List[Dict]:
        return await asyncio.to_thread(self.find_similar, query_vector, n_results, filters, similarity_threshold)

    def load(self):
        """
        Explicitly load the collection state from disk.
        """
        try:
            self.client = chromadb.PersistentClient(path=self.config.persistence_directory)
            self.collection = self._get_or_create_collection()
            logger.info("Collection state loaded from disk.")
        except Exception as e:
            logger.error(f"Failed to load collection state: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    config = ChromaDBHelperConfig('chromadb_config.yaml')
    chromadb_helper = ChromaDBHelper(config)

    # Add a record
    vector = np.random.rand(1280*13)
    chromadb_helper.add_record("example_id", vector, {"key": "value"})

    # Find similar records
    similar_records = chromadb_helper.find_similar(vector, n_results=5)
    print("Similar records:", similar_records)

    # Load the state (simulating a restart)
    chromadb_helper.load()

    # Verify the record is still there
    all_records = chromadb_helper.get_all_records()
    print("All records after loading:", all_records)

from __future__ import annotations

import asyncio
from hashlib import md5
from typing import Any, Dict, List, Optional, Union

try:
    from meilisearch import Client
    from meilisearch.errors import MeilisearchApiError
except ImportError:
    raise ImportError("`meilisearch` not installed. Please install using `pip install meilisearch`")

from agno.filters import FilterExpr
from agno.knowledge.document import Document
from agno.knowledge.embedder import Embedder
from agno.knowledge.reranker.base import Reranker
from agno.utils.log import log_debug, log_error, log_info, log_warning
from agno.utils.string import generate_id
from agno.vectordb.base import VectorDb
from agno.vectordb.distance import Distance
from agno.vectordb.search import SearchType


class Meilisearch(VectorDb):
    """
    Meilisearch class for managing vector/keyword search operations with Meilisearch.

    Args:
        index_name : The name of the index in the database.
        url : The URL of the Meilisearch instance. Defaults to "http://127.0.0.1:7700".
        api_key : The API key for authentication. Defaults to None.
        client : The Meilisearch client instance. Defaults to None.
        name : The name of the index. Defaults to None.
        description : The description of the index. Defaults to None.
        id : The ID of the index. Defaults to None.
        embedder : The embedder instance for generating embeddings. Defaults to None.
        embedder_name : The name of the embedder. Defaults to "default".
        search_type : The search type. Defaults to SearchType.vector.
        distance : The distance metric. Defaults to Distance.cosine.
        semantic_ratio : The semantic ratio for search. Defaults to 0.5.
        wait_for_tasks : Whether to wait for tasks to complete. Defaults to True.
        timeout_in_ms : The timeout in milliseconds for tasks. Defaults to 60000.
        interval_in_ms : The interval in milliseconds for polling tasks. Defaults to 50.
        reranker : The reranker instance for reranking search results. Defaults to None.
        primary_key : The primary key for the index. Defaults to "id".
        filterable_attributes : The filterable attributes for the index. Defaults to None.
        searchable_attributes : The searchable attributes for the index. Defaults to None.
    """

    def __init__(
        self,
        index_name: str,
        url: str = "http://127.0.0.1:7700",
        api_key: Optional[str] = None,
        client: Optional[Client] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        id: Optional[str] = None,
        embedder: Optional[Embedder] = None,
        embedder_name: str = "default",
        search_type: SearchType = SearchType.vector,
        distance: Distance = Distance.cosine,
        semantic_ratio: float = 0.5,
        primary_key: str = "id",
        filterable_attributes: Optional[List[str]] = None,
        searchable_attributes: Optional[List[str]] = None,
        wait_for_tasks: bool = True,
        timeout_in_ms: int = 60000,
        interval_in_ms: int = 50,
        reranker: Optional[Reranker] = None,
    ):
        if not index_name:
            raise ValueError("index_name must be provided.")
        if not embedder_name:
            raise ValueError("embedder_name must be provided.")
        if not 0.0 <= semantic_ratio <= 1.0:
            raise ValueError("semantic_ratio must be between 0.0 and 1.0.")

        if id is None:
            id = generate_id(f"{url}#{index_name}#{embedder_name}")
        super().__init__(id=id, name=name, description=description)

        if embedder is None:
            from agno.knowledge.embedder.openai import OpenAIEmbedder

            embedder = OpenAIEmbedder()
            log_debug("Embedder not provided, using OpenAIEmbedder as default.")

        if embedder.dimensions is None:
            raise ValueError("Embedder.dimensions must be set.")

        self.index_name = index_name
        self.url = url
        self.api_key = api_key
        self.client = client or Client(url, api_key)
        self.index = self.client.index(index_name)
        self.embedder = embedder
        self.dimensions = embedder.dimensions
        self.embedder_name = embedder_name
        self.search_type = search_type
        self.distance = distance
        self.semantic_ratio = semantic_ratio
        self.primary_key = primary_key
        self.wait_for_tasks = wait_for_tasks
        self.timeout_in_ms = timeout_in_ms
        self.interval_in_ms = interval_in_ms
        self.reranker = reranker

        default_filterable = [
            "id",
            "name",
            "content_hash",
            "content_id",
            "namespace",
            "user_id",
            "agent_id",
            "team_id",
            "tags",
        ]
        self.filterable_attributes = filterable_attributes or default_filterable
        self.searchable_attributes = searchable_attributes or ["name", "content"]

    def _wait_for_task(self, task_info: Any) -> None:
        if self.wait_for_tasks and task_info is not None:
            task_uid = getattr(task_info, "task_uid", None)
            if task_uid is not None:
                self.client.wait_for_task(task_uid, self.timeout_in_ms, self.interval_in_ms)

    def _ensure_settings(self) -> None:
        embedder_task = self.index.update_embedders(
            {
                self.embedder_name: {
                    "source": "userProvided",
                    "dimensions": self.dimensions,
                }
            }
        )
        self._wait_for_task(embedder_task)

        filter_task = self.index.update_filterable_attributes(self.filterable_attributes)
        self._wait_for_task(filter_task)

        searchable_task = self.index.update_searchable_attributes(self.searchable_attributes)
        self._wait_for_task(searchable_task)

    def create(self) -> None:
        if not self.exists():
            log_debug(f"Creating Meilisearch index: {self.index_name}")
            task = self.client.create_index(self.index_name, {"primaryKey": self.primary_key})
            self._wait_for_task(task)
            self.index = self.client.index(self.index_name)
        self._ensure_settings()

    async def async_create(self) -> None:
        await asyncio.to_thread(self.create)

    def exists(self) -> bool:
        try:
            self.client.get_index(self.index_name)
            return True
        except MeilisearchApiError as e:
            if e.status_code == 404:
                return False
            raise

    async def async_exists(self) -> bool:
        return await asyncio.to_thread(self.exists)

    def _format_filter_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if value is None:
            return "NULL"
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def _build_filter(self, filters: Optional[Union[str, Dict[str, Any], List[FilterExpr]]]) -> Optional[Any]:
        if filters is None:
            return None
        if isinstance(filters, str):
            return filters
        if isinstance(filters, list):
            log_warning("Filter Expressions are not supported in Meilisearch. No filters will be applied.")
            return None
        if isinstance(filters, dict):
            parts = []
            for key, value in filters.items():
                if isinstance(value, (list, tuple, set)):
                    if len(value) == 0:
                        continue
                    values = ", ".join(self._format_filter_value(item) for item in value)
                    parts.append(f"{key} IN [{values}]")
                else:
                    parts.append(f"{key} = {self._format_filter_value(value)}")
            return " AND ".join(parts) if parts else None
        log_warning(f"Unsupported filter type for Meilisearch: {type(filters)}. No filters will be applied.")
        return None

    def _prepare_vector(self, embedding: Optional[List[float]], document_name: Optional[str]) -> List[float]:
        if not embedding:
            raise ValueError(f"Document '{document_name}' has no embedding.")
        if len(embedding) != self.dimensions:
            raise ValueError(
                f"Document '{document_name}' embedding dimension mismatch: expected {self.dimensions}, got {len(embedding)}."
            )
        return [float(value) for value in embedding]

    def _document_id(self, document: Document, content_hash: str) -> str:
        cleaned_content = document.content.replace("\x00", "\ufffd")
        base_id = document.id or md5(cleaned_content.encode()).hexdigest()
        return md5(f"{base_id}_{content_hash}".encode()).hexdigest()

    def _serialize_document(
        self, document: Document, content_hash: str, filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if filters:
            meta_data = document.meta_data.copy() if document.meta_data else {}
            meta_data.update(filters)
            document.meta_data = meta_data

        if document.embedding is None or (isinstance(document.embedding, list) and len(document.embedding) == 0):
            document.embed(embedder=self.embedder)

        cleaned_content = document.content.replace("\x00", "\ufffd")
        meta_data = document.meta_data or {}
        payload = {
            "id": self._document_id(document, content_hash),
            "name": document.name,
            "content": cleaned_content,
            "content_hash": content_hash,
            "content_id": document.content_id,
            "meta_data": meta_data,
            "usage": document.usage,
            "_vectors": {self.embedder_name: self._prepare_vector(document.embedding, document.name)},
        }

        # Flatten metadata used by Agno filters so Meilisearch can index it.
        for key, value in meta_data.items():
            if key in self.filterable_attributes:
                payload[key] = value

        return payload

    def insert(self, content_hash: str, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        if len(documents) == 0:
            log_info("No documents to insert")
            return
        if not self.exists():
            self.create()

        serialized = [self._serialize_document(doc, content_hash, filters) for doc in documents]
        task = self.index.add_documents(serialized, primary_key=self.primary_key)
        self._wait_for_task(task)
        log_debug(f"Inserted {len(serialized)} documents into Meilisearch index '{self.index_name}'")

    async def async_insert(
        self, content_hash: str, documents: List[Document], filters: Optional[Dict[str, Any]] = None
    ) -> None:
        await asyncio.to_thread(self.insert, content_hash, documents, filters)

    def upsert_available(self) -> bool:
        return True

    def upsert(self, content_hash: str, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        if self.content_hash_exists(content_hash):
            self._delete_by_content_hash(content_hash)
        self.insert(content_hash=content_hash, documents=documents, filters=filters)

    async def async_upsert(
        self, content_hash: str, documents: List[Document], filters: Optional[Dict[str, Any]] = None
    ) -> None:
        await asyncio.to_thread(self.upsert, content_hash, documents, filters)

    def _query_vector(self, query: str) -> Optional[List[float]]:
        embedding = self.embedder.get_embedding(query)
        if not embedding:
            log_error(f"Error getting embedding for query: {query}")
            return None
        if len(embedding) != self.dimensions:
            log_error(f"Query embedding dimension mismatch: expected {self.dimensions}, got {len(embedding)}")
            return None
        return [float(value) for value in embedding]

    def search(
        self, query: str, limit: int = 5, filters: Optional[Union[str, Dict[str, Any], List[FilterExpr]]] = None
    ) -> List[Document]:
        if not self.exists():
            return []

        params: Dict[str, Any] = {"limit": limit}
        filter_expression = self._build_filter(filters)
        if filter_expression:
            params["filter"] = filter_expression

        search_query = query
        if self.search_type == SearchType.vector:
            query_vector = self._query_vector(query)
            if query_vector is None:
                return []
            search_query = ""
            params["vector"] = query_vector
            params["hybrid"] = {"embedder": self.embedder_name, "semanticRatio": 1.0}
        elif self.search_type == SearchType.hybrid:
            query_vector = self._query_vector(query)
            if query_vector is None:
                return []
            params["vector"] = query_vector
            params["hybrid"] = {"embedder": self.embedder_name, "semanticRatio": self.semantic_ratio}
        elif self.search_type != SearchType.keyword:
            log_error(f"Invalid search type '{self.search_type}'.")
            return []

        try:
            results = self.index.search(search_query, params)
        except Exception as e:
            log_error(f"Error searching Meilisearch index '{self.index_name}': {str(e)}")
            return []

        documents = [self._hit_to_document(hit) for hit in results.get("hits", [])]
        if self.reranker and documents:
            documents = self.reranker.rerank(query=query, documents=documents)
        return documents

    async def async_search(
        self, query: str, limit: int = 5, filters: Optional[Union[str, Dict[str, Any], List[FilterExpr]]] = None
    ) -> List[Document]:
        return await asyncio.to_thread(self.search, query, limit, filters)

    def _hit_to_document(self, hit: Dict[str, Any]) -> Document:
        meta_data = hit.get("meta_data") or {}
        return Document(
            id=str(hit.get("id")) if hit.get("id") is not None else None,
            name=hit.get("name"),
            content=hit.get("content") or "",
            meta_data=meta_data,
            usage=hit.get("usage"),
            content_id=hit.get("content_id"),
        )

    def drop(self) -> None:
        if self.exists():
            task = self.index.delete()
            self._wait_for_task(task)

    async def async_drop(self) -> None:
        await asyncio.to_thread(self.drop)

    def delete(self) -> bool:
        try:
            task = self.index.delete_all_documents()
            self._wait_for_task(task)
            return True
        except Exception as e:
            log_error(f"Error deleting all Meilisearch documents: {str(e)}")
            return False

    def delete_by_id(self, id: str) -> bool:
        try:
            task = self.index.delete_document(id)
            self._wait_for_task(task)
            return True
        except Exception as e:
            log_error(f"Error deleting Meilisearch document with id '{id}': {str(e)}")
            return False

    def _delete_by_filter(self, filter_expression: Optional[Any]) -> bool:
        if not filter_expression:
            return False
        try:
            task = self.index.delete_documents(filter=filter_expression)
            self._wait_for_task(task)
            return True
        except Exception as e:
            log_error(f"Error deleting Meilisearch documents with filter '{filter_expression}': {str(e)}")
            return False

    def delete_by_name(self, name: str) -> bool:
        return self._delete_by_filter(self._build_filter({"name": name}))

    def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        return self._delete_by_filter(self._build_filter(metadata))

    def delete_by_content_id(self, content_id: str) -> bool:
        return self._delete_by_filter(self._build_filter({"content_id": content_id}))

    def _delete_by_content_hash(self, content_hash: str) -> bool:
        return self._delete_by_filter(self._build_filter({"content_hash": content_hash}))

    def name_exists(self, name: str) -> bool:
        return self._exists_by_filter({"name": name})

    async def async_name_exists(self, name: str) -> bool:
        return await asyncio.to_thread(self.name_exists, name)

    def id_exists(self, id: str) -> bool:
        try:
            self.index.get_document(id)
            return True
        except MeilisearchApiError as e:
            if e.status_code == 404:
                return False
            raise
        except Exception:
            return False

    def content_hash_exists(self, content_hash: str) -> bool:
        return self._exists_by_filter({"content_hash": content_hash})

    def _exists_by_filter(self, filters: Dict[str, Any]) -> bool:
        filter_expression = self._build_filter(filters)
        if not filter_expression:
            return False
        try:
            docs = self.index.get_documents({"filter": filter_expression, "limit": 1, "fields": ["id"]})
            return len(docs.results) > 0
        except Exception as e:
            log_error(f"Error checking Meilisearch document existence: {str(e)}")
            return False

    def update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> None:
        filter_expression = self._build_filter({"content_id": content_id})
        if not filter_expression:
            return

        docs = self.index.get_documents({"filter": filter_expression, "limit": 10000})
        updates = []
        for hit in docs.results:
            current = dict(hit)
            meta_data = current.get("meta_data") or {}
            meta_data.update(metadata)
            current["meta_data"] = meta_data
            for key, value in metadata.items():
                if key in self.filterable_attributes:
                    current[key] = value
            updates.append(current)

        if updates:
            task = self.index.update_documents(updates, primary_key=self.primary_key)
            self._wait_for_task(task)

    def optimize(self) -> None:
        pass

    def get_supported_search_types(self) -> List[str]:
        return [SearchType.vector, SearchType.keyword, SearchType.hybrid]

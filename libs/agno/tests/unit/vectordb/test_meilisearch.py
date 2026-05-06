import importlib
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("meilisearch")

from agno.filters import EQ
from agno.knowledge.document import Document
from agno.vectordb.meilisearch.meilisearch import Meilisearch
from agno.vectordb.search import SearchType


class FakeDocumentsResponse:
    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results


@pytest.fixture
def mock_meili_index():
    index = MagicMock()
    index.search.return_value = {"hits": []}
    index.get_documents.return_value = FakeDocumentsResponse(results=[])
    return index


@pytest.fixture
def mock_meili_client(mock_meili_index):
    client = MagicMock()
    client.index.return_value = mock_meili_index
    client.get_index.return_value = mock_meili_index
    return client


@pytest.fixture
def meili_db(mock_meili_client, mock_embedder):
    return Meilisearch(
        index_name="test_index",
        client=mock_meili_client,
        embedder=mock_embedder,
        wait_for_tasks=False,
    )


@pytest.fixture
def sample_documents() -> List[Document]:
    return [
        Document(content="Doc A", meta_data={"category": "A"}, name="doc_a"),
        Document(content="Doc B", meta_data={"category": "B"}, name="doc_b"),
    ]


def test_initialization_validates_inputs(mock_meili_client, mock_embedder):
    with pytest.raises(ValueError):
        Meilisearch(index_name="", client=mock_meili_client, embedder=mock_embedder)

    with pytest.raises(ValueError):
        Meilisearch(index_name="test", embedder_name="", client=mock_meili_client, embedder=mock_embedder)

    with pytest.raises(ValueError):
        Meilisearch(index_name="test", semantic_ratio=1.5, client=mock_meili_client, embedder=mock_embedder)

    bad_embedder = MagicMock()
    bad_embedder.dimensions = None
    with pytest.raises(ValueError):
        Meilisearch(index_name="test", client=mock_meili_client, embedder=bad_embedder)


def test_format_filter_value(meili_db):
    assert meili_db._format_filter_value(True) == "true"
    assert meili_db._format_filter_value(False) == "false"
    assert meili_db._format_filter_value(42) == "42"
    assert meili_db._format_filter_value(3.5) == "3.5"
    assert meili_db._format_filter_value(None) == "NULL"
    assert meili_db._format_filter_value("path\\file") == '"path\\\\file"'
    assert meili_db._format_filter_value('name"doc') == '"name\\"doc"'


def test_build_filter_handles_inputs(meili_db):
    assert meili_db._build_filter(None) is None
    assert meili_db._build_filter('name = "doc"') == 'name = "doc"'
    assert meili_db._build_filter([EQ("name", "doc")]) is None

    filters = {"category": ["A", "B"], "score": 10}
    assert meili_db._build_filter(filters) == 'category IN ["A", "B"] AND score = 10'

    assert meili_db._build_filter({"tags": []}) is None
    assert meili_db._build_filter(123) is None


def test_prepare_vector_validation(meili_db):
    with pytest.raises(ValueError):
        meili_db._prepare_vector([], "doc")

    with pytest.raises(ValueError):
        meili_db._prepare_vector([0.1], "doc")

    vector = meili_db._prepare_vector([1] * meili_db.dimensions, "doc")
    assert len(vector) == meili_db.dimensions
    assert all(isinstance(value, float) for value in vector)


def test_document_id_is_deterministic(meili_db):
    doc = Document(content="Hello\x00World")
    content_hash = "hash"
    assert meili_db._document_id(doc, content_hash) == meili_db._document_id(doc, content_hash)


def test_serialize_document_merges_filters_and_flattens(meili_db):
    meili_db.embedder.get_embedding_and_usage.reset_mock()
    doc = Document(content="Hello\x00World", meta_data={"category": "A"}, name="doc_a", content_id="cid_1")

    payload = meili_db._serialize_document(doc, content_hash="hash", filters={"team_id": "team_1", "tags": ["x"]})

    assert payload["content"] == "Hello\ufffdWorld"
    assert payload["meta_data"]["category"] == "A"
    assert payload["meta_data"]["team_id"] == "team_1"
    assert payload["team_id"] == "team_1"
    assert payload["tags"] == ["x"]
    meili_db.embedder.get_embedding_and_usage.assert_called_once_with("Hello\x00World")


def test_insert_creates_index_and_adds_documents(meili_db, sample_documents):
    with patch.object(meili_db, "exists", return_value=False), patch.object(meili_db, "create") as create_mock:
        meili_db.insert(content_hash="hash", documents=sample_documents)
        create_mock.assert_called_once()
        meili_db.index.add_documents.assert_called_once()

    args, kwargs = meili_db.index.add_documents.call_args
    assert len(args[0]) == len(sample_documents)
    assert kwargs["primary_key"] == meili_db.primary_key


def test_insert_no_documents_short_circuits(meili_db):
    meili_db.index.add_documents.reset_mock()
    meili_db.insert(content_hash="hash", documents=[])
    meili_db.index.add_documents.assert_not_called()


def test_search_vector_builds_params_and_returns_documents(meili_db):
    meili_db.search_type = SearchType.vector
    meili_db.index.search.return_value = {
        "hits": [
            {
                "id": "1",
                "name": "doc",
                "content": "Hello",
                "meta_data": {"category": "A"},
                "usage": {"tokens": 1},
                "content_id": "cid_1",
            }
        ]
    }

    with patch.object(meili_db, "exists", return_value=True):
        results = meili_db.search("hello", limit=2, filters={"name": "doc"})

    assert len(results) == 1
    assert results[0].name == "doc"
    assert results[0].content == "Hello"

    search_query, params = meili_db.index.search.call_args.args
    assert search_query == ""
    assert params["limit"] == 2
    assert params["filter"] == 'name = "doc"'
    assert params["hybrid"] == {"embedder": meili_db.embedder_name, "semanticRatio": 1.0}
    assert params["vector"] == meili_db.embedder.get_embedding.return_value


def test_search_hybrid_builds_params(meili_db):
    meili_db.search_type = SearchType.hybrid
    meili_db.semantic_ratio = 0.3
    meili_db.index.search.return_value = {"hits": []}

    with patch.object(meili_db, "exists", return_value=True):
        meili_db.search("hello", limit=2)

    search_query, params = meili_db.index.search.call_args.args
    assert search_query == "hello"
    assert params["hybrid"] == {"embedder": meili_db.embedder_name, "semanticRatio": 0.3}


def test_search_keyword_does_not_request_embedding(meili_db):
    meili_db.search_type = SearchType.keyword
    meili_db.embedder.get_embedding.reset_mock()
    meili_db.index.search.return_value = {"hits": []}

    with patch.object(meili_db, "exists", return_value=True):
        meili_db.search("hello", limit=2)

    meili_db.embedder.get_embedding.assert_not_called()


def test_search_returns_empty_on_embedding_mismatch(mock_meili_client, mock_meili_index):
    bad_embedder = MagicMock()
    bad_embedder.dimensions = 1024
    bad_embedder.get_embedding.return_value = [0.1]

    db = Meilisearch(index_name="test_index", client=mock_meili_client, embedder=bad_embedder, wait_for_tasks=False)
    db.search_type = SearchType.vector

    with patch.object(db, "exists", return_value=True):
        results = db.search("hello", limit=2)

    assert results == []
    mock_meili_index.search.assert_not_called()


def test_search_returns_empty_when_index_missing(meili_db):
    with patch.object(meili_db, "exists", return_value=False):
        assert meili_db.search("hello", limit=2) == []


def test_exists_handles_not_found(monkeypatch, mock_meili_client, mock_embedder):
    meili_module = importlib.import_module("agno.vectordb.meilisearch.meilisearch")

    class DummyError(Exception):
        def __init__(self, status_code: int):
            self.status_code = status_code

    monkeypatch.setattr(meili_module, "MeilisearchApiError", DummyError, raising=False)

    mock_meili_client.get_index.side_effect = DummyError(404)
    db = meili_module.Meilisearch(index_name="test_index", client=mock_meili_client, embedder=mock_embedder)

    assert db.exists() is False


def test_id_exists_handles_not_found(monkeypatch, mock_meili_client, mock_meili_index, mock_embedder):
    meili_module = importlib.import_module("agno.vectordb.meilisearch.meilisearch")

    class DummyError(Exception):
        def __init__(self, status_code: int):
            self.status_code = status_code

    monkeypatch.setattr(meili_module, "MeilisearchApiError", DummyError, raising=False)

    mock_meili_index.get_document.side_effect = DummyError(404)
    db = meili_module.Meilisearch(index_name="test_index", client=mock_meili_client, embedder=mock_embedder)

    assert db.id_exists("missing") is False


def test_exists_by_filter(meili_db, mock_meili_index):
    mock_meili_index.get_documents.return_value = FakeDocumentsResponse(results=[{"id": "1"}])
    assert meili_db._exists_by_filter({"name": "doc"}) is True

    mock_meili_index.get_documents.return_value = FakeDocumentsResponse(results=[])
    assert meili_db._exists_by_filter({"name": "doc"}) is False


def test_update_metadata_updates_filterable_fields(meili_db, mock_meili_index):
    mock_meili_index.get_documents.return_value = FakeDocumentsResponse(
        results=[{"id": "1", "meta_data": {"category": "A"}, "team_id": "old"}]
    )

    meili_db.update_metadata("cid_1", {"team_id": "new", "rating": 5})

    updates = mock_meili_index.update_documents.call_args.args[0]
    assert updates[0]["meta_data"]["team_id"] == "new"
    assert updates[0]["meta_data"]["rating"] == 5
    assert updates[0]["team_id"] == "new"


def test_delete_by_filter_short_circuits(meili_db):
    assert meili_db._delete_by_filter(None) is False


def test_delete_returns_false_on_exception(meili_db):
    meili_db.index.delete_all_documents.side_effect = Exception("boom")
    assert meili_db.delete() is False


def test_delete_by_id_returns_false_on_exception(meili_db):
    meili_db.index.delete_document.side_effect = Exception("boom")
    assert meili_db.delete_by_id("doc_id") is False


def test_upsert_deletes_existing_content_hash(meili_db, sample_documents):
    with (
        patch.object(meili_db, "content_hash_exists", return_value=True),
        patch.object(meili_db, "_delete_by_content_hash") as delete_mock,
        patch.object(meili_db, "insert") as insert_mock,
    ):
        meili_db.upsert(content_hash="hash", documents=sample_documents)

    delete_mock.assert_called_once_with("hash")
    insert_mock.assert_called_once_with(content_hash="hash", documents=sample_documents, filters=None)

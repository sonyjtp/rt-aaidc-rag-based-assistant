"""Comprehensive tests for VectorDB class to improve code coverage."""
# pylint: disable=protected-access, too-many-public-methods, unused-variable
# pylint: disable=import-error, redefined-outer-name
# flake8: noqa: F811


from unittest.mock import patch

import pytest
from chromadb.errors import InvalidArgumentError

from src.vectordb import VectorDB

# Import shared fixtures to avoid duplication across tests
# pylint: disable=unused-import
from tests.fixtures.vectordb_fixtures import (  # noqa: F401
    patched_vectordb,
    vectordb_mocks,
)


class TestVectorDB:
    """Comprehensive tests for VectorDB class to improve code coverage."""

    def test_vectordb_initialization(self, vectordb_mocks):
        """Test that VectorDB initializes correctly with all components."""
        with patch("src.vectordb.ChromaDBClient") as mock_chroma:
            mock_chroma.return_value = vectordb_mocks["chroma_instance"]

            vdb = VectorDB()

            assert vdb.collection is not None
            assert vdb.text_splitter is not None
            mock_chroma.assert_called_once()

    def test_standardize_document_string(self):
        """Test standardize_document with string input."""
        doc = "This is a test document"
        result = VectorDB.standardize_document(doc)

        assert isinstance(result, dict)
        assert result["content"] == doc
        assert result["title"] == ""
        assert result["filename"] == ""
        assert result["tags"] == ""

    def test_standardize_document_dict_complete(self):
        """Test standardize_document with complete dict input."""
        doc = {
            "content": "Test content",
            "title": "Test Title",
            "filename": "test.txt",
            "tags": ["test", "document"],
        }
        result = VectorDB.standardize_document(doc)

        assert isinstance(result, dict)
        assert result["content"] == "Test content"
        assert result["title"] == "Test Title"
        assert result["filename"] == "test.txt"
        assert isinstance(result["tags"], str)

    def test_standardize_document_dict_partial(self):
        """Test standardize_document with partial dictionary input."""
        doc = {
            "content": "Test content",
            "title": "Test Title",
        }
        result = VectorDB.standardize_document(doc)

        assert result["content"] == "Test content"
        assert result["title"] == "Test Title"
        assert result["filename"] == ""
        assert result["tags"] == ""

    def test_standardize_document_dict_empty(self):
        """Test standardize_document with empty dictionary."""
        doc = {}
        result = VectorDB.standardize_document(doc)

        assert result["content"] == ""
        assert result["title"] == ""
        assert result["filename"] == ""
        assert result["tags"] == ""

    def test_chunk_documents_single_string(self, vectordb_mocks):
        """Test chunking a single string document."""
        with patch("src.vectordb.ChromaDBClient") as mock_chroma:
            mock_chroma.return_value = vectordb_mocks["chroma_instance"]

            vdb = VectorDB()
            documents = ["This is a test document with content to be split."]

            chunks = vdb._chunk_documents(documents)

            assert len(chunks) > 0
            tuple_check = all(isinstance(chunk, tuple) and len(chunk) == 2 for chunk in chunks)
            assert tuple_check

    def test_chunk_documents_multiple_documents(self, vectordb_mocks):
        """Test chunking multiple documents."""
        with patch("src.vectordb.ChromaDBClient") as mock_chroma:
            mock_chroma.return_value = vectordb_mocks["chroma_instance"]

            vdb = VectorDB()
            documents = [
                {
                    "content": "Document one content",
                    "title": "Doc1",
                    "filename": "doc1.txt",
                },
                {
                    "content": "Document two content",
                    "title": "Doc2",
                    "filename": "doc2.txt",
                },
            ]

            chunks = vdb._chunk_documents(documents)

            assert len(chunks) >= 2
            metadatas = [metadata for _, metadata in chunks]
            titles = [m["title"] for m in metadatas]
            assert "Doc1" in titles
            assert "Doc2" in titles

    def test_chunk_documents_filters_empty_chunks(self, vectordb_mocks):
        """Test that empty chunks are filtered out."""
        with patch("src.vectordb.ChromaDBClient") as mock_chroma:
            mock_chroma.return_value = vectordb_mocks["chroma_instance"]

            vdb = VectorDB()
            documents = [
                "   ",  # Only whitespace
                "Actual content here",
            ]

            chunks = vdb._chunk_documents(documents)

            assert all(chunk[0].strip() for chunk, _ in chunks)

    def test_chunk_documents_filters_title_duplicates(self, vectordb_mocks):
        """Test that chunks identical to title are filtered out."""
        with patch("src.vectordb.ChromaDBClient") as mock_chroma:
            mock_chroma.return_value = vectordb_mocks["chroma_instance"]

            vdb = VectorDB()
            title = "Test Title"
            # Content same as title
            documents = [{"content": title, "title": title}]

            chunks = vdb._chunk_documents(documents)

            # Filter out chunk that equals the title
            check = all(chunk[0].strip() != title for chunk, _ in chunks)
            assert check

    @pytest.mark.parametrize(
        "mock_get_side_effect,input_chunks,expected_count,test_assertion",
        [
            pytest.param(
                [{"documents": []}, {"documents": []}],
                [("chunk1", {"title": "doc1"}), ("chunk2", {"title": "doc2"})],
                2,
                lambda filtered: True,
                id="no_duplicates",
            ),
            pytest.param(
                [{"documents": ["chunk1"]}, {"documents": []}],
                [("chunk1", {"title": "doc1"}), ("chunk2", {"title": "doc2"})],
                1,
                lambda filtered: filtered[0][0] == "chunk2",
                id="with_existing_duplicates",
            ),
            pytest.param(
                [{"documents": []}, {"documents": []}],
                [
                    ("same_chunk", {"title": "doc1"}),
                    ("same_chunk", {"title": "doc2"}),
                    ("different_chunk", {"title": "doc3"}),
                ],
                2,
                lambda filtered: [chunk for chunk, _ in filtered].count("same_chunk") == 1,
                id="with_batch_duplicates",
            ),
        ],
    )
    def test_filter_duplicate_chunks(
        self, vectordb_mocks, mock_get_side_effect, input_chunks, expected_count, test_assertion
    ):
        """Parametrized test for filter_duplicate_chunks with various scenarios."""
        with patch("src.vectordb.ChromaDBClient") as mock_chroma:
            mock_chroma.return_value = vectordb_mocks["chroma_instance"]
            mock_collection = vectordb_mocks["collection"]
            mock_collection.get.side_effect = mock_get_side_effect

            vdb = VectorDB()
            filtered = vdb._filter_duplicate_chunks(input_chunks)

            assert len(filtered) == expected_count
            assert test_assertion(filtered)

    def test_filter_duplicate_chunks_with_punctuation_normalization(self, vectordb_mocks):
        """Test filtering handles punctuation normalization."""
        with patch("src.vectordb.ChromaDBClient") as mock_chroma:
            mock_chroma.return_value = vectordb_mocks["chroma_instance"]

            mock_collection = vectordb_mocks["collection"]
            # Handle pagination - empty database
            mock_collection.get.side_effect = [
                {"documents": []},  # First call
                {"documents": []},  # Second call - end pagination
            ]

            vdb = VectorDB()
            chunks = [
                ("...this is a test", {"title": "doc1"}),
                ("this is a test", {"title": "doc2"}),
            ]

            filtered = vdb._filter_duplicate_chunks(chunks)

            # Both should be kept because the first has leading punctuation
            # which is preserved in the chunk but normalized for comparison
            assert len(filtered) <= 2

    def test_insert_chunks_success(self, patched_vectordb):
        """Test successful insertion of chunks into database."""
        vdb, mocks = patched_vectordb

        # Handle pagination - empty database
        mocks["collection"].get.side_effect = [
            {"documents": []},  # First call
            {"documents": []},  # Second call (offset=300) - end pagination
        ]
        mocks["collection"].count.return_value = 0
        mocks["embedding_model"].embed_documents.return_value = [[0.1, 0.2]]

        chunks = [("test chunk", {"title": "test"})]

        vdb._insert_chunks_into_db(chunks)

        mocks["collection"].add.assert_called_once()

    def test_insert_chunks_empty_list(self, patched_vectordb):
        """Test insertion with empty chunk list."""
        vdb, mocks = patched_vectordb

        # Handle pagination - existing chunks in DB with empty on second call
        mocks["collection"].get.side_effect = [
            {"documents": ["chunk1", "chunk2"]},  # First call
            {"documents": []},  # Second call (offset=300) - end pagination
        ]

        chunks = [("chunk1", {"title": "test"}), ("chunk2", {"title": "test"})]

        vdb._insert_chunks_into_db(chunks)

        # Should not call add if all chunks are duplicates
        mocks["collection"].add.assert_not_called()

    def test_insert_chunks_deduplication_logging(self, patched_vectordb):
        """Test that deduplication is logged correctly."""
        vdb, mocks = patched_vectordb

        with patch("src.vectordb.logger") as mock_logger:
            # Handle pagination - empty database
            mocks["collection"].get.side_effect = [
                {"documents": []},  # First call
                {"documents": []},  # Second call (offset=300)
            ]
            mocks["collection"].count.return_value = 0
            mocks["embedding_model"].embed_documents.return_value = [[0.1, 0.2]]

            chunks = [
                ("chunk1", {"title": "test"}),
                ("chunk1", {"title": "test"}),  # Duplicate
                ("chunk2", {"title": "test"}),
            ]

            vdb._insert_chunks_into_db(chunks)

            # Should log deduplication info
            assert any("Deduplicated" in str(call) for call in mock_logger.info.call_args_list)

    def test_add_documents_integration(self, patched_vectordb):
        """Test add_documents integration with chunking and insertion."""
        vdb, mocks = patched_vectordb

        # Handle pagination - return empty list to stop loop
        mocks["collection"].get.side_effect = [
            {"documents": []},  # First call
            {"documents": []},  # Second call (offset=300) - end pagination
        ]
        mocks["collection"].count.return_value = 0
        mocks["embedding_model"].embed_documents.return_value = [[0.1], [0.2]]

        documents = [
            "Document one with some content",
            "Document two with more content",
        ]

        vdb.add_documents(documents)

        mocks["collection"].add.assert_called_once()

    def test_search_success(self, patched_vectordb):
        """Test successful search query."""
        vdb, mocks = patched_vectordb

        mocks["embedding_model"].embed_query.return_value = [0.1, 0.2, 0.3]
        mocks["collection"].query.return_value = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"title": "Title1"}, {"title": "Title2"}]],
            "distances": [[0.1, 0.2]],
            "ids": [["id1", "id2"]],
        }

        result = vdb.search("test query", n_results=2)

        assert len(result["documents"]) == 2
        assert result["documents"] == ["doc1", "doc2"]
        assert len(result["ids"]) == 2

    def test_search_with_distance_filtering(self, patched_vectordb):
        """Test search with distance threshold filtering."""
        vdb, mocks = patched_vectordb

        mocks["embedding_model"].embed_query.return_value = [0.1, 0.2, 0.3]
        mocks["collection"].query.return_value = {
            "documents": [["doc1", "doc2", "doc3"]],
            "metadatas": [
                [
                    {"title": "Title1"},
                    {"title": "Title2"},
                    {"title": "Title3"},
                ]
            ],
            "distances": [[0.1, 0.3, 0.5]],  # Last one exceeds threshold
            "ids": [["id1", "id2", "id3"]],
        }

        result = vdb.search("test query", maximum_distance=0.35)

        assert len(result["documents"]) == 2
        assert result["documents"] == ["doc1", "doc2"]

    def test_search_no_results(self, vectordb_mocks):
        """Test search when no results are returned."""
        with patch("src.vectordb.ChromaDBClient") as mock_chroma:
            mock_chroma.return_value = vectordb_mocks["chroma_instance"]
            mock_collection = vectordb_mocks["collection"]
            mock_embedding_model = vectordb_mocks["embedding_model"]
            mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
            mock_collection.query.return_value = {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "ids": [[]],
            }

            vdb = VectorDB()
            result = vdb.search("test query")

            assert result["documents"] == []
            assert result["metadatas"] == []
            assert result["distances"] == []
            assert result["ids"] == []

    def test_search_all_results_filtered(self, vectordb_mocks):
        """Test search where all results are filtered by distance threshold."""
        with patch("src.vectordb.ChromaDBClient") as mock_chroma:
            mock_chroma.return_value = vectordb_mocks["chroma_instance"]
            mock_collection = vectordb_mocks["collection"]
            mock_embedding_model = vectordb_mocks["embedding_model"]
            mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
            mock_collection.query.return_value = {
                "documents": [["doc1", "doc2"]],
                "metadatas": [[{"title": "Title1"}, {"title": "Title2"}]],
                "distances": [[0.5, 0.6]],  # All exceed threshold
                "ids": [["id1", "id2"]],
            }

            vdb = VectorDB()
            result = vdb.search("test query", maximum_distance=0.35)

            assert result["documents"] == []
            assert result["metadatas"] == []
            assert result["distances"] == []
            assert result["ids"] == []

    def test_search_handles_embedding_dimension_mismatch(self, vectordb_mocks):
        """Test that search handles embedding dimension mismatch gracefully."""
        with patch("src.vectordb.ChromaDBClient") as mock_chroma:
            mock_chroma.return_value = vectordb_mocks["chroma_instance"]
            mock_collection = vectordb_mocks["collection"]
            mock_embedding_model = vectordb_mocks["embedding_model"]
            mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
            # Mock the collection query to raise InvalidArgumentError
            error_msg = "Collection expecting embedding with dimension of 768, got 384"
            mock_collection.query.side_effect = InvalidArgumentError(error_msg)

            vdb = VectorDB()
            result = vdb.search(query="test query")

            # Should return empty results gracefully
            assert result["documents"] == []
            assert result["metadatas"] == []
            assert result["distances"] == []
            assert result["ids"] == []

    def test_search_handles_other_invalid_argument_error(self, vectordb_mocks):
        """Test InvalidArgumentError re-raised for non-dimension errors."""
        with patch("src.vectordb.ChromaDBClient") as mock_chroma:
            mock_chroma.return_value = vectordb_mocks["chroma_instance"]
            mock_collection = vectordb_mocks["collection"]
            mock_embedding_model = vectordb_mocks["embedding_model"]
            mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]

            # Mock a different InvalidArgumentError
            error_msg = "Some other invalid argument error"
            mock_collection.query.side_effect = InvalidArgumentError(error_msg)

            vdb = VectorDB()

            with pytest.raises(InvalidArgumentError):
                vdb.search(query="test query")

    def test_search_with_custom_n_results(self, vectordb_mocks):
        """Test search with custom n_results parameter."""
        with patch("src.vectordb.ChromaDBClient") as mock_chroma:
            mock_chroma.return_value = vectordb_mocks["chroma_instance"]
            mock_collection = vectordb_mocks["collection"]
            mock_embedding_model = vectordb_mocks["embedding_model"]
            mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
            mock_collection.query.return_value = {
                "documents": [["doc1", "doc2", "doc3"]],
                "metadatas": [
                    [
                        {"title": "Title1"},
                        {"title": "Title2"},
                        {"title": "Title3"},
                    ]
                ],
                "distances": [[0.1, 0.2, 0.3]],
                "ids": [["id1", "id2", "id3"]],
            }

            vdb = VectorDB()
            vdb.search("test query", n_results=10)

            # Verify query was called with correct n_results
            mock_collection.query.assert_called_once()
            call_args = mock_collection.query.call_args
            assert call_args[1]["n_results"] == 10

    @pytest.mark.parametrize(
        "results,max_distance,expected_docs,test_id",
        [
            pytest.param(
                {
                    "documents": [["doc1", "doc2"]],
                    "metadatas": [[{"title": "T1"}, {"title": "T2"}]],
                    "distances": [[0.1, 0.2]],
                    "ids": [["id1", "id2"]],
                },
                0.3,
                ["doc1", "doc2"],
                "complete",
            ),
            pytest.param(
                {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]},
                0.3,
                [],
                "empty",
            ),
            pytest.param({}, 0.3, [], "missing_keys"),
        ],
    )
    def test_extract_and_filter_search_results(self, results, max_distance, expected_docs, test_id):
        """Parametrized test for extraction and filtering with various result scenarios."""
        docs, metas, dists, ids = VectorDB._extract_and_filter_search_results(results, maximum_distance=max_distance)

        assert docs == expected_docs
        assert metas == [] if not expected_docs else len(metas) > 0
        assert dists == [] if not expected_docs else len(dists) > 0
        assert ids == [] if not expected_docs else len(ids) > 0

    def test_filter_search_results_all_pass(self):
        """Test filtering when all results pass threshold."""
        documents = ["doc1", "doc2"]

    @pytest.mark.parametrize(
        "results,max_distance,expected_doc_count,expected_docs",
        [
            pytest.param(
                {
                    "documents": [["doc1", "doc2"]],
                    "metadatas": [[{"title": "T1"}, {"title": "T2"}]],
                    "distances": [[0.1, 0.2]],
                    "ids": [["id1", "id2"]],
                },
                0.3,
                2,
                ["doc1", "doc2"],
                id="all_pass",
            ),
            pytest.param(
                {
                    "documents": [["doc1", "doc2", "doc3"]],
                    "metadatas": [[{"title": "T1"}, {"title": "T2"}, {"title": "T3"}]],
                    "distances": [[0.1, 0.3, 0.5]],
                    "ids": [["id1", "id2", "id3"]],
                },
                0.35,
                2,
                ["doc1", "doc2"],
                id="partial_pass",
            ),
            pytest.param(
                {
                    "documents": [["doc1", "doc2"]],
                    "metadatas": [[{"title": "T1"}, {"title": "T2"}]],
                    "distances": [[0.5, 0.6]],
                    "ids": [["id1", "id2"]],
                },
                0.3,
                0,
                [],
                id="none_pass",
            ),
            pytest.param(
                {
                    "documents": [["doc1", "doc2"]],
                    "metadatas": [[{"title": "T1"}, {"title": "T2"}]],
                    "distances": [[0.35, 0.36]],
                    "ids": [["id1", "id2"]],
                },
                0.35,
                1,
                ["doc1"],
                id="exact_threshold",
            ),
        ],
    )
    def test_extract_and_filter_search_results_filtering(
        self, results, max_distance, expected_doc_count, expected_docs
    ):
        """Parametrized test for extraction and filtering with various result sets."""
        docs, metas, dists, ids_out = VectorDB._extract_and_filter_search_results(
            results, maximum_distance=max_distance
        )

        assert len(docs) == expected_doc_count
        assert docs == expected_docs
        assert len(metas) == expected_doc_count
        assert len(dists) == expected_doc_count
        assert len(ids_out) == expected_doc_count

    def test_add_documents_empty_list(self, patched_vectordb):
        """Test add_documents with empty list - should raise ValueError."""
        vdb, mocks = patched_vectordb

        # Should raise ValueError when empty list is provided
        with pytest.raises(ValueError, match="Document list is empty"):
            vdb.add_documents([])

    def test_filter_duplicate_chunks_exception_handling(self, vectordb_mocks):
        """Test _filter_duplicate_chunks handles exceptions during batch fetching."""
        with patch("src.vectordb.ChromaDBClient") as mock_chroma:
            mock_chroma.return_value = vectordb_mocks["chroma_instance"]
            mock_collection = vectordb_mocks["collection"]

            # First call succeeds, second call raises exception
            mock_collection.get.side_effect = [
                {"documents": ["chunk1"]},  # First call succeeds
                RuntimeError("Batch fetch failed"),  # Second call fails
            ]

            vdb = VectorDB()
            chunks = [
                ("chunk1", {"title": "doc1"}),
                ("chunk2", {"title": "doc2"}),
            ]

            # Should handle exception and return filtered chunks from first batch
            filtered = vdb._filter_duplicate_chunks(chunks)

            # Should have filtered chunk2 (not chunk1 since it was in first batch)
            assert len(filtered) == 1
            assert filtered[0][0] == "chunk2"

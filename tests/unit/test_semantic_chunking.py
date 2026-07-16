from counterpoint.config import Settings
from counterpoint.ingest import split_sentences, semantic_chunks, chunk_document
from counterpoint.embeddings import get_backend, HashingBackend, TfidfBackend
from counterpoint.models import Document


def test_split_sentences_basic():
    s = split_sentences("The oven failed. A new supplier was used! Was it the flour?")
    assert len(s) == 3
    assert s[0].startswith("The oven")


def test_semantic_chunks_groups_related_and_splits_topics():
    settings = Settings(embedding_backend="tfidf", chunk_max_sentences=8,
                        chunk_similarity_percentile=25)
    backend = TfidfBackend()
    sentences = [
        "The oven temperature sensors reported stable readings all night.",
        "The oven controller firmware is current and raised no fault codes.",
        "The bakery switched flour suppliers two weeks ago.",
        "The new flour supplier delivered late on three occasions.",
    ]
    chunks = semantic_chunks(sentences, backend, settings)
    # should produce >1 chunk (oven topic vs flour topic) and cover all sentences
    assert len(chunks) >= 2
    joined = " ".join(chunks)
    for s in sentences:
        assert s in joined


def test_semantic_chunks_respects_max_sentences():
    settings = Settings(embedding_backend="hashing", chunk_max_sentences=2)
    backend = HashingBackend()
    sentences = [f"Sentence number {i} about widgets." for i in range(6)]
    chunks = semantic_chunks(sentences, backend, settings)
    for c in chunks:
        assert c.count(".") <= 2


def test_chunk_document_end_to_end():
    settings = Settings(embedding_backend="hashing")
    doc = Document(doc_id="d1", name="t", text=(
        "The thermostat was replaced on Monday. Calibration was not signed off. "
        "Separately, the flour supplier changed in November. Deliveries were late."))
    chunks = chunk_document(doc, settings, HashingBackend())
    assert len(chunks) >= 1
    assert all(c.doc_id == "d1" for c in chunks)
    assert all(c.chunk_id.startswith("d1::c") for c in chunks)


def test_empty_document():
    settings = Settings(embedding_backend="hashing")
    doc = Document(doc_id="d0", name="empty", text="   ")
    assert chunk_document(doc, settings, HashingBackend()) == []

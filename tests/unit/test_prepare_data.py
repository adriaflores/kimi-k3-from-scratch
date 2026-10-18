"""Small behavior checks for data-preparation helpers (no dataset downloads)."""

import hashlib

import pytest

from scripts import prepare_data as data


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, texts, **kwargs):
        assert kwargs == {
            "add_special_tokens": False,
            "padding": False,
            "truncation": False,
            "return_attention_mask": False,
            "return_token_type_ids": False,
            "verbose": False,
        }
        return {"input_ids": [[len(text)] for text in texts]}


class TokenSink:
    def __init__(self):
        self.tokens = 0
        self.documents = []

    def append(self, token_ids):
        self.documents.append(tuple(token_ids))
        self.tokens += len(token_ids)


def test_token_batch_preserves_order_and_appends_eos(monkeypatch):
    monkeypatch.setattr(data, "split_for", lambda text, seed, fraction: "val" if text == "beta" else "train")
    writers = {"train": TokenSink(), "val": TokenSink()}
    data._write_token_batch(["alpha", "beta", "gamma"], FakeTokenizer(), writers, 1337, 0.05, None)
    assert writers["train"].documents == [(5, 0), (5, 0)]
    assert writers["val"].documents == [(4, 0)]


def test_token_batch_stops_after_target_document(monkeypatch):
    monkeypatch.setattr(data, "split_for", lambda *args: "train")
    writers = {"train": TokenSink(), "val": TokenSink()}
    data._write_token_batch(["a", "bb", "ccc"], FakeTokenizer(), writers, 1337, 0.05, 4)
    assert writers["train"].documents == [(1, 0), (2, 0)]
    assert writers["train"].tokens == 4


def test_replay_checks_hash_and_preserves_unread_tail():
    documents = iter(["alpha", "beta", "gamma"])
    digest = hashlib.sha256()

    def consume(text):
        raw = text.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)

    expected = hashlib.sha256()
    for text in ("alpha", "beta"):
        raw = text.encode("utf-8")
        expected.update(len(raw).to_bytes(8, "big"))
        expected.update(raw)

    data._replay_source_prefix(documents, 2, consume, digest, expected.hexdigest(), 100000)
    assert next(documents) == "gamma"


def test_replay_rejects_changed_or_shortened_source():
    digest = hashlib.sha256()
    with pytest.raises(ValueError, match="Source order/content differs"):
        data._replay_source_prefix(iter(["changed"]), 1, lambda _: None, digest, "bad-hash", 100000)
    with pytest.raises(ValueError, match="Source ended before"):
        data._replay_source_prefix(iter([]), 1, lambda _: None, digest, digest.hexdigest(), 100000)

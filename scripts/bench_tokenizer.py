import argparse
import json
import platform
from importlib.metadata import version

from datasets import load_dataset
from huggingface_hub import HfApi
from pyarrow.dataset import ParquetFragmentScanOptions

from prepare_data import DATASET, TOKENIZER, load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--revision", default="main", help="Dataset commit or ref.")
    parser.add_argument("--tokenizer", default=TOKENIZER)
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--max-documents", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle-buffer", type=int, default=1_000)
    args = parser.parse_args()
    if args.max_documents <= 0 or args.shuffle_buffer <= 0:
        parser.error("--max-documents and --shuffle-buffer must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    return args


def benchmark(args: argparse.Namespace) -> dict:
    revision = HfApi().dataset_info(DATASET, revision=args.revision).sha
    if not revision:
        raise RuntimeError("Could not resolve the dataset revision to a commit SHA.")
    tokenizer, tokenizer_metadata = load_tokenizer(args.tokenizer, args.tokenizer_revision)
    documents = load_dataset(
        DATASET, name=args.config, split="train", revision=revision,
        streaming=True, columns=["text"],
        fragment_scan_options=ParquetFragmentScanOptions(pre_buffer=False),
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    visited = nonempty = byte_count = char_count = token_count = 0
    for document in documents.take(args.max_documents):
        visited += 1
        text = document["text"]
        if not isinstance(text, str):
            raise TypeError(f"Expected string text at source row {visited}.")
        if not text.strip():
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        nonempty += 1
        byte_count += len(text.encode("utf-8"))
        char_count += len(text)
        token_count += len(tokens)
    if token_count == 0:
        raise ValueError("The sample contains no text tokens. Increase --max-documents.")

    with_eos = token_count + nonempty
    return {
        "source": {"dataset": DATASET, "config": args.config, "split": "train", "revision": revision},
        "tokenizer": tokenizer_metadata,
        "sampling": {"max_documents": args.max_documents, "seed": args.seed,
                     "shuffle_buffer": args.shuffle_buffer, "method": "buffer shuffle then take"},
        "source_rows_visited": visited,
        "documents": nonempty,
        "empty_documents_skipped": visited - nonempty,
        "utf8_bytes": byte_count,
        "unicode_code_points": char_count,
        "text_tokens": token_count,
        "tokens_with_eos": with_eos,
        "bytes_per_token": byte_count / token_count,
        "chars_per_token": char_count / token_count,
        "bytes_per_token_with_eos": byte_count / with_eos,
        "chars_per_token_with_eos": char_count / with_eos,
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("datasets", "huggingface-hub", "numpy", "pyarrow", "transformers", "tokenizers")},
        },
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(parse_args()), indent=2))

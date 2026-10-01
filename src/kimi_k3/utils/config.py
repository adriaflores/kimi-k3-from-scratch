from dataclasses import fields
from pathlib import Path
from typing import TypeVar

import yaml


T = TypeVar("T")


def load_config(path: str | Path, config_cls: type[T]) -> T:
	path = Path(path)

	with path.open("r", encoding="utf-8") as f:
		data = yaml.safe_load(f)

	if not isinstance(data, dict):
		raise ValueError(f"Expected a YAML mapping in {path}")

	expected = {field.name for field in fields(config_cls)}
	provided = set(data)

	missing = expected - provided
	unknown = provided - expected

	if missing:
		raise ValueError(
			f"Missing config fields in {path}: {sorted(missing)}"
		)

	if unknown:
		raise ValueError(
			f"Unknown config fields in {path}: {sorted(unknown)}"
		)

	return config_cls(**data)
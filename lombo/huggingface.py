"""Read-only integration with the official Hugging Face Hub APIs.

This module is the only layer that knows the external endpoints. This separation
keeps the UI independent of the Hub and provides a single place to add tokens,
caching, downloads and execution in future stages.
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

import httpx
from fastapi import APIRouter, HTTPException, Query, Request

HUB_URL = "https://huggingface.co"
DATASETS_SERVER_URL = "https://datasets-server.huggingface.co"
MAX_CARD_BYTES = 2_000_000

router = APIRouter()


class HuggingFaceHubService:
    """Small asynchronous client for the official public Hub API."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def list_models(self, filters: dict[str, Any]) -> dict:
        params = _clean_params({
            "search": filters.get("search"),
            "author": filters.get("author"),
            "pipeline_tag": filters.get("pipeline"),
            "gated": filters.get("gated"),
            "filter": f"license:{filters['license']}" if filters.get("license") else None,
            "sort": "downloads",
            "direction": -1,
            "limit": filters.get("limit", 30),
            "cursor": filters.get("cursor"),
            "full": "true",
            "cardData": "true",
            "config": "true",
        })
        data, next_cursor = await self._json_page(f"{HUB_URL}/api/models", params=params)
        return {"items": [_model_summary(item) for item in data],
                "next_cursor": next_cursor}

    async def get_model(self, repo_id: str) -> dict:
        safe_id = _repo_path(repo_id)
        metadata, card, config = await asyncio.gather(
            self._json(f"{HUB_URL}/api/models/{safe_id}", params={"blobs": "true"}),
            self._text_optional(f"{HUB_URL}/{safe_id}/resolve/main/README.md"),
            self._json_optional(f"{HUB_URL}/{safe_id}/resolve/main/config.json"),
        )
        result = _model_summary(metadata)
        result.update({
            "card": card,
            "files": [_file_summary(item) for item in (metadata.get("siblings") or [])],
            "config": config or metadata.get("config") or {},
            "context_length": _context_length(config or metadata.get("config") or {}),
            "official_url": f"{HUB_URL}/{safe_id}",
        })
        if not result.get("architecture"):
            result["architecture"] = _architecture(result["config"])
        return result

    async def list_datasets(self, filters: dict[str, Any]) -> dict:
        hub_filters = []
        if filters.get("license"):
            hub_filters.append(f"license:{filters['license']}")
        if filters.get("modality"):
            hub_filters.append(f"modality:{filters['modality']}")
        if filters.get("task"):
            hub_filters.append(f"task_categories:{filters['task']}")
        params = _clean_params({
            "search": filters.get("search"),
            "author": filters.get("author"),
            "gated": filters.get("gated"),
            "filter": hub_filters or None,
            "sort": "downloads",
            "direction": -1,
            "limit": filters.get("limit", 30),
            "cursor": filters.get("cursor"),
            "full": "true",
        })
        data, next_cursor = await self._json_page(f"{HUB_URL}/api/datasets", params=params)
        return {"items": [_dataset_summary(item) for item in data],
                "next_cursor": next_cursor}

    async def get_dataset(self, repo_id: str) -> dict:
        safe_id = _repo_path(repo_id)
        metadata, card, viewer = await asyncio.gather(
            self._json(f"{HUB_URL}/api/datasets/{safe_id}", params={"blobs": "true"}),
            self._text_optional(f"{HUB_URL}/datasets/{safe_id}/resolve/main/README.md"),
            self._json_optional(f"{DATASETS_SERVER_URL}/info", params={"dataset": repo_id}),
        )
        card_data = metadata.get("cardData") or metadata.get("card_data") or {}
        dataset_info = ((viewer or {}).get("dataset_info") or
                        _card_dataset_info(card_data.get("dataset_info")))
        result = _dataset_summary(metadata)
        result.update({
            "card": card,
            "description": _dataset_description(metadata, card),
            "splits": _dataset_splits(dataset_info),
            "features": _dataset_features(dataset_info),
            "size": _dataset_size(dataset_info) or result.get("size"),
            "files": [_file_summary(item) for item in (metadata.get("siblings") or [])],
            "official_url": f"{HUB_URL}/datasets/{safe_id}",
        })
        return result

    async def _json(self, url: str, params: dict | None = None) -> Any:
        data, _ = await self._json_page(url, params=params)
        return data

    async def _json_page(self, url: str, params: dict | None = None) -> tuple[Any, str | None]:
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            return response.json(), _next_cursor(response.headers.get("link"))
        except httpx.HTTPStatusError as exc:
            status = 404 if exc.response.status_code == 404 else 502
            raise HTTPException(status, _hub_error(exc.response)) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(502, f"Hugging Face Hub unavailable: {exc}") from exc

    async def _json_optional(self, url: str, params: dict | None = None) -> Any | None:
        try:
            response = await self.client.get(url, params=params)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return None

    async def _text_optional(self, url: str) -> str | None:
        try:
            response = await self.client.get(url)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            if len(response.content) > MAX_CARD_BYTES:
                return response.content[:MAX_CARD_BYTES].decode("utf-8", "replace")
            return response.text
        except httpx.HTTPError:
            return None


@router.get("/huggingface/models")
async def list_models(
    request: Request,
    search: str | None = None,
    author: str | None = None,
    pipeline: str | None = None,
    license: str | None = None,
    gated: bool | None = None,
    cursor: str | None = None,
    limit: int = Query(30, ge=1, le=100),
):
    service: HuggingFaceHubService = request.app.state.huggingface
    return await service.list_models(locals())


@router.get("/huggingface/models/{repo_id:path}")
async def get_model(request: Request, repo_id: str):
    return await request.app.state.huggingface.get_model(repo_id)


@router.get("/huggingface/datasets")
async def list_datasets(
    request: Request,
    search: str | None = None,
    author: str | None = None,
    modality: str | None = None,
    task: str | None = None,
    license: str | None = None,
    gated: bool | None = None,
    cursor: str | None = None,
    limit: int = Query(30, ge=1, le=100),
):
    service: HuggingFaceHubService = request.app.state.huggingface
    return await service.list_datasets(locals())


@router.get("/huggingface/datasets/{repo_id:path}")
async def get_dataset(request: Request, repo_id: str):
    return await request.app.state.huggingface.get_dataset(repo_id)


def _clean_params(params: dict) -> dict:
    return {key: value for key, value in params.items() if value not in (None, "", [])}


def _next_cursor(link_header: str | None) -> str | None:
    """Extract only the cursor from the rel=next link provided by the Hub."""
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' not in part:
            continue
        target = part.partition("<")[2].partition(">")[0]
        values = parse_qs(urlparse(target).query).get("cursor")
        return values[0] if values else None
    return None


def _repo_path(repo_id: str) -> str:
    parts = repo_id.strip("/").split("/")
    if not repo_id or len(parts) > 2 or any(not part or part in {".", ".."} for part in parts):
        raise HTTPException(400, "invalid repository identifier")
    return "/".join(quote(part, safe="@._-") for part in parts)


def _author(item: dict) -> str:
    return item.get("author") or (item.get("id") or "").partition("/")[0]


def _license(item: dict) -> str | None:
    card = item.get("cardData") or item.get("card_data") or {}
    value = card.get("license")
    if isinstance(value, list):
        return ", ".join(str(part) for part in value)
    if value:
        return str(value)
    tag = next((tag for tag in item.get("tags", []) if tag.startswith("license:")), None)
    return tag.split(":", 1)[1] if tag else None


def _architecture(config: dict) -> str | None:
    architectures = config.get("architectures") or []
    if architectures:
        return ", ".join(str(value) for value in architectures)
    return config.get("model_type")


def _model_summary(item: dict) -> dict:
    config = item.get("config") or {}
    return {
        "id": item.get("id") or item.get("modelId"),
        "author": _author(item),
        "downloads": item.get("downloads"),
        "likes": item.get("likes"),
        "pipeline": item.get("pipeline_tag"),
        "license": _license(item),
        "last_modified": item.get("lastModified") or item.get("last_modified"),
        "tags": item.get("tags") or [],
        "gated": item.get("gated", False),
        "private": item.get("private", False),
        "architecture": _architecture(config),
    }


def _dataset_summary(item: dict) -> dict:
    tags = item.get("tags") or []
    card = item.get("cardData") or item.get("card_data") or {}
    modalities = _tag_values(tags, "modality:") or _as_list(card.get("modality"))
    tasks = (_tag_values(tags, "task_categories:") or
             _as_list(card.get("task_categories")) or _as_list(card.get("task_ids")))
    return {
        "id": item.get("id"),
        "author": _author(item),
        "downloads": item.get("downloads"),
        "likes": item.get("likes"),
        "size": (item.get("usedStorage") or item.get("used_storage") or
                 _card_dataset_size(card.get("dataset_info"))),
        "modalities": modalities,
        "tasks": tasks,
        "license": _license(item),
        "last_modified": item.get("lastModified") or item.get("last_modified"),
        "tags": tags,
        "gated": item.get("gated", False),
        "private": item.get("private", False),
    }


def _tag_values(tags: list[str], prefix: str) -> list[str]:
    return [tag[len(prefix):] for tag in tags if tag.startswith(prefix)]


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _file_summary(item: dict) -> dict:
    return {
        "name": item.get("rfilename") or item.get("path"),
        "size": item.get("size") or (item.get("lfs") or {}).get("size"),
    }


def _context_length(config: dict) -> int | None:
    for key in ("max_position_embeddings", "n_positions", "max_sequence_length",
                "seq_length", "model_max_length"):
        value = config.get(key)
        if isinstance(value, int) and value < 10**9:
            return value
    text_config = config.get("text_config")
    return _context_length(text_config) if isinstance(text_config, dict) else None


def _dataset_splits(dataset_info: dict) -> list[dict]:
    result = []
    for config, info in dataset_info.items():
        for split in (info or {}).get("splits", []):
            result.append({"config": config, **split})
    return result


def _dataset_features(dataset_info: dict) -> dict:
    return {config: (info or {}).get("features") or {}
            for config, info in dataset_info.items()}


def _dataset_size(dataset_info: dict) -> int | None:
    sizes = [(info or {}).get("dataset_size") for info in dataset_info.values()]
    valid = [value for value in sizes if isinstance(value, int)]
    return sum(valid) if valid else None


def _card_dataset_info(value: Any) -> dict:
    """Convert dataset_info from the card YAML to the Viewer format."""
    entries = value if isinstance(value, list) else [value] if isinstance(value, dict) else []
    return {str(entry.get("config_name") or "default"): entry for entry in entries}


def _card_dataset_size(value: Any) -> int | None:
    return _dataset_size(_card_dataset_info(value))


def _dataset_description(metadata: dict, card: str | None) -> str | None:
    card_data = metadata.get("cardData") or metadata.get("card_data") or {}
    return metadata.get("description") or card_data.get("description") or (card[:500] if card else None)


def _hub_error(response: httpx.Response) -> str:
    try:
        data = response.json()
        return data.get("error") or data.get("message") or f"Hub returned HTTP {response.status_code}"
    except ValueError:
        return f"Hub returned HTTP {response.status_code}"

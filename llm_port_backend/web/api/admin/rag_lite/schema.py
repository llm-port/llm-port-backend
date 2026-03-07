"""Pydantic DTOs for RAG Lite admin endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------
# Upload
# -----------------------------------------------------------------------


class RagLiteUploadResponse(BaseModel):
    document_id: uuid.UUID
    job_id: uuid.UUID
    filename: str
    doc_type: str
    status: str = "pending"
    message: str = "File accepted — processing in background."


# -----------------------------------------------------------------------
# Documents
# -----------------------------------------------------------------------


class RagLiteDocumentDTO(BaseModel):
    id: uuid.UUID
    filename: str
    doc_type: str
    collection_id: uuid.UUID | None
    size_bytes: int
    chunk_count: int
    status: str
    created_at: datetime


class RagLiteDocumentDetailDTO(RagLiteDocumentDTO):
    file_store_key: str | None = None
    sha256: str
    metadata_json: dict | None = None


# -----------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------


class RagLiteSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    collection_ids: list[uuid.UUID] | None = None


class RagLiteSearchResult(BaseModel):
    chunk_text: str
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    score: float


class RagLiteSearchResponse(BaseModel):
    results: list[RagLiteSearchResult]
    query: str


# -----------------------------------------------------------------------
# Collections
# -----------------------------------------------------------------------


class RagLiteCollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    description: str | None = None


class RagLiteCollectionUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = None


class RagLiteCollectionDTO(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime


# -----------------------------------------------------------------------
# Jobs
# -----------------------------------------------------------------------


class RagLiteJobEventDTO(BaseModel):
    id: uuid.UUID
    event_type: str
    message: str
    created_at: datetime


class RagLiteJobDTO(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    status: str
    error_message: str | None = None
    stats_json: dict | None = None
    created_at: datetime
    updated_at: datetime
    events: list[RagLiteJobEventDTO] = []


# -----------------------------------------------------------------------
# Config / Health
# -----------------------------------------------------------------------


class RagLiteHealthResponse(BaseModel):
    status: str = "ok"
    mode: str = "lite"


class RagLiteConfigDTO(BaseModel):
    embedding_provider_id: str
    embedding_model: str
    embedding_dim: int
    chunk_max_tokens: int
    chunk_overlap_tokens: int
    file_store_root: str
    upload_max_file_mb: int

"""Pydantic schemas for admin RAG proxy endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RagChunkingPolicyDTO(BaseModel):
    """Chunking settings pushed to RAG runtime config."""

    max_tokens: int = Field(default=512, ge=64, le=4096)
    overlap: int = Field(default=64, ge=0, le=1024)
    by_headings: bool = False


class RagRuntimeConfigPayloadDTO(BaseModel):
    """Runtime embedding config payload."""

    embedding_provider: str = Field(min_length=1, max_length=64)
    embedding_model: str = Field(min_length=1, max_length=256)
    embedding_base_url: str | None = Field(default=None, max_length=1024)
    embedding_api_key_ref: str | None = Field(default=None, max_length=256)
    embedding_dim: int = Field(ge=8, le=4096)
    chunking_policy: RagChunkingPolicyDTO = Field(default_factory=RagChunkingPolicyDTO)


class RagRuntimeConfigUpdateRequest(BaseModel):
    """Admin request for updating RAG runtime config."""

    payload: RagRuntimeConfigPayloadDTO
    embedding_api_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=4096,
        description="Optional secret sent via header to RAG; never persisted in backend.",
    )


class RagRuntimeConfigResponse(BaseModel):
    """Runtime config response."""

    updated_at: datetime
    payload: RagRuntimeConfigPayloadDTO


class RagPrincipalsDTO(BaseModel):
    """Resolved principals for ACL filtering."""

    user_id: str = Field(min_length=1, max_length=256)
    group_ids: list[str] = Field(default_factory=list)


class RagSearchFiltersDTO(BaseModel):
    """Search filters payload."""

    sources: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    doc_types: list[str] = Field(default_factory=list)
    time_from: datetime | None = None
    time_to: datetime | None = None


class RagKnowledgeSearchRequestDTO(BaseModel):
    """Knowledge search request."""

    tenant_id: str = Field(min_length=1, max_length=256)
    workspace_id: str | None = Field(default=None, max_length=256)
    query: str = Field(min_length=1, max_length=8192)
    principals: RagPrincipalsDTO
    filters: RagSearchFiltersDTO = Field(default_factory=RagSearchFiltersDTO)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: str = Field(default="hybrid", pattern="^(vector|keyword|hybrid)$")
    debug: bool = False


class RagSearchResultDTO(BaseModel):
    """One search hit."""

    chunk_text: str
    doc_title: str | None = None
    source_uri: str
    section: str | None = None
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagKnowledgeSearchResponseDTO(BaseModel):
    """Search response payload."""

    results: list[RagSearchResultDTO]
    debug: dict[str, Any] | None = None


class RagCollectorSummaryDTO(BaseModel):
    """Collector summary."""

    id: str
    type: str
    enabled: bool
    schedule: str
    tenant_id: str
    workspace_id: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class RagCollectorListResponseDTO(BaseModel):
    """Collector list response."""

    collectors: list[RagCollectorSummaryDTO]


class RagAdminRunCollectorResponseDTO(BaseModel):
    """Collector run trigger response."""

    job_id: str
    source_id: str
    status: str


class RagIngestJobEventDTO(BaseModel):
    """One ingestion event."""

    event_type: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class RagIngestJobDTO(BaseModel):
    """Ingestion job status payload."""

    job_id: str
    collector_id: str
    source_id: str | None = None
    tenant_id: str
    workspace_id: str | None = None
    status: str
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    events: list[RagIngestJobEventDTO] = Field(default_factory=list)


class RagIngestJobListResponseDTO(BaseModel):
    """Ingestion jobs list response."""

    jobs: list[RagIngestJobDTO]


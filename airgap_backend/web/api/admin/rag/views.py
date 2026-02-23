"""Admin proxy endpoints for llm_port_rag internal APIs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from airgap_backend.db.models.users import User
from airgap_backend.services.rag.client import RagServiceClient, get_rag_client
from airgap_backend.web.api.admin.dependencies import require_superuser
from airgap_backend.web.api.admin.rag.schema import (
    RagAdminRunCollectorResponseDTO,
    RagCollectorListResponseDTO,
    RagIngestJobDTO,
    RagIngestJobListResponseDTO,
    RagKnowledgeSearchRequestDTO,
    RagKnowledgeSearchResponseDTO,
    RagRuntimeConfigResponse,
    RagRuntimeConfigUpdateRequest,
)

router = APIRouter()


@router.get("/health")
async def rag_health(
    _user: User = Depends(require_superuser),
    rag_client: RagServiceClient = Depends(get_rag_client),
) -> dict[str, str]:
    """Check backend-to-rag connectivity."""
    payload = await rag_client.health()
    status_value = str(payload.get("status", "unknown"))
    return {"status": status_value}


@router.get("/runtime-config", response_model=RagRuntimeConfigResponse)
async def get_runtime_config(
    _user: User = Depends(require_superuser),
    rag_client: RagServiceClient = Depends(get_rag_client),
) -> RagRuntimeConfigResponse:
    """Get active RAG runtime config."""
    payload = await rag_client.get_runtime_config()
    return RagRuntimeConfigResponse.model_validate(payload)


@router.post("/runtime-config", response_model=RagRuntimeConfigResponse)
async def update_runtime_config(
    body: RagRuntimeConfigUpdateRequest,
    _user: User = Depends(require_superuser),
    rag_client: RagServiceClient = Depends(get_rag_client),
) -> RagRuntimeConfigResponse:
    """Update active runtime config in llm_port_rag."""
    payload = await rag_client.update_runtime_config(
        payload=body.payload.model_dump(mode="json"),
        embedding_secret=body.embedding_api_key,
    )
    return RagRuntimeConfigResponse.model_validate(payload)


@router.post("/knowledge/search", response_model=RagKnowledgeSearchResponseDTO)
async def search_knowledge(
    body: RagKnowledgeSearchRequestDTO,
    _user: User = Depends(require_superuser),
    rag_client: RagServiceClient = Depends(get_rag_client),
) -> RagKnowledgeSearchResponseDTO:
    """Proxy ACL-aware knowledge search."""
    payload = await rag_client.search_knowledge(body.model_dump(mode="json"))
    return RagKnowledgeSearchResponseDTO.model_validate(payload)


@router.get("/collectors", response_model=RagCollectorListResponseDTO)
async def list_collectors(
    _user: User = Depends(require_superuser),
    rag_client: RagServiceClient = Depends(get_rag_client),
) -> RagCollectorListResponseDTO:
    """List configured collectors."""
    payload = await rag_client.list_collectors()
    return RagCollectorListResponseDTO.model_validate(payload)


@router.post("/collectors/{collector_id}/run", response_model=RagAdminRunCollectorResponseDTO)
async def run_collector(
    collector_id: str,
    _user: User = Depends(require_superuser),
    rag_client: RagServiceClient = Depends(get_rag_client),
) -> RagAdminRunCollectorResponseDTO:
    """Trigger immediate collector run."""
    payload = await rag_client.run_collector(collector_id)
    return RagAdminRunCollectorResponseDTO.model_validate(payload)


@router.get("/jobs", response_model=RagIngestJobListResponseDTO)
async def list_jobs(
    limit: int = Query(default=50, ge=1, le=200),
    _user: User = Depends(require_superuser),
    rag_client: RagServiceClient = Depends(get_rag_client),
) -> RagIngestJobListResponseDTO:
    """List ingestion jobs."""
    payload = await rag_client.list_jobs(limit=limit)
    return RagIngestJobListResponseDTO.model_validate(payload)


@router.get("/jobs/{job_id}", response_model=RagIngestJobDTO)
async def get_job(
    job_id: str,
    _user: User = Depends(require_superuser),
    rag_client: RagServiceClient = Depends(get_rag_client),
) -> RagIngestJobDTO:
    """Get one ingestion job."""
    payload = await rag_client.get_job(job_id)
    return RagIngestJobDTO.model_validate(payload)


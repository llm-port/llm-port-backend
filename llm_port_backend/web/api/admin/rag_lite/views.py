"""FastAPI endpoints for the RAG Lite admin API.

Registered under ``/admin/rag`` when the full RAG module is disabled
and RAG Lite is enabled — providing a lightweight, pgvector-only
knowledge-base experience.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from starlette import status

from llm_port_backend.db.dao.rag_lite_dao import (
    RagLiteChunkDAO,
    RagLiteCollectionDAO,
    RagLiteDocumentDAO,
    RagLiteIngestJobDAO,
)
from llm_port_backend.db.models.rag_lite import RagLiteIngestJob
from llm_port_backend.db.models.users import User
from llm_port_backend.services.rag_lite.tasks import rag_lite_ingest_task
from llm_port_backend.web.api.admin.rag_lite.schema import (
    RagLiteCollectionCreate,
    RagLiteCollectionDTO,
    RagLiteCollectionUpdate,
    RagLiteConfigDTO,
    RagLiteDocumentDetailDTO,
    RagLiteDocumentDTO,
    RagLiteHealthResponse,
    RagLiteJobDTO,
    RagLiteJobEventDTO,
    RagLiteSearchRequest,
    RagLiteSearchResponse,
    RagLiteSearchResult,
    RagLiteUploadResponse,
)
from llm_port_backend.web.api.rbac import require_permission

router = APIRouter()

# -----------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------


@router.get("/health", response_model=RagLiteHealthResponse)
async def rag_lite_health(
    _user: Annotated[User, Depends(require_permission("rag.search", "read"))],
) -> RagLiteHealthResponse:
    return RagLiteHealthResponse()


# -----------------------------------------------------------------------
# Config (read-only — managed via Settings page)
# -----------------------------------------------------------------------


@router.get("/config", response_model=RagLiteConfigDTO)
async def get_rag_lite_config(
    _user: Annotated[User, Depends(require_permission("rag.runtime", "read"))],
) -> RagLiteConfigDTO:
    from llm_port_backend.settings import settings  # noqa: PLC0415

    return RagLiteConfigDTO(
        embedding_provider_id=settings.rag_lite_embedding_provider_id,
        embedding_model=settings.rag_lite_embedding_model,
        embedding_dim=settings.rag_lite_embedding_dim,
        chunk_max_tokens=settings.rag_lite_chunk_max_tokens,
        chunk_overlap_tokens=settings.rag_lite_chunk_overlap_tokens,
        file_store_root=settings.rag_lite_file_store_root,
        upload_max_file_mb=settings.rag_lite_upload_max_file_mb,
    )


# -----------------------------------------------------------------------
# Upload
# -----------------------------------------------------------------------


@router.post("/upload", response_model=RagLiteUploadResponse)
async def upload_file(
    file: UploadFile,
    _user: Annotated[User, Depends(require_permission("rag.search", "write"))],
    document_dao: RagLiteDocumentDAO = Depends(),
    job_dao: RagLiteIngestJobDAO = Depends(),
    collection_id: uuid.UUID | None = None,
    request: "starlette.requests.Request" = None,  # type: ignore[assignment]  # noqa: F821
) -> RagLiteUploadResponse:
    from starlette.requests import Request as _Req  # noqa: PLC0415

    req: _Req = request  # type: ignore[assignment]
    rag_service = req.app.state.rag_lite_service

    # Read file bytes (enforce size limit)
    from llm_port_backend.settings import settings as _settings  # noqa: PLC0415

    max_mb = _settings.rag_lite_upload_max_file_mb
    file_bytes = await file.read()
    if len(file_bytes) > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large — max {max_mb} MB.",
        )

    doc, job = await rag_service.submit_file(
        file_bytes,
        file.filename or "upload",
        collection_id,
        document_dao=document_dao,
        job_dao=job_dao,
    )

    # Dispatch async ingest task
    await rag_lite_ingest_task.kiq(str(doc.id), str(job.id))

    return RagLiteUploadResponse(
        document_id=doc.id,
        job_id=job.id,
        filename=doc.filename,
        doc_type=doc.doc_type,
    )


# -----------------------------------------------------------------------
# Documents
# -----------------------------------------------------------------------


@router.get("/documents", response_model=list[RagLiteDocumentDTO])
async def list_documents(
    _user: Annotated[User, Depends(require_permission("rag.search", "read"))],
    document_dao: RagLiteDocumentDAO = Depends(),
    collection_id: uuid.UUID | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[RagLiteDocumentDTO]:
    docs = await document_dao.list_by_collection(
        collection_id,
        limit=limit,
        offset=offset,
    )
    return [
        RagLiteDocumentDTO(
            id=d.id,
            filename=d.filename,
            doc_type=d.doc_type,
            collection_id=d.collection_id,
            size_bytes=d.size_bytes,
            chunk_count=d.chunk_count,
            status=d.status.value,
            created_at=d.created_at,
        )
        for d in docs
    ]


@router.get("/documents/{document_id}", response_model=RagLiteDocumentDetailDTO)
async def get_document(
    document_id: uuid.UUID,
    _user: Annotated[User, Depends(require_permission("rag.search", "read"))],
    document_dao: RagLiteDocumentDAO = Depends(),
) -> RagLiteDocumentDetailDTO:
    doc = await document_dao.get(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return RagLiteDocumentDetailDTO(
        id=doc.id,
        filename=doc.filename,
        doc_type=doc.doc_type,
        collection_id=doc.collection_id,
        size_bytes=doc.size_bytes,
        chunk_count=doc.chunk_count,
        status=doc.status.value,
        created_at=doc.created_at,
        file_store_key=doc.file_store_key,
        sha256=doc.sha256,
        metadata_json=doc.metadata_json,
    )


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    _user: Annotated[User, Depends(require_permission("rag.search", "write"))],
    document_dao: RagLiteDocumentDAO = Depends(),
    chunk_dao: RagLiteChunkDAO = Depends(),
    request: "starlette.requests.Request" = None,  # type: ignore[assignment]  # noqa: F821
) -> None:
    from starlette.requests import Request as _Req  # noqa: PLC0415

    req: _Req = request  # type: ignore[assignment]
    rag_service = req.app.state.rag_lite_service
    deleted = await rag_service.delete_document(
        document_id,
        document_dao=document_dao,
        chunk_dao=chunk_dao,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")


# -----------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------


@router.post("/search", response_model=RagLiteSearchResponse)
async def search(
    body: RagLiteSearchRequest,
    _user: Annotated[User, Depends(require_permission("rag.search", "read"))],
    chunk_dao: RagLiteChunkDAO = Depends(),
    request: "starlette.requests.Request" = None,  # type: ignore[assignment]  # noqa: F821
) -> RagLiteSearchResponse:
    from starlette.requests import Request as _Req  # noqa: PLC0415

    req: _Req = request  # type: ignore[assignment]
    rag_service = req.app.state.rag_lite_service

    # Resolve embedding client at request time
    from llm_port_backend.services.rag_lite.embedding import EmbeddingClient  # noqa: PLC0415
    from llm_port_backend.services.system_settings.crypto import SettingsCrypto  # noqa: PLC0415
    from llm_port_backend.settings import settings  # noqa: PLC0415

    crypto = SettingsCrypto(settings.settings_master_key)
    pref_id_str = settings.rag_lite_embedding_provider_id
    pref_id = uuid.UUID(pref_id_str) if pref_id_str else None

    session = chunk_dao.session
    embedding_client = await EmbeddingClient.auto_detect(
        session,
        preferred_provider_id=pref_id,
        model_override=settings.rag_lite_embedding_model or None,
        dim=settings.rag_lite_embedding_dim,
        crypto=crypto,
    )

    results = await rag_service.search(
        body.query,
        chunk_dao=chunk_dao,
        embedding_client=embedding_client,
        top_k=body.top_k,
        collection_ids=body.collection_ids,
    )
    return RagLiteSearchResponse(
        query=body.query,
        results=[
            RagLiteSearchResult(
                chunk_text=r["chunk_text"],
                document_id=uuid.UUID(r["document_id"]),
                filename=r["filename"],
                chunk_index=r["chunk_index"],
                score=r["score"],
            )
            for r in results
        ],
    )


# -----------------------------------------------------------------------
# Collections
# -----------------------------------------------------------------------


@router.post("/collections", response_model=RagLiteCollectionDTO, status_code=201)
async def create_collection(
    body: RagLiteCollectionCreate,
    _user: Annotated[User, Depends(require_permission("rag.search", "write"))],
    collection_dao: RagLiteCollectionDAO = Depends(),
) -> RagLiteCollectionDTO:
    col = await collection_dao.create(body.name, body.description)
    return RagLiteCollectionDTO(
        id=col.id,
        name=col.name,
        description=col.description,
        created_at=col.created_at,
        updated_at=col.updated_at,
    )


@router.get("/collections", response_model=list[RagLiteCollectionDTO])
async def list_collections(
    _user: Annotated[User, Depends(require_permission("rag.search", "read"))],
    collection_dao: RagLiteCollectionDAO = Depends(),
) -> list[RagLiteCollectionDTO]:
    cols = await collection_dao.list_all()
    return [
        RagLiteCollectionDTO(
            id=c.id,
            name=c.name,
            description=c.description,
            created_at=c.created_at,
            updated_at=c.updated_at,
        )
        for c in cols
    ]


@router.patch("/collections/{collection_id}", response_model=RagLiteCollectionDTO)
async def update_collection(
    collection_id: uuid.UUID,
    body: RagLiteCollectionUpdate,
    _user: Annotated[User, Depends(require_permission("rag.search", "write"))],
    collection_dao: RagLiteCollectionDAO = Depends(),
) -> RagLiteCollectionDTO:
    col = await collection_dao.update(
        collection_id,
        name=body.name,
        description=body.description,
    )
    if col is None:
        raise HTTPException(status_code=404, detail="Collection not found")
    return RagLiteCollectionDTO(
        id=col.id,
        name=col.name,
        description=col.description,
        created_at=col.created_at,
        updated_at=col.updated_at,
    )


@router.delete("/collections/{collection_id}", status_code=204)
async def delete_collection(
    collection_id: uuid.UUID,
    _user: Annotated[User, Depends(require_permission("rag.search", "write"))],
    collection_dao: RagLiteCollectionDAO = Depends(),
) -> None:
    deleted = await collection_dao.delete(collection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Collection not found")


# -----------------------------------------------------------------------
# Jobs
# -----------------------------------------------------------------------


def _job_to_dto(job: RagLiteIngestJob) -> RagLiteJobDTO:
    events = [
        RagLiteJobEventDTO(
            id=e.id,
            event_type=e.event_type.value,
            message=e.message,
            created_at=e.created_at,
        )
        for e in (job.events or [])
    ]
    return RagLiteJobDTO(
        id=job.id,
        document_id=job.document_id,
        status=job.status.value,
        error_message=job.error_message,
        stats_json=job.stats_json,
        created_at=job.created_at,
        updated_at=job.updated_at,
        events=events,
    )


@router.get("/jobs", response_model=list[RagLiteJobDTO])
async def list_jobs(
    _user: Annotated[User, Depends(require_permission("rag.search", "read"))],
    job_dao: RagLiteIngestJobDAO = Depends(),
    limit: int = 50,
) -> list[RagLiteJobDTO]:
    jobs = await job_dao.list_recent(limit=limit)
    return [_job_to_dto(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=RagLiteJobDTO)
async def get_job(
    job_id: uuid.UUID,
    _user: Annotated[User, Depends(require_permission("rag.search", "read"))],
    job_dao: RagLiteIngestJobDAO = Depends(),
) -> RagLiteJobDTO:
    job = await job_dao.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_dto(job)

"""HTTP client for llm_port_rag internal APIs."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException
from starlette import status

from airgap_backend.settings import settings


class RagServiceClient:
    """Thin typed wrapper around llm_port_rag internal endpoints."""

    def __init__(
        self,
        base_url: str,
        service_token: str,
        timeout_sec: float = 30.0,
        runtime_secret_header_name: str = "x-embedding-api-key",  # noqa: S107
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.service_token = service_token
        self.timeout_sec = timeout_sec
        self.runtime_secret_header_name = runtime_secret_header_name

    async def _request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        if not self.service_token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG integration is not configured (missing service token).",
            )

        request_headers = {
            "Authorization": f"Bearer {self.service_token}",
            "Content-Type": "application/json",
        }
        if headers:
            request_headers.update(headers)

        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    json=json_body,
                    headers=request_headers,
                )
            except httpx.TimeoutException as exc:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail=f"RAG service timed out calling {path}.",
                ) from exc
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to reach RAG service: {exc}",
                ) from exc

        if response.status_code >= 400:
            detail = response.text
            try:
                payload = response.json()
                if isinstance(payload, dict) and "detail" in payload:
                    detail = str(payload["detail"])
            except ValueError:
                pass
            raise HTTPException(status_code=response.status_code, detail=detail)

        if response.status_code == status.HTTP_204_NO_CONTENT:
            return None
        return response.json()

    async def health(self) -> dict[str, Any]:
        """Check internal knowledge health endpoint."""
        return await self._request("GET", "/internal/knowledge/health")

    async def get_runtime_config(self) -> dict[str, Any]:
        """Get active runtime config."""
        return await self._request("GET", "/internal/runtime-config")

    async def update_runtime_config(
        self,
        payload: dict[str, Any],
        embedding_secret: str | None = None,
    ) -> dict[str, Any]:
        """Update runtime config and optional embedding secret header."""
        extra_headers: dict[str, str] = {}
        if embedding_secret:
            extra_headers[self.runtime_secret_header_name] = embedding_secret
        return await self._request(
            "POST",
            "/internal/runtime-config",
            json_body=payload,
            headers=extra_headers,
        )

    async def search_knowledge(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Proxy knowledge search request."""
        return await self._request(
            "POST",
            "/internal/knowledge/search",
            json_body=payload,
        )

    async def list_collectors(self) -> dict[str, Any]:
        """List configured collectors."""
        return await self._request("GET", "/internal/admin/collectors")

    async def run_collector(self, collector_id: str) -> dict[str, Any]:
        """Trigger immediate collector run."""
        return await self._request("POST", f"/internal/admin/collectors/{collector_id}/run")

    async def list_jobs(self, limit: int = 50) -> dict[str, Any]:
        """List ingestion jobs."""
        return await self._request("GET", f"/internal/admin/jobs?limit={limit}")

    async def get_job(self, job_id: str) -> dict[str, Any]:
        """Fetch one ingestion job."""
        return await self._request("GET", f"/internal/admin/jobs/{job_id}")


def get_rag_client() -> RagServiceClient:
    """Build a request-scoped RAG client from settings."""
    return RagServiceClient(
        base_url=settings.rag_base_url,
        service_token=settings.rag_service_token,
        timeout_sec=settings.rag_timeout_sec,
        runtime_secret_header_name=settings.rag_runtime_secret_header_name,
    )

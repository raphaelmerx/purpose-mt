"""
Label Studio API client using requests.

Handles authentication with both legacy tokens and JWT refresh tokens.
"""
from datetime import datetime, timezone

import jwt
import requests


class LabelStudioClient:
    """Simple Label Studio API client."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key.strip('"').strip("'")
        self._access_token: str | None = None
        self._is_jwt = self._check_is_jwt(self._api_key)
    
    @staticmethod
    def _check_is_jwt(token: str) -> bool:
        """Check if a token is a valid JWT token."""
        try:
            jwt.decode(token, options={"verify_signature": False})
            return True
        except jwt.InvalidTokenError:
            return False
    
    @staticmethod
    def _is_token_valid(token: str) -> bool:
        """Check if a JWT token is still valid (not expired)."""
        try:
            decoded = jwt.decode(token, options={"verify_signature": False})
            expiration = decoded.get("exp")
            if expiration is None:
                return False
            expiration_time = datetime.fromtimestamp(expiration, timezone.utc)
            return expiration_time > datetime.now(timezone.utc)
        except jwt.InvalidTokenError:
            return False
    
    def _refresh_access_token(self) -> str:
        """Refresh the access token using the refresh token."""
        response = requests.post(
            f"{self.base_url}/api/token/refresh/",
            json={"refresh": self._api_key},
            headers={"Content-Type": "application/json"},
        )
        
        if response.status_code == 200:
            self._access_token = response.json()["access"]
            return self._access_token
        else:
            raise Exception(f"Failed to refresh token: {response.status_code} - {response.text}")
    
    def _get_token(self) -> str:
        """Get a valid API token, refreshing if necessary."""
        if not self._is_jwt:
            return self._api_key
        
        if self._access_token and self._is_token_valid(self._access_token):
            return self._access_token
        
        return self._refresh_access_token()
    
    def _get_headers(self) -> dict:
        """Get headers for API requests."""
        token = self._get_token()
        auth_prefix = "Bearer" if self._is_jwt else "Token"
        return {
            "Authorization": f"{auth_prefix} {token}",
            "Content-Type": "application/json"
        }
    
    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make an authenticated API request."""
        url = f"{self.base_url}/api/{endpoint.lstrip('/')}"
        return requests.request(method, url, headers=self._get_headers(), **kwargs)
    
    # Project methods
    
    def list_projects(self) -> list[dict]:
        """List all projects."""
        response = self._request("GET", "/projects")
        response.raise_for_status()
        data = response.json()
        return data.get("results", data) if isinstance(data, dict) else data
    
    def get_project_by_title(self, title: str) -> dict | None:
        """Find a project by title."""
        for project in self.list_projects():
            if title in project.get("title"):
                return project
        return None
    
    def create_project(self, title: str, label_config: str, description: str = "") -> dict:
        """Create a new project."""
        response = self._request(
            "POST", "/projects",
            json={
                "title": title,
                "description": description,
                "label_config": label_config,
                "is_published": True,
            }
        )
        
        if response.status_code == 401:
            raise PermissionError(
                f"Authentication failed. Check your LABEL_STUDIO_API_KEY.\n"
                f"Get a token from {self.base_url}/user/account"
            )
        
        response.raise_for_status()
        return response.json()
    
    def update_project(self, project_id: int, **kwargs) -> dict:
        """Update a project's settings.
        
        Args:
            project_id: The project ID
            **kwargs: Fields to update (title, description, label_config, etc.)
        
        Returns:
            Updated project data
        """
        response = self._request(
            "PATCH", f"/projects/{project_id}",
            json=kwargs
        )
        if not response.ok:
            raise Exception(f"Update failed: {response.status_code} - {response.text}")
        return response.json()
    
    def delete_all_tasks(self, project_id: int) -> dict:
        """Delete all tasks (and their annotations) from a project.
        
        Args:
            project_id: The project ID
        
        Returns:
            Deletion result
        """
        response = self._request(
            "DELETE", f"/projects/{project_id}/tasks"
        )
        if not response.ok:
            raise Exception(f"Delete failed: {response.status_code} - {response.text}")
        return {"status": "deleted"}
    
    def validate_label_config(self, project_id: int, label_config: str) -> dict:
        """Validate a labeling config before applying it.
        
        Args:
            project_id: The project ID
            label_config: The XML labeling config to validate
        
        Returns:
            Validation result
        """
        response = self._request(
            "POST", f"/projects/{project_id}/validate-label-config",
            json={"label_config": label_config}
        )
        if not response.ok:
            raise Exception(f"Validation failed: {response.status_code} - {response.text}")
        return response.json()
    
    # Task methods
    
    def import_tasks(self, project_id: int, tasks: list[dict]) -> dict:
        """Import tasks into a project."""
        response = self._request(
            "POST", f"/projects/{project_id}/import",
            json=tasks
        )
        response.raise_for_status()
        return response.json()
    
    def export_tasks(self, project_id: int, export_type: str = "JSON") -> list[dict]:
        """Export all tasks with annotations from a project.
        
        Args:
            project_id: The project ID
            export_type: Export format (JSON, CSV, etc.)
        
        Returns:
            List of tasks with their annotations
        """
        response = self._request(
            "GET", f"/projects/{project_id}/export",
            params={"exportType": export_type}
        )
        response.raise_for_status()
        return response.json()

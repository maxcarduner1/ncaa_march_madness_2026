import os
from databricks.sdk import WorkspaceClient

IS_DATABRICKS_APP = bool(os.environ.get("DATABRICKS_APP_NAME"))

CATALOG = os.environ.get("UC_CATALOG", "serverless_stable_82l7qq_catalog")
SCHEMA = os.environ.get("UC_SCHEMA", "march_madness_2026")
WAREHOUSE_HTTP_PATH = os.environ.get("DATABRICKS_WAREHOUSE_HTTP_PATH", "/sql/1.0/warehouses/aa2256da874c652c")


def get_workspace_client() -> WorkspaceClient:
    if IS_DATABRICKS_APP:
        return WorkspaceClient()
    profile = os.environ.get("DATABRICKS_PROFILE", "sandbox")
    return WorkspaceClient(profile=profile)


def get_workspace_host() -> str:
    if IS_DATABRICKS_APP:
        host = os.environ.get("DATABRICKS_HOST", "")
        if host and not host.startswith("http"):
            host = f"https://{host}"
        return host
    w = get_workspace_client()
    return w.config.host


def get_access_token() -> str:
    w = get_workspace_client()
    auth_headers = w.config.authenticate()
    if auth_headers and "Authorization" in auth_headers:
        return auth_headers["Authorization"].replace("Bearer ", "")
    raise RuntimeError("Could not obtain access token")

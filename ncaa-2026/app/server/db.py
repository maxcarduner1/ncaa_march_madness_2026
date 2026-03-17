import databricks.sql as dbsql
from .config import get_workspace_host, get_access_token, WAREHOUSE_HTTP_PATH
import pandas as pd


def get_connection():
    host = get_workspace_host().replace("https://", "").replace("http://", "")
    token = get_access_token()
    return dbsql.connect(
        server_hostname=host,
        http_path=WAREHOUSE_HTTP_PATH,
        access_token=token,
    )


def query_df(sql: str) -> pd.DataFrame:
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            cols = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=cols)

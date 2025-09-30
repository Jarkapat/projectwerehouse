from typing import Dict, Any, List, Optional
import pandas as pd
from clickhouse_connect import get_client
from clickhouse_connect.driver.exceptions import OperationalError

class CH:
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        try:
            self.client = get_client(host=host, port=port, username=user, password=password, database=database)
            self.client.command("SELECT 1")
        except OperationalError as e:
            hint = (
                f"Cannot connect to ClickHouse at {host}:{port} / DB={database}. "
                f"ตรวจสอบว่า HTTP port=8123 และ DB (edu) มีอยู่จริง. "
                f"ลอง: curl http://{host}:{port}/ping → ควรได้ 'Ok.'"
            )
            raise OperationalError(f"{e}\n\nHINT: {hint}") from e

    def df(self, sql: str, settings: Optional[dict] = None) -> pd.DataFrame:
        base_settings = {"use_query_cache": 0, "max_result_rows": 0}
        if settings:
            base_settings.update(settings)
        res = self.client.query(sql, settings=base_settings)
        return pd.DataFrame(res.result_rows, columns=res.column_names)

    def insert(self, table: str, row: Dict[str, Any]):
        cols = list(row.keys())
        vals = [row[c] for c in cols]
        self.client.insert(table, [vals], column_names=cols)

    def insert_df(self, table: str, df: pd.DataFrame, cols: List[str]):
        if df.empty:
            return
        self.client.insert(table, df[cols].values.tolist(), column_names=cols)

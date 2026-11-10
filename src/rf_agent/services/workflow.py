from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

from rf_agent.contracts import DataLabel, Phase
from rf_agent.services import mock_provider as mock

DB_PATH = Path("data/rf_agent.db")
MAX_UPLOAD = 5_000_000
ALLOWED_EXTENSIONS = {".csv", ".json"}


class WorkflowService:
    """Application boundary: all UI and HTTP operations go through this service."""

    def _connect(self) -> sqlite3.Connection:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(DB_PATH, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("""CREATE TABLE IF NOT EXISTS ui_runs (
            run_id TEXT PRIMARY KEY, snapshot_json TEXT NOT NULL, created_at TEXT NOT NULL)""")
        connection.execute("""CREATE TABLE IF NOT EXISTS ui_artifacts (
            artifact_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, name TEXT NOT NULL,
            content BLOB NOT NULL, media_type TEXT NOT NULL, sha256 TEXT NOT NULL,
            data_kind TEXT NOT NULL, created_at TEXT NOT NULL)""")
        connection.execute("""CREATE TABLE IF NOT EXISTS ui_actions (
            idempotency_key TEXT PRIMARY KEY, run_id TEXT NOT NULL, action TEXT NOT NULL)""")
        return connection

    def _save(self, snapshot: dict) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO ui_runs VALUES(?,?,?) ON CONFLICT(run_id) DO UPDATE SET snapshot_json=excluded.snapshot_json",
                (
                    snapshot["run_id"],
                    json.dumps(snapshot, ensure_ascii=False),
                    snapshot["created_at"],
                ),
            )

    def create_run(self, request: str) -> dict:
        spec = mock.design_spec(request)
        run_id = f"run-{uuid.uuid4().hex[:10]}"
        snapshot = {
            "run_id": run_id,
            "created_at": mock.iso_now(),
            "updated_at": mock.iso_now(),
            "request": request,
            "phase": Phase.CONFIRM.value,
            "status": "waiting_approval",
            "data_kind": DataLabel.SIMULATED.value,
            "spec": spec.model_dump(mode="json"),
            "evidence": [item.model_dump(mode="json") for item in mock.evidence()],
            "pending_confirmations": list(spec.needs_confirmation),
            "approval": None,
            "sample_batch": None,
            "errors": [],
        }
        self._save(snapshot)
        return snapshot

    def _get(self, run_id: str) -> dict:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT snapshot_json FROM ui_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return json.loads(row[0])

    def snapshot(self, run_id: str) -> dict:
        return self._get(run_id)

    def list_runs(self) -> list[dict]:
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT snapshot_json FROM ui_runs ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        return [
            {
                k: item[k]
                for k in (
                    "run_id",
                    "created_at",
                    "updated_at",
                    "phase",
                    "status",
                    "data_kind",
                    "request",
                )
            }
            for row in rows
            if (item := json.loads(row[0]))
        ]

    def resolve_approval(self, run_id: str, action: str, reason: str, key: str) -> dict:
        snapshot = self._get(run_id)
        with closing(self._connect()) as db, db:
            previous = db.execute(
                "SELECT action FROM ui_actions WHERE idempotency_key=?", (key,)
            ).fetchone()
            if previous:
                if previous[0] != action:
                    raise ValueError("idempotency key already used for another action")
                return snapshot
            if snapshot["phase"] != Phase.CONFIRM.value:
                raise ValueError("run is not awaiting approval")
            db.execute("INSERT INTO ui_actions VALUES(?,?,?)", (key, run_id, action))
        snapshot["approval"] = {"action": action, "reason": reason, "at": mock.iso_now()}
        snapshot["pending_confirmations"] = []
        snapshot["updated_at"] = mock.iso_now()
        if action == "approve":
            spec = __import__(
                "rf_agent.contracts", fromlist=["DesignSpec"]
            ).DesignSpec.model_validate(snapshot["spec"])
            batch = mock.sample(spec)
            snapshot["sample_batch"] = batch.model_dump(mode="json")
            snapshot["phase"] = Phase.SUBMIT.value
            snapshot["status"] = "sampled"
        else:
            snapshot["phase"] = Phase.FAILED.value
            snapshot["status"] = "rejected"
        self._save(snapshot)
        return snapshot

    def ingest(
        self,
        run_id: str,
        filename: str,
        content: bytes,
        data_kind: DataLabel = DataLabel.SIMULATION,
    ) -> dict:
        self._get(run_id)
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError("only CSV and JSON files are accepted")
        if not content or len(content) > MAX_UPLOAD:
            raise ValueError("file must be between 1 byte and 5 MB")
        try:
            if suffix == ".csv":
                rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
                if not rows or not {"frequency_hz", "s11_db", "s21_db"}.issubset(rows[0]):
                    raise ValueError("CSV requires frequency_hz, s11_db, and s21_db columns")
                for row in rows:
                    for col in ("frequency_hz", "s11_db", "s21_db"):
                        float(row[col])
            else:
                data = json.loads(content)
                if not isinstance(data, list) or not data:
                    raise ValueError("JSON must be a non-empty array of result rows")
                rows = data
                if not {"frequency_hz", "s11_db", "s21_db"}.issubset(rows[0]):
                    raise ValueError("JSON rows require frequency_hz, s11_db, and s21_db")
        except (UnicodeDecodeError, json.JSONDecodeError, csv.Error) as exc:
            raise ValueError(f"invalid {suffix[1:].upper()} file") from exc
        for row in rows[:100_000]:
            try:
                [float(row[key]) for key in ("frequency_hz", "s11_db", "s21_db")]
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError("result values must be numeric") from exc
        digest = hashlib.sha256(content).hexdigest()
        artifact_id = f"artifact-{uuid.uuid4().hex[:10]}"
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO ui_artifacts VALUES(?,?,?,?,?,?,?,?)",
                (
                    artifact_id,
                    run_id,
                    Path(filename).name,
                    content,
                    "text/csv" if suffix == ".csv" else "application/json",
                    digest,
                    data_kind.value,
                    mock.iso_now(),
                ),
            )
        return {
            "artifact_id": artifact_id,
            "run_id": run_id,
            "name": Path(filename).name,
            "sha256": digest,
            "rows": len(rows),
            "data_kind": DataLabel.SIMULATED.value,
        }

    def artifacts(self, run_id: str) -> list[dict]:
        self._get(run_id)
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT artifact_id, name, sha256, data_kind, created_at FROM ui_artifacts WHERE run_id=? ORDER BY created_at DESC",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def download(self, run_id: str, artifact_id: str) -> tuple[str, bytes, str]:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT name, content, media_type FROM ui_artifacts WHERE run_id=? AND artifact_id=?",
                (run_id, artifact_id),
            ).fetchone()
        if row is None:
            raise KeyError(artifact_id)
        return row[0], row[1], row[2]

    def mock_artifact(self) -> tuple[str, bytes, str]:
        return mock.artifacts()

    def report(self, run_id: str, format: str = "md") -> bytes:
        snapshot = self._get(run_id)
        rows = self.artifacts(run_id)
        if format == "csv":
            out = io.StringIO()
            writer = csv.writer(out)
            writer.writerow(["run_id", "artifact", "sha256", "data_kind"])
            for item in rows:
                writer.writerow([run_id, item["name"], item["sha256"], item["data_kind"]])
            return out.getvalue().encode()
        lines = [
            f"# RF AI Agent Report: {run_id}",
            "",
            f"- Status: {snapshot['status']}",
            f"- Phase: {snapshot['phase']}",
            f"- data_kind: {snapshot['data_kind']}",
            "",
            "## Artifacts",
        ]
        lines.extend(
            f"- {item['name']} | SHA-256: `{item['sha256']}` | data_kind: `{item['data_kind']}`"
            for item in rows
        )
        if not rows:
            lines.append("- No uploaded artifacts")
        return "\n".join(lines).encode()


workflow = WorkflowService()

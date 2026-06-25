from dataclasses import asdict
from pathlib import Path
import os
import shutil

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from infrastructure.adapter.SQLiteActivityLedgerAdapter import SQLiteActivityLedgerAdapter


CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
os.chdir(PROJECT_ROOT)

TEMPLATES = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

DEFAULT_ACTIVITY_LEDGER_DB_PATH = PROJECT_ROOT / "database" / "activity_ledger.db"
DEFAULT_INTERACTION_LOG_DB_PATH = PROJECT_ROOT / "database" / "interaction_logs.db"


def _run_payload(run) -> dict:
    return asdict(run)


def _detail_payload(detail) -> dict:
    return {
        "run": asdict(detail.run),
        "steps": [asdict(step) for step in detail.steps],
        "artifacts": [asdict(item) for item in detail.artifacts],
        "links": [asdict(item) for item in detail.links],
        "tags": list(detail.tags),
        "traces": [asdict(item) for item in detail.traces],
    }


def create_app(
    *,
    activity_ledger: SQLiteActivityLedgerAdapter | None = None,
    activity_ledger_db_path: str | None = None,
    interaction_log_db_path: str | None = None,
) -> FastAPI:
    ledger = activity_ledger or SQLiteActivityLedgerAdapter(
        db_path=activity_ledger_db_path or str(DEFAULT_ACTIVITY_LEDGER_DB_PATH),
        interaction_log_db_path=interaction_log_db_path or str(DEFAULT_INTERACTION_LOG_DB_PATH),
    )
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")
    app.state.activity_ledger = ledger

    @app.get("/")
    @app.get("/home/")
    def home(request: Request):
        summary = ledger.summarize()
        recent_runs = ledger.list_runs(limit=10)
        return TEMPLATES.TemplateResponse(
            request,
            "index.html",
            {
                "summary": summary,
                "recent_runs": recent_runs,
            },
        )

    @app.get("/upload-form/")
    def upload_form(request: Request):
        return TEMPLATES.TemplateResponse(request, "upload_form.html", {})

    @app.post("/upload-audio/")
    async def upload_audio(request: Request, file: UploadFile = File(...)):
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return RedirectResponse(url=f"/upload-success?filename={file.filename}", status_code=303)

    @app.get("/upload-success")
    def upload_success(request: Request, filename: str):
        return TEMPLATES.TemplateResponse(
            request,
            "upload_form.html",
            {
                "message": f"File '{filename}' uploaded successfully!",
            },
        )

    @app.get("/activity/")
    def activity_dashboard(
        request: Request,
        status: str | None = None,
        source_kind: str | None = None,
        trigger_kind: str | None = None,
        tag: str | None = None,
        q: str | None = None,
    ):
        runs = ledger.list_runs(
            status=status,
            source_kind=source_kind,
            trigger_kind=trigger_kind,
            tag=tag,
            query=q,
            limit=100,
        )
        return TEMPLATES.TemplateResponse(
            request,
            "activity_list.html",
            {
                "runs": runs,
                "filters": {
                    "status": status or "",
                    "source_kind": source_kind or "",
                    "trigger_kind": trigger_kind or "",
                    "tag": tag or "",
                    "q": q or "",
                },
            },
        )

    @app.get("/activity/{run_id}")
    def activity_run_detail(request: Request, run_id: str):
        detail = ledger.get_run_detail(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return TEMPLATES.TemplateResponse(
            request,
            "activity_detail.html",
            {
                "detail": detail,
            },
        )

    @app.get("/api/activity/runs")
    def list_activity_runs(
        status: str | None = None,
        source_kind: str | None = None,
        trigger_kind: str | None = None,
        tag: str | None = None,
        q: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = Query(default=50, le=200),
    ):
        runs = ledger.list_runs(
            status=status,
            source_kind=source_kind,
            trigger_kind=trigger_kind,
            tag=tag,
            query=q,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )
        return {"items": [_run_payload(run) for run in runs]}

    @app.get("/api/activity/runs/{run_id}")
    def get_activity_run(run_id: str):
        detail = ledger.get_run_detail(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return _detail_payload(detail)

    @app.get("/api/activity/artifacts/{artifact_id}")
    def get_activity_artifact(artifact_id: str):
        artifact = ledger.get_artifact(artifact_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return asdict(artifact)

    @app.get("/api/activity/search")
    def search_activity_runs(q: str = Query(default="", min_length=1), limit: int = Query(default=50, le=200)):
        runs = ledger.search_runs(q, limit=limit)
        return {"items": [_run_payload(run) for run in runs]}

    @app.get("/api/activity/summary")
    def activity_summary():
        return ledger.summarize()

    return app


app = create_app()

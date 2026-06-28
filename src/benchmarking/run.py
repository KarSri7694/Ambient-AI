import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmarking.case_loader import build_inline_case, load_cases, load_suite
from benchmarking.provider import BenchmarkingLLMProvider
from benchmarking.services import available_services, run_service_case, score_case
from config import CONFIG
from core.models import BenchmarkResult, BenchmarkRun
from infrastructure.adapter.SQLiteBenchmarkAdapter import SQLiteBenchmarkAdapter
from infrastructure.adapter.llamaCppAdapter import LlamaCppAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run subsystem benchmarks against local models.")
    parser.add_argument("--service", required=False, help="Benchmark service name.")
    parser.add_argument("--case", action="append", dest="cases", help="Optional case_id filter. Repeatable.")
    parser.add_argument("--models", required=False, help="Comma-separated model names.")
    parser.add_argument("--suite", required=False, help="Path to a suite JSON containing models plus screenshots/transcripts.")
    parser.add_argument("--screenshot-path", required=False, help="Direct screenshot path for passive_observer.")
    parser.add_argument("--transcript-path", required=False, help="Direct transcript path for reflection_service.")
    parser.add_argument("--cases-root", default=str(REPO_ROOT / "benchmarking" / "cases"))
    parser.add_argument(
        "--db-path",
        default=CONFIG.get_str("benchmarking", "db_path", str(REPO_ROOT / "database" / "benchmarking.db")),
    )
    parser.add_argument("--api-base-url", default=CONFIG.get_str("runtime", "api_base_url", "http://localhost:8080"))
    parser.add_argument("--list-services", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_services:
        for name in available_services():
            print(name)
        return 0

    suite = load_suite(args.suite) if args.suite else None

    if not args.service:
        if suite is None:
            parser.error("--service is required unless --list-services or --suite is used.")
        args.service = suite.service

    all_cases = load_cases(args.cases_root, service=args.service)
    if args.list_cases:
        for case in all_cases:
            print(f"{case.case_id}\t{case.title}")
        return 0

    if not args.models and suite is None:
        parser.error("--models is required unless --list-services, --list-cases, or --suite is used.")

    selected_case_ids = {item.strip() for item in (args.cases or []) if item and item.strip()}
    if suite is not None:
        if args.screenshot_path or args.transcript_path or selected_case_ids:
            parser.error("--suite cannot be combined with --screenshot-path, --transcript-path, or --case.")
        cases = suite.cases
    elif args.screenshot_path:
        if args.service != "passive_observer":
            parser.error("--screenshot-path is only supported with --service passive_observer.")
        cases = [
            build_inline_case(
                service="passive_observer",
                screenshot_path=args.screenshot_path,
            )
        ]
    elif args.transcript_path:
        if args.service != "reflection_service":
            parser.error("--transcript-path is only supported with --service reflection_service.")
        cases = [
            build_inline_case(
                service="reflection_service",
                transcript_path=args.transcript_path,
            )
        ]
    else:
        cases = [case for case in all_cases if not selected_case_ids or case.case_id in selected_case_ids]
    if not cases:
        if args.service == "passive_observer":
            print("No benchmark cases matched the request. Use --screenshot-path for a direct passive observer run.")
        else:
            print("No benchmark cases matched the request.")
        return 1

    models = suite.models if suite is not None else [item.strip() for item in str(args.models).split(",") if item.strip()]
    if not models:
        print("No models were provided.")
        return 1

    store = SQLiteBenchmarkAdapter(args.db_path)
    run_id = uuid.uuid4().hex
    run = BenchmarkRun(
        run_id=run_id,
        created_at=datetime.now().isoformat(),
        completed_at=None,
        service_name=args.service,
        model_names_json=json.dumps(models, ensure_ascii=False),
        case_ids_json=json.dumps([case.case_id for case in cases], ensure_ascii=False),
        status="running",
        notes=None,
    )
    store.insert_run(run)

    provider = LlamaCppAdapter(base_url=args.api_base_url)
    benchmark_provider = BenchmarkingLLMProvider(provider)
    overall_status = "completed"

    try:
        for model_name in models:
            benchmark_provider.reset_benchmark_metrics()
            import asyncio

            asyncio.run(provider.load_model(model_name))
            for case in cases:
                benchmark_provider.reset_benchmark_metrics()
                started_at = datetime.now().isoformat()
                status = "completed"
                error_text = None
                response_text = None
                structured_output_json = None
                screenshot_path = case.inputs.get("screenshot_path")
                transcript_path = case.inputs.get("transcript_path")
                auto_score = None
                auto_score_details_json = None
                metadata = {"case_source_path": case.source_path, "rubric_notes": case.rubric_notes}
                try:
                    execution = run_service_case(case, benchmark_provider, model_name)
                    response_text = execution.response_text
                    structured_output_json = execution.structured_output_json
                    screenshot_path = execution.screenshot_path or screenshot_path
                    transcript_path = execution.transcript_path or transcript_path
                    auto_score, details = score_case(case, execution)
                    auto_score_details_json = json.dumps(details, ensure_ascii=False, indent=2)
                    if execution.metadata:
                        metadata.update(execution.metadata)
                except Exception as exc:
                    overall_status = "completed_with_errors"
                    status = "error"
                    error_text = str(exc)

                metrics = benchmark_provider.benchmark_metrics()
                result = BenchmarkResult(
                    result_id=uuid.uuid4().hex,
                    run_id=run_id,
                    created_at=started_at,
                    completed_at=datetime.now().isoformat(),
                    service_name=case.service,
                    case_id=case.case_id,
                    case_title=case.title,
                    model_name=model_name,
                    screenshot_path=str(screenshot_path) if screenshot_path else None,
                    transcript_path=str(transcript_path) if transcript_path else None,
                    response_text=response_text,
                    structured_output_json=structured_output_json,
                    error_text=error_text,
                    prompt_tokens=metrics.prompt_tokens,
                    completion_tokens=metrics.completion_tokens,
                    total_tokens=metrics.total_tokens,
                    prefill_seconds=metrics.prefill_seconds,
                    generation_seconds=metrics.generation_seconds,
                    prefill_tokens_per_second=metrics.prefill_tokens_per_second,
                    generation_tokens_per_second=metrics.generation_tokens_per_second,
                    token_count_method=metrics.token_count_method,
                    auto_score=auto_score,
                    auto_score_details_json=auto_score_details_json,
                    metadata_json=json.dumps(
                        {
                            **metadata,
                            "call_count": metrics.call_count,
                            "calls_json": json.loads(metrics.calls_json or "[]"),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    status=status,
                )
                store.insert_result(result)
                print(
                    f"[{status}] service={case.service} case={case.case_id} model={model_name} "
                    f"score={auto_score if auto_score is not None else 'n/a'} "
                    f"prefill_tps={metrics.prefill_tokens_per_second:.3f} "
                    f"total_tokens={metrics.total_tokens} "
                    f"gen_tps={metrics.generation_tokens_per_second:.3f}"
                )
    finally:
        import asyncio

        try:
            asyncio.run(provider.unload_model())
        except Exception:
            pass
        store.update_run_status(
            run_id,
            status=overall_status,
            completed_at=datetime.now().isoformat(),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import requests
from fastmcp import FastMCP
from todoist_api_python.api import TodoistAPI
import asyncio
import json
import os
import sqlite3
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import pickle
import datetime
import uuid
from google import genai
from typing import Annotated
import serpapi
import night_mode
from utils.threading_util import run_async
import yt_dlp
from application.services.semantic_deduplication_service import SemanticDeduplicationService
from config import CONFIG
from infrastructure.adapter.LoggingLLMProvider import LoggingLLMProvider
from infrastructure.adapter.llamaCppAdapter import LlamaCppAdapter
from infrastructure.adapter.SQLiteInteractionLogAdapter import SQLiteInteractionLogAdapter
from infrastructure.adapter.SQLiteMemoryAdapter import SQLiteMemoryAdapter
import csv
import subprocess

mcp = FastMCP("My MCP Server")

TODOIST_API_TOKEN = os.environ.get("TODOIST_API_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

SCOPES = ['https://www.googleapis.com/auth/calendar.events']
USER_DATA_DIR = CONFIG.get_str("runtime", "user_data_dir", "D:\\USER_DATA")
MEMORY_DB_PATH = os.path.join(USER_DATA_DIR, "database", "memory.db")
MEMORY_ROOT = os.path.join(USER_DATA_DIR, "memory")
INTERACTION_LOG_DB_PATH = os.path.join(USER_DATA_DIR, "database", "interaction_logs.db")
SEMANTIC_DEDUPE_MODEL = CONFIG.get_model("model", CONFIG.get_str("runtime", "default_model", ""), section="semantic_dedupe")
SEMANTIC_DEDUPE_ENABLED = CONFIG.get_bool("semantic_dedupe", "enabled", True)
SEMANTIC_DEDUPE_CANDIDATE_LIMIT = CONFIG.get_int("semantic_dedupe", "candidate_limit", 8)
SEMANTIC_DEDUPE_DEFAULT_TTL_SECONDS = CONFIG.get_int("semantic_dedupe", "default_ttl_seconds", 604800)
SEMANTIC_DEDUPE_TTL_BY_KIND = {
    "todoist_reminder": CONFIG.get_int("semantic_dedupe", "todoist_reminder_ttl_seconds", 7 * 24 * 60 * 60),
    "internal_task": CONFIG.get_int("semantic_dedupe", "internal_task_ttl_seconds", 24 * 60 * 60),
    "reflection_task": CONFIG.get_int("semantic_dedupe", "reflection_task_ttl_seconds", 7 * 24 * 60 * 60),
    "do_now_action": CONFIG.get_int("semantic_dedupe", "do_now_action_ttl_seconds", 2 * 60 * 60),
    "calendar_event": CONFIG.get_int("semantic_dedupe", "calendar_event_ttl_seconds", 14 * 24 * 60 * 60),
}


def _build_semantic_dedupe_service() -> SemanticDeduplicationService:
    return SemanticDeduplicationService(
        memory=SQLiteMemoryAdapter(db_path=MEMORY_DB_PATH, memory_root=MEMORY_ROOT),
        llm_provider=LoggingLLMProvider(
            provider=LlamaCppAdapter(
                base_url=CONFIG.get_str("runtime", "api_base_url", "http://localhost:8080"),
                api_key=CONFIG.get_str("runtime", "api_key", "testkey"),
            ),
            log_store=SQLiteInteractionLogAdapter(db_path=INTERACTION_LOG_DB_PATH),
            current_response_path=None,
        ),
        enabled=SEMANTIC_DEDUPE_ENABLED,
        model=SEMANTIC_DEDUPE_MODEL,
        candidate_limit=SEMANTIC_DEDUPE_CANDIDATE_LIMIT,
        default_ttl_seconds=SEMANTIC_DEDUPE_DEFAULT_TTL_SECONDS,
        per_entity_ttl_seconds=SEMANTIC_DEDUPE_TTL_BY_KIND,
    )


def _evaluate_creation_candidate(*, entity_kind: str, source_kind: str, text: str, metadata: dict | None = None) -> dict:
    return asyncio.run(
        _build_semantic_dedupe_service().evaluate_candidate(
            entity_kind=entity_kind,
            source_kind=source_kind,
            text=text,
            metadata=metadata or {},
            model=SEMANTIC_DEDUPE_MODEL,
        )
    )


def _record_created_candidate(
    *,
    entity_kind: str,
    source_kind: str,
    text: str,
    metadata: dict | None = None,
    provider_ref: str | None = None,
) -> None:
    _build_semantic_dedupe_service().record_created(
        entity_kind=entity_kind,
        source_kind=source_kind,
        text=text,
        metadata=metadata or {},
        provider_ref=provider_ref,
    )


def _record_skipped_candidate(
    *,
    entity_kind: str,
    source_kind: str,
    text: str,
    duplicate_of_item_id: str | None,
    metadata: dict | None = None,
) -> None:
    _build_semantic_dedupe_service().record_skipped_duplicate(
        entity_kind=entity_kind,
        source_kind=source_kind,
        text=text,
        duplicate_of_item_id=duplicate_of_item_id,
        metadata=metadata or {},
    )


def read_model_details() -> list[dict[str, str]]:
    """Load model metadata from the project-local model registry."""
    details_path = os.path.join("D:\\Projects\\ambient_ai", "model_details.csv")
    models: list[dict[str, str]] = []
    if not os.path.exists(details_path):
        return models

    with open(details_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                name, description = line.split(":", 1)
                models.append({
                    "name": name.strip(),
                    "description": description.strip(),
                })
            else:
                models.append({
                    "name": line,
                    "description": "",
                })
    return models

def connect_facts_db():
    db_path = os.path.join("D:\\Projects\\ambient_ai\\database", "facts.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS facts (
            fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            fact TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    ''')
    conn.commit()
    return conn

def get_calendar_service():
    creds = None
    token_path = os.path.join("D:\\Projects\\ambient_ai", 'token.pickle')
    creds_path = os.path.join("D:\\Projects\\ambient_ai", 'credentials.json')  # put OAuth client secrets here

    if os.path.exists(token_path):
        with open(token_path, 'rb') as token_file:
            creds = pickle.load(token_file)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(creds_path):
                raise FileNotFoundError(f"credentials.json not found. Create OAuth client credentials and place credentials.json in the project root. current path: {creds_path}")
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'wb') as token_file:
            pickle.dump(creds, token_file)

    service = build('calendar', 'v3', credentials=creds)
    return service

@mcp.tool
def get_current_datetime():
    """
    Get the current date and time in ISO 8601 format.
    Can be used when current date/time is needed or when user types "now", "today","tomorrow" etc.
    Returns:
        A string representing the current date and time in ISO 8601 format.
    """
    return datetime.datetime.now().isoformat()

@mcp.tool
def add_task(content :Annotated[str, "The content of the reminder/to-do to be added"] ,
             due_datetime : Annotated[str, "Due date and time in YYYY-MM-DDTHH:MM:SS format"] ):
    """
    Add a reminder or to-do task to Todoist with an optional due date. 
    If due date is not given, add the due datetime of 5 hours from the current date and time   
    """
    api = TodoistAPI(TODOIST_API_TOKEN)
    try:
        parsed_due_date_time = datetime.datetime.fromisoformat(due_datetime)
        metadata = {"due_datetime": parsed_due_date_time.isoformat()}
        dedupe = _evaluate_creation_candidate(
            entity_kind="todoist_reminder",
            source_kind="mcp_add_task",
            text=content,
            metadata=metadata,
        )
        if dedupe["decision"] != "create_new":
            _record_skipped_candidate(
                entity_kind="todoist_reminder",
                source_kind="mcp_add_task",
                text=content,
                duplicate_of_item_id=dedupe.get("duplicate_of_item_id"),
                metadata={
                    **metadata,
                    "dedupe_reason": dedupe.get("reason"),
                    "dedupe_confidence": dedupe.get("confidence"),
                },
            )
            return {"Task Status": "Skipped Duplicate", "duplicate_of_item_id": dedupe.get("duplicate_of_item_id")}
        task = api.add_task(content = content, due_datetime=parsed_due_date_time)
        if task is not None:
            _record_created_candidate(
                entity_kind="todoist_reminder",
                source_kind="mcp_add_task",
                text=content,
                metadata=metadata,
                provider_ref=str(getattr(task, "id", "")).strip() or None,
            )
            return {"Task Status": "Success", "Task ID": task.id, "Content": task.content, "Due Date": task.due}
        else:
            return {"error": "Failed to add task"}
    except Exception as e:

        return {"error": f"Exception occurred: {str(e)}"}
    
@mcp.tool
def schedule_meeting(title: Annotated[str, "Title of the meeting"] = None,
                     date: Annotated[str, "Date in YYYY-MM-DD"] = None,
                     time: Annotated[str, "Time in HH:MM (24h). If omitted, uses current time + 5 minutes"] = None,
                     participants: Annotated[list, "List of attendee emails (strings)"] = None,
                     duration_minutes: Annotated[int, "Meeting length in minutes"] = 30):
    """
    Schedule a Google Calendar event with a Google Meet link.
    """
    if not title or not date:
        return {"error": "title and date are required"}

    # parse start and end datetimes in RFC3339 (assumes local timezone)
    try:
        # get local timezone info
        local_tz = datetime.datetime.now().astimezone().tzinfo

        if time:
            start_dt = datetime.datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
            # attach local timezone to the parsed naive datetime
            start_dt = start_dt.replace(tzinfo=local_tz)
        else:
            start_dt = datetime.datetime.now().astimezone()  # timezone-aware now

        end_dt = start_dt + datetime.timedelta(minutes=duration_minutes)

        # RFC3339 with offset (isoformat on a tz-aware dt)
        start_iso = start_dt.isoformat()
        end_iso = end_dt.isoformat()
    except Exception as e:
        return {"error": f"Invalid date/time format: {e}"}

    attendees = []
    if participants:
        # expect emails; if user supplied names, they should be resolved externally
        attendees = [{"email": p} for p in participants]

    event_body = {
        "summary": title,
        "start": {"dateTime": start_iso},
        "end": {"dateTime": end_iso},
        "attendees": attendees,
        "conferenceData": {
            "createRequest": {
                "requestId": str(uuid.uuid4()),
                "conferenceSolutionKey": {"type": "hangoutsMeet"}
            }
        }
    }

    try:
        metadata = {
            "schedule_info": {
                "date": date,
                "time": time,
                "duration_minutes": duration_minutes,
                "participants": participants or [],
                "start_iso": start_iso,
                "end_iso": end_iso,
            }
        }
        dedupe = _evaluate_creation_candidate(
            entity_kind="calendar_event",
            source_kind="mcp_schedule_meeting",
            text=title,
            metadata=metadata,
        )
        if dedupe["decision"] != "create_new":
            _record_skipped_candidate(
                entity_kind="calendar_event",
                source_kind="mcp_schedule_meeting",
                text=title,
                duplicate_of_item_id=dedupe.get("duplicate_of_item_id"),
                metadata={
                    **metadata,
                    "dedupe_reason": dedupe.get("reason"),
                    "dedupe_confidence": dedupe.get("confidence"),
                },
            )
            return {"status": "Skipped Duplicate", "duplicate_of_item_id": dedupe.get("duplicate_of_item_id")}
        service = get_calendar_service()
        event = service.events().insert(
            calendarId='primary',
            body=event_body,
            conferenceDataVersion=1,
            sendUpdates='all'  # set to 'none' if you don't want invites sent
        ).execute()

        meet_link = event.get('hangoutLink') or (event.get('conferenceData') or {}).get('entryPoints', [{}])[0].get('uri')
        _record_created_candidate(
            entity_kind="calendar_event",
            source_kind="mcp_schedule_meeting",
            text=title,
            metadata=metadata,
            provider_ref=str(event.get('id', '')).strip() or None,
        )
        return {"eventId": event.get('id'), "meetLink": meet_link}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def add(a: int, b: int) -> int:
    return a + b

@mcp.tool
def google_search(query: Annotated[str, "The google search query"], 
               num_results: Annotated[int, "Number of top results to return"] = 5) -> str:
    """
    Search the web using Google Search and return the top results.
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY
    }
    
    search = serpapi.search(params)
    
    organic_results = search['organic_results']
    return organic_results

@mcp.tool()
def queue_night_task(
    task_description: Annotated[str, "Detailed description of the task to be added to the night queue"],
    priority: Annotated[str, "Priority level of the task, 'high', 'medium', 'low'. Tasks that have to be done first get the highest priority"] = "medium"  
    ):
    """
    Add a task to the night queue for later processing.
    """
    try:
        metadata = {"priority": priority, "task_kind": "night_queue"}
        dedupe = _evaluate_creation_candidate(
            entity_kind="internal_task",
            source_kind="mcp_queue_night_task",
            text=task_description,
            metadata=metadata,
        )
        if dedupe["decision"] != "create_new":
            _record_skipped_candidate(
                entity_kind="internal_task",
                source_kind="mcp_queue_night_task",
                text=task_description,
                duplicate_of_item_id=dedupe.get("duplicate_of_item_id"),
                metadata={
                    **metadata,
                    "dedupe_reason": dedupe.get("reason"),
                    "dedupe_confidence": dedupe.get("confidence"),
                },
            )
            return f"Skipped duplicate night task: {task_description}"
        night_mode.add_task(task_description, priority)
        _record_created_candidate(
            entity_kind="internal_task",
            source_kind="mcp_queue_night_task",
            text=task_description,
            metadata=metadata,
        )
        return f"✅ Queued for tonight: {task_description}"
    except Exception as e:
        return f"❌ Database error: {e}"

@mcp.tool()
@run_async
def download_youtube_video(
    video_url: Annotated[str, "The URL of the YouTube video to download"],
    output_directory: Annotated[str, "The directory where the video will be saved. Video will be saved with its title as filename"] = "downloads"
    ):
    """
    Download a YouTube video using yt-dlp. The video is saved with its original title as the filename.
    """
    try:
        # Ensure output directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)
        
        # First, extract video info to get the title
        ydl_info_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_info_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            video_title = info.get('title', 'video')
            
            # Sanitize the title for use as filename (remove invalid characters)
            invalid_chars = '<>:"/\\|?*'
            for char in invalid_chars:
                video_title = video_title.replace(char, '')
            
            # Limit filename length to avoid filesystem issues
            if len(video_title) > 200:
                video_title = video_title[:200]
        
        # Construct the full output path with sanitized title
        output_path = os.path.join(output_directory, f"{video_title}.mp4")
        
        # Download the video
        ydl_opts = {
            'outtmpl': output_path,
            'format': 'bestvideo+bestaudio/best',
            'merge_output_format': 'mp4',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            
        completion_msg = f"BACKGROUND TASK COMPLETE: Downloaded '{video_title}' successfully to {output_path}"
        night_mode.add_notification(completion_msg)
        return f"✅ Downloaded successfully: {video_title}.mp4 → {output_directory}"
        
    except Exception as e:
        completion_msg = f"BACKGROUND TASK FAILED: Downloading {video_url} failed with error: {e}"
        night_mode.add_notification(completion_msg)
        return f"❌ Download error: {e}"

@mcp.tool()
def powershell_terminal(
    command: Annotated[str, "A PowerShell command to run"],
) -> str:
    """
    Run a PowerShell command. returns stdout/stderr.
    """
    blocked_tokens = [
        "remove-item",
        "del ",
        "rmdir ",
        "format ",
        "git",
    ]
    normalized = command.lower()
    if any(token in normalized for token in blocked_tokens):
        return "Blocked command."

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=r"D:\Projects\ambient_ai",
        )
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds."
    except Exception as e:
        return f"Terminal execution error: {e}"

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    combined_output = stdout
    if stderr:
        combined_output = f"{stdout}\n{stderr}".strip()

    if result.returncode != 0:
        return (
            f"Command failed with exit code {result.returncode}.\n"
            f"{combined_output or 'No output returned.'}"
        )
    return combined_output or "Command completed with no output."

@mcp.tool()
def list_available_models() -> dict:
    """
    List the available local models and their intended use cases.
    Call this before load_agent. Do not invent model names.
    """
    models = read_model_details()
    return {
        "models": models,
        "count": len(models),
    }

@mcp.tool
async def restore_previous_agent(message_to_agent: Annotated[str, "Message to the agent about the task and the  detailed summary of the task you performed"]):
    """
    Restore the LLM state from the most recently saved state.
    """
    # This tool is implemented directly in the LLMInteractionService to ensure the message history is properly handled during the model restoration.
    pass


@mcp.tool
async def use_browser(
    task: Annotated[
        str,
        "Detailed browser task including the goal, relevant URLs, stopping conditions, and expected result",
    ],
    headless: Annotated[
        bool,
        "Run in an isolated headless browser when true; use the visible persistent browser profile when false",
    ] = False,
) -> str:
    """
    Delegate one browser task to the dedicated browser-control model.

    The main model is saved and unloaded while the browser agent works. Raw
    browser tools are available only to that delegated agent. This tool is
    implemented by LLMInteractionService because it owns model handoff state.
    """
    pass
    
@mcp.tool
def load_agent(model_name: Annotated[str, "The name of the model to load"], message: Annotated[str, "Task to be performed by the model"]) -> str:
    """
    Load a different specialised model and perform a task with it.
    Call list_available_models first and use one of the exact returned model names.
    """
    # This tool is implemented directly in the LLMInteractionService to ensure the message history is properly handled during the model switch.
    pass
    
if __name__ == "__main__":
    mcp.disable("get_current_datetime", "add")
    mcp.run()

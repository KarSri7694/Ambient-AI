from fastmcp import FastMCP
from todoist_api_python.api import TodoistAPI
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

mcp = FastMCP("My MCP Server")

TODOIST_API_TOKEN = os.environ.get("TODOIST_API_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

SCOPES = ['https://www.googleapis.com/auth/calendar.events']

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

@mcp.tool(enabled=False)
def get_current_datetime():
    """
    Get the current date and time in ISO 8601 format.
    Can be used when current date/time is needed or when user types "now", "today","tomorrow" etc.
    Returns:
        A string representing the current date and time in ISO 8601 format.
    """
    return datetime.datetime.now().isoformat()

@mcp.tool
def add_task(content :Annotated[str, "The content of the task to be added"] ,
             due_datetime : Annotated[str, "Due date and time in YYYY-MM-DDTHH:MM:SS format"] ):
    """
    Add a task or reminder to Todoist with an optional due date.    
    """
    api = TodoistAPI(TODOIST_API_TOKEN)
    try:
        parsed_due_date_time = datetime.datetime.fromisoformat(due_datetime)
        task = api.add_task(content = content, due_datetime=parsed_due_date_time, due_lang="en")
        if task is not None:
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
        service = get_calendar_service()
        event = service.events().insert(
            calendarId='primary',
            body=event_body,
            conferenceDataVersion=1,
            sendUpdates='all'  # set to 'none' if you don't want invites sent
        ).execute()

        meet_link = event.get('hangoutLink') or (event.get('conferenceData') or {}).get('entryPoints', [{}])[0].get('uri')
        return {"eventId": event.get('id'), "meetLink": meet_link}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(enabled=False)
def add(a: int, b: int) -> int:
    return a + b

@mcp.tool(enabled=False)
def google_search(query: Annotated[str, "The google search query"], 
               num_results: Annotated[int, "Number of top results to return"] = 5) -> str:
    """
    Search the web using Google Search and return the top results.
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": ""
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
        night_mode.add_task(task_description, priority)
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

if __name__ == "__main__":
    mcp.run()

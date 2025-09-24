from fastmcp import FastMCP
from todoist_api_python.api import TodoistAPI
import os

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import pickle
import datetime
import uuid

#add_task() -- Done
#schedule_meeting() -- Done
#get_task()  
#edit_task()
#manage_person_profile()
#create_obsidian_note()
#log_expense()
#download movie()
#download music()
#download books()
#download tv_shows()

mcp = FastMCP("My MCP Server")

TODOIST_API_TOKEN = os.environ.get("TODOIST_API_TOKEN")

SCOPES = ['https://www.googleapis.com/auth/calendar.events']

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
def add_task(content :str = None, due_string : str = None ):
    """
    Add a task to Todoist with an optional due date.
    
    Args:
        content: (Required) The content of the task to be added.
        due_string: (Optional) A natural language description of the due date (e.g., 'tomorrow at 5pm').
    
    """
    api = TodoistAPI(TODOIST_API_TOKEN)
    try:
        task = api.add_task(content = content, due_string=due_string, due_lang="en")
        print(f"Task: {content} added successfully")
    except Exception as e:
        print(f"Error: {e}")

@mcp.tool
def schedule_meeting(title: str = None, date: str = None, time: str = None, participants: list = None, duration_minutes: int = 30):
    """
    Schedule a Google Calendar event with a Google Meet link.

    Args:
        title: (Required) Title of the meeting.
        date: (Required) Date in YYYY-MM-DD.
        time: (Required) Time in HH:MM (24h). If omitted, uses current time + 5 minutes.
        participants: (Optional) List of attendee emails (strings).
        duration_minutes: (Optional) Meeting length in minutes.

    Returns:
        Dictionary with event id and meet link on success, or error message.
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
    
@mcp.tool
def add(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    mcp.run()
import asyncio
import logging
from typing import Optional

from application.services.llm_interaction_service import LLMInteractionService

GUI_MCP_PATH = "GUI_mcp.json"
class GUIAgentService:
    """
    Does GUI automation tasks using pyautogui and other tools.
    """
    
    async def perform_task(task: str, llm_service : LLMInteractionService, screenshot_path: str):
        GUI_Prompt ="You are an agent that controls the computer using vision and tools. You will be provided with a screenshot of the current screen for every turn.Use the screenshot to understand the state of the system and use tools to achieve the user's goal. Explain your actions before making a tool call.Once clicked a text field do not click it again and again, assume it is selected and in next step start typing.ALWAYS USE open_app tool open apps, DO NOT USE START BUTTON OR TASKBAR ICONS TO OPEN APPS. "

        await llm_service.run_interaction(user_input=task, system_prompt=GUI_Prompt, image_path=screenshot_path)
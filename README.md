# 🌌 Ambient AI

**The Fully Local, Zero-Interaction Autonomous Agent.**

## 📖 Overview

**Ambient AI** is a fully local autonomous agent designed to 'ambient' around you. Unlike traditional assistants that wait for commands, Ambient AI actively listens to your conversations and watches your screen to understand your intent and context in real-time.

Its core philosophy is **Zero-Interaction**, but it goes beyond simple automation. Ambient AI acts as a proactive partner:

- **Contextual Research:** It catches topics you discuss in passing and autonomously performs deep web research or Google searches, presenting you with the answers before you even ask.
    
- **Intelligent Planning:** It generates daily to-do lists based on your previous days' context and historical conversations, ensuring nothing slips through the cracks.
    
- **Seamless Delegation:** While it works autonomously, you can also assign specific complex workflows by simply adding them to a dedicated **"Ambient AI Tasks"** project in Todoist. The agent picks these up and executes them during your computer's idle time.
---

## 🚀 Key Capabilities

### 👂 The "Ear" (Audio Intelligence)

- **Hinglish Transcription:** Uses a fine-tuned variant of the Whisper model optimized for code-mixed audio.
    
    - _Model:_ [Hindi2Hinglish (Oriserve/Whisper-Hindi2Hinglish-Apex)](https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Apex)
        
- **Speaker Diarization:** Real-time Voice Activity Detection (VAD) and speaker separation using **Pyannote**.
    
- **Voice Identity:** Secure voice embedding creation using the **SpeechBrain VoxCeleb** model to distinguish the user from guests.
    

### 🧠 The "Brain" (Reasoning & Control)

- **Local Inference:** Powered by **Qwen 3 VL 4B** as the default model for reasoning and tool use.
    
- **Notification System:** A state-aware feedback loop that updates the LLM on the status/outcome of previously performed tasks.
    
- **Chat Mode:** A direct interface to interact with the local LLM for general queries and conversation.
    

### 🛠️ The "Hands" (Tool Ecosystem)

Ambient AI features a custom **MCP (Model Context Protocol) Bridge** that allows unlimited extensibility.

- **Productivity:**
    
    - ✅ **Todoist:** Automatically extracts and adds tasks based on audio context.
        
    - 📅 **Google Meet:** Creates meetings automatically from conversation details.
        
    - 📓 **Obsidian:** Manages your personal knowledge base directly.
        
- **Research:**
    
    - 🌐 **Tavily:** Autonomous web search for deep research.
        
    - 🕸️ **Web Browsing:** Full page navigation and extraction.
        
- **System & Custom:**
    
    - 🔌 **Custom MCP Support:** Users can bridge as many MCP servers as needed (e.g., Filesystem, GitHub).
        
    - 📊 **Live Dashboard:** Visualizes agent activity and task completion history in real-time.
        
    - 🎙️ **FastAPI Server:** Endpoint to stream audio notes directly to the ASR model for transcription.
        

---

## 🗺️ Future Roadmap

We are actively developing the next generation of autonomous local agents.

- [ ] **Hexagonal Architecture Refactor:** Refactoring the entire codebase using the Ports and Adapters pattern to decouple core logic from external tools.
    
- [ ] **GUI Agent:** Developing a vision-based agent capable of direct screen control (clicking, typing, scrolling). _(Currently in Progress)_
    
- [ ] **AgentProg Implementation:** Implementing **Memory Pruning**, **Variable Stores**, and **Global Belief State** based on the [AgentProg Paper (Dec 2025)](https://arxiv.org/pdf/2512.10371v1). This will allow the agent to handle long-horizon tasks (50+ steps) without context overflow or hallucinations.
    
- [ ] **Context Fusion:** Merging Audio and Screenshot context streams to create a unified understanding of user intent.
    
- [ ] **Model Fine-tuning:** Fine-tuning Qwen 3 specifically for Ambient AI's unique autonomous task workflows.
    

---


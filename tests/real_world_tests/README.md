# Real-world testing lab

The real-world testing lab evaluates Ambient AI against prerecorded screenshots and audio. It runs the production vision, audio, autonomy, model, and MCP tool pipeline without capturing the live desktop or microphone.

The lab provides a local web UI where you can inspect:

- Every image or audio input released into the pipeline
- Image deduplication and visual observations
- Audio preprocessing, diarization, speaker identification, ASR, and merged transcripts
- Exact system, context, and user messages sent to each model
- Model responses, exposed reasoning, timing, and token metrics
- Tool definitions, tool calls, arguments, results, and errors
- The final result and manual 1–5 review scores

## Important safety warning

The lab uses the real MCP configuration and production capability policy. Tool calls can modify external services or local data. These changes are not automatically rolled back.

Before starting a run, the UI requires you to type:

```text
RUN LIVE TOOLS
```

Use test accounts or disable unwanted MCP servers in `mcp.json` when evaluating tool behavior.

## Prerequisites

Run all commands from the repository root.

1. Install the Python dependencies:

   ```powershell
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Ensure `config.ini` exists and contains usable model, audio, and runtime settings.

3. Live tool runs require active autonomy mode:

   ```ini
   [autonomy]
   mode = active
   ```

4. Ensure `models_preset.ini` contains every model referenced by the task-specific model roles in `config.ini`.

5. Start the configured llama.cpp model router. The normal project configuration uses:

   ```powershell
   llama-server --models-preset .\models_preset.ini --models-max 1 --no-models-autoload --host 127.0.0.1 --port 8080 --api-key testkey
   ```

6. Stop the normal Ambient AI process (`python src/app.py`). The lab and normal runtime intentionally cannot run at the same time because they share models, GPU resources, and MCP tools.

## Start the lab

From the repository root, run:

```powershell
python tests/real_world_tests/run_lab.py
```

Open the following page in a browser:

```text
http://127.0.0.1:8766/real-world-tests
```

Optional launcher arguments:

```powershell
python tests/real_world_tests/run_lab.py `
  --config .\config.ini `
  --data-root .\.ambient_data\real_world_tests `
  --host 127.0.0.1 `
  --port 8766
```

The host must remain `127.0.0.1`, `localhost`, or `::1`. The lab has no network authentication and refuses non-loopback hosts.

## Run uploaded media from the UI

This is the quickest way to create an ad-hoc test:

1. Open **Real-world Tests**.
2. Under **Or upload an ad-hoc sequence**, choose **Images** or **Audio**.
3. Select one or more files. Files are replayed in selection order, 10 seconds apart before playback-speed scaling.
4. Select a playback speed:
   - `1×` preserves the declared interval.
   - `2×`, `5×`, or `10×` accelerates the interval.
5. Review or override the task-specific model assignments.
6. Type `RUN LIVE TOOLS`.
7. Select **Start run**.
8. Select the run in **Run history** to watch its timeline and review its results.

Supported image formats are PNG, JPEG, and WebP. Supported audio formats are WAV, MP3, M4A, Opus, and FLAC.

## Run a reusable suite

Suite manifests live in `tests/real_world_tests/suites`. The included `example.json` contains image and audio examples, but its media files are intentionally not committed.

Place local files under:

```text
tests/real_world_tests/media/
```

For the example suite, the expected files are:

```text
tests/real_world_tests/media/desktop_01.png
tests/real_world_tests/media/desktop_02.png
tests/real_world_tests/media/conversation.wav
```

You can also use absolute local paths in a manifest.

Minimal image-sequence manifest:

```json
{
  "schema_version": 1,
  "suite_id": "desktop_workflow",
  "title": "Desktop workflow",
  "scenarios": [
    {
      "scenario_id": "task_progression",
      "title": "Task progression across screenshots",
      "modality": "image_sequence",
      "rubric_notes": "Check continuity, judgment, and tool selection.",
      "events": [
        {
          "offset_seconds": 0,
          "media_path": "../media/desktop_01.png",
          "screen_context": {
            "app_name": "Browser",
            "window_title": "Project page"
          }
        },
        {
          "offset_seconds": 10,
          "media_path": "../media/desktop_02.png"
        }
      ]
    }
  ]
}
```

Minimal audio-sequence scenario:

```json
{
  "scenario_id": "recorded_conversation",
  "title": "Recorded conversation",
  "modality": "audio_sequence",
  "events": [
    {
      "offset_seconds": 0,
      "media_path": "../media/conversation.wav"
    }
  ]
}
```

Event offsets must be non-negative and in ascending order. State is preserved between events within one scenario and reset before the next scenario.

## Model selection

The lab loads task-specific defaults from the `[models]` section of `config.ini`, including:

- `passive_observer_model`
- `full_passive_observer_model`
- `passive_followup_model`
- `followup_execution_model`
- `transcript_processing_model`
- `reporter_model`
- `browser_agent_model`

The UI can override a role with a model found in `models_preset.ini`. A run stores a snapshot of the resolved role-to-model mapping so previous results remain understandable after configuration changes.

## Results and stored data

Lab data is written under `.ambient_data/real_world_tests` by default:

```text
.ambient_data/real_world_tests/
├── media/                  # UI-uploaded media
├── runs/                   # Isolated scenario workspaces
└── real_world_tests.db     # Runs, results, trace events, and reviews
```

Each scenario receives fresh memory, autonomy, voice, capture, transcript, task, and artifact state. Uploaded media and completed results remain available after restarting the lab.

Use **Cancel safely** to request cancellation. Cancellation occurs between scheduled inputs, model iterations, or tool calls. An external tool call that is already executing is allowed to finish.

Stop the lab with `Ctrl+C` in its terminal.

## Run the automated lab tests

The automated tests use fake model and tool adapters. They do not execute live MCP tools:

```powershell
pytest tests/real_world_tests/test_real_world_lab.py -q
```

Run the complete project test suite with:

```powershell
pytest tests -q
```

Hardware/model tests should use the `real_world` pytest marker and are intentionally opt-in.

## Troubleshooting

### Ambient AI or the lab already owns the runtime

Stop `python src/app.py` or another `run_lab.py` process. Only one process may own the runtime lock.

### Model load or connection failure

Confirm the llama.cpp router is running at the `runtime.api_base_url` configured in `config.ini`, and verify the requested model exists in `models_preset.ini`.

### Real live-tool runs require autonomy mode active

Set `[autonomy] mode = active` in the selected configuration file and restart the lab.

### Missing scenario media

Correct the manifest path or upload the media through the UI. Relative paths are resolved from the suite JSON file.

### MCP server fails to start

Check `mcp.json`, required environment variables, credentials, and the command used by the failing MCP server. A failure is recorded in the lab timeline and runtime log.

### Audio pipeline fails

Verify the ASR model path, FFmpeg/audio dependencies, CUDA availability, Hugging Face token for gated diarization models, and the audio paths in the `[audio]` configuration section.

# Real-world testing lab

The real-world testing lab evaluates Ambient AI against prerecorded screenshots and audio. It runs the production perception, autonomy, semantic-memory, user-biodata, reflection, future-action, model, and MCP tool pipeline without capturing the live desktop or microphone.

The lab provides a local web UI where you can inspect:

- Every image or audio input released into the pipeline
- Image deduplication and visual observations
- Audio preprocessing, diarization, speaker identification, ASR, and merged transcripts
- Exact system, context, and user messages sent to each model
- Model responses, exposed reasoning, timing, and token metrics
- Tool definitions, tool calls, arguments, results, and errors
- Semantic indexing before and after proactive reasoning
- User biodata extracted into isolated `USER_INFO.md` and `MEMORY.md` files
- Reflection cleanup, generated tasks, semantic dedupe decisions, and skipped duplicates
- Bounded future actions executed through the normal tool-enabled interaction loop
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

4. To exercise semantic retrieval and reranking, enable `[semantic_memory]` and configure a working embedding model (and optionally a reranker). The loop still runs when semantic memory is unavailable, but its trace records the semantic stage as skipped.

5. Configure the finite live replay and its safety bounds:

   ```ini
   [real_world_tests]
   full_proactive_loop = true
   post_replay_idle_cycles = 3
   max_future_actions = 1
   semantic_sync_batches = 4
   ```

6. Ensure `models_preset.ini` contains every model referenced by the task-specific model roles in `config.ini`.

7. Start the configured llama.cpp model router. The normal project configuration uses:

   ```powershell
   llama-server --models-preset .\models_preset.ini --models-max 1 --no-models-autoload --host 127.0.0.1 --port 8080 --api-key testkey
   ```

8. Stop the normal Ambient AI process (`python src/app.py`). The lab and normal runtime intentionally cannot run at the same time because they share models, GPU resources, and MCP tools.

## Start the lab

From the repository root, run:

```powershell
python tests/real_world_tests/run_lab.py
```

Open the following page in a browser:

```text
http://127.0.0.1:8766/real-world-tests
```

Do not use the Real-world Tests tab on the normal runtime at port `8765` for
uploads. That process intentionally does not attach the isolated lab backend;
the page reports this state and keeps its upload control disabled.

When developing the React UI with Vite, point its API proxy at the lab:

```powershell
$env:VITE_API_PROXY_TARGET = "http://127.0.0.1:8766"
npm --prefix src/infrastructure/runtime_ui run dev
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
- `user_biodata_model`
- `followup_execution_model`
- `reflection_model`
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

For image scenarios, prerecorded files replace only the live screenshot-capture source. Each image is released in sequence and follows the same downstream route used by `app.py`:

1. Apply the production SSIM screenshot filter.
2. Copy the accepted image into the normal capture store without modifying the source file.
3. Enqueue a `lightweight_visual_capture` event with manifest-provided UI context.
4. Let `AutonomyCoordinatorService.process_batch()` perform vision enrichment, judgment, personalization, investigation, tools, artifacts, and inbox handling.
5. Trigger user-biodata processing at the configured live `biodata_update_event_interval`.

The lab does not call the vision observer directly and does not force reflection immediately after every image. Once the finite image stream ends, it advances up to `post_replay_idle_cycles` normal idle work units. These use `ReflectionService.run_if_due()`, semantic dedupe, and one queued task per idle cycle, matching live priority and cadence while keeping the test finite. Set `post_replay_idle_cycles = 0` to stop exactly when the final image's autonomy batch finishes.

Audio scenarios retain the explicit full proactive tail controlled by `full_proactive_loop`.

Audio transcripts are stored as semantic evidence and passed to the same production `UserBioDataService`, so audio-only scenarios can also populate working memory before reflection. The transcript is treated as untrusted contextual evidence, not as instructions. Empty or non-useful observations may still produce no biodata or future task; skipped stages remain visible in the trace and are never reported as completed.

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

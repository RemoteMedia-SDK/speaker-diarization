# speaker-diarization — Speaker diarization via pyannote-rs

Standalone Path 3 Rust cdylib that registers `SpeakerDiarizationNode` into
the [RemoteMedia SDK](https://github.com/RemoteMedia-SDK/remotemedia-sdk)
streaming pipeline registry.

Identifies "who spoke when" in audio streams using two ONNX models:

- **Segmentation model** (`segmentation-3.0.onnx`) — detects speech regions
- **Embedding model** (`wespeaker_en_voxceleb_CAM++.onnx`) — speaker fingerprints

Per-session state tracks speakers across chunks so the same person gets the
same speaker ID across the entire stream.

## Use from a manifest

```json
{
  "version": "v1",
  "plugins": ["speaker-diarization@v0.1.0"],
  "nodes": [
    {
      "id": "diarize",
      "node_type": "SpeakerDiarizationNode",
      "params": {
        "search_threshold": 0.5,
        "sample_rate": 16000,
        "passthrough_audio": true,
        "max_speakers": 10
      }
    }
  ]
}
```

The SDK resolver expands `speaker-diarization@v0.1.0` to
`github.com/RemoteMedia-SDK/speaker-diarization`, fetches `plugin.toml`,
then falls through to `release-manifest.json` for the platform-specific
prebuilt `.so` / `.dylib` / `.dll` asset.

## Build the cdylib locally

```bash
git clone https://github.com/RemoteMedia-SDK/speaker-diarization
cd speaker-diarization
cargo build --release
# → target/release/libspeaker_diarization_plugin.so
```

### Model files

At runtime the node looks for both ONNX files in the directory pointed to
by the `SPEAKER_DIARIZATION_MODELS_DIR` env var captured at build time
(defaults to `.`):

- `segmentation-3.0.onnx` — pyannote 3.0 segmentation
- `wespeaker_en_voxceleb_CAM++.onnx` — WeSpeaker CAM++ embeddings

Override the directory at build time:

```bash
SPEAKER_DIARIZATION_MODELS_DIR=/opt/models/pyannote cargo build --release
```

## What it exports

| Node type                  | Input            | Output                                                              |
|----------------------------|------------------|---------------------------------------------------------------------|
| `SpeakerDiarizationNode`   | `Audio` f32 PCM  | `Audio` (passthrough) with `metadata.diarization.{segments,...}`    |

Audio is auto-resampled to 16 kHz mono before diarization. When
`passthrough_audio` is `true` (default), the original audio is re-emitted
unchanged with a `diarization` metadata envelope:

```json
{
  "diarization": {
    "segments": [
      { "start": 0.12, "end": 1.84, "speaker": "0" },
      { "start": 1.91, "end": 3.40, "speaker": "1" }
    ],
    "num_speakers": 2,
    "time_offset": 0.0,
    "duration": 4.0
  }
}
```

## Config

| Field               | Default  | Description                                                       |
|---------------------|----------|-------------------------------------------------------------------|
| `search_threshold`  | `0.5`    | Cosine-similarity threshold for matching to a known speaker (0–1) |
| `sample_rate`       | `16000`  | Target sample rate (pyannote requires 16 kHz)                     |
| `passthrough_audio` | `true`   | Re-emit annotated audio (set false for metadata-only sinks)       |
| `max_speakers`      | `10`     | Soft cap; warns when exceeded (does not enforce)                  |

## Dependency notes

This plugin pulls `pyannote-rs` from the
[matbeedotcom/pyannote-rs `ort-rc12` fork](https://github.com/matbeedotcom/pyannote-rs/tree/ort-rc12)
rather than the upstream crates.io release. The fork is what the
RemoteMedia SDK host workspace itself used before this node was
extracted — it bumps `ndarray` 0.16 → 0.17 and aligns with
`ort = 2.0.0-rc.12` so pyannote's `eyre`-based error wrapping
compiles against `ort`'s `!Sync` operator types. Using the same fork
here guarantees bit-for-bit behavioural parity with the previous
in-host implementation. The standalone workspace ensures no
cross-tree unification interferes with the pin.

## License

See `LICENSE.md`. Governed by the RemoteMedia SDK Community License 1.0.

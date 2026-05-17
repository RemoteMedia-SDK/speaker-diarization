//! SpeakerDiarizationNode as a standalone Path 3 loadable plugin.
//!
//! Identifies "who spoke when" in audio streams using `pyannote-rs`.
//! Uses two ONNX models:
//!  - Segmentation model: detects when speech occurs
//!  - Embedding model: identifies which speaker is talking
//!
//! Originally lived in `remotemedia-core` under the
//! `speaker-diarization` feature; extracted here so the host crate
//! doesn't drag in `ort` (and the matbeedotcom pyannote-rs fork) just
//! for this single ML node.
//!
//! ## Node types exported
//!
//!   SpeakerDiarizationNode — Audio → Audio (passthrough + diarization
//!                                            metadata envelope)

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use remotemedia_plugin_sdk::abi_stable::sabi_trait::TD_Opaque;
use remotemedia_plugin_sdk::abi_stable::std_types::{ROk, RResult, RString};
use remotemedia_plugin_sdk::adapter::StreamingNodeFfiAdapter;
use remotemedia_plugin_sdk::traits::streaming::AsyncStreamingNode;
use remotemedia_plugin_sdk::types::{ControlMessageType, Error, RuntimeData};
use remotemedia_plugin_sdk::{FfiNodeBox, FfiNodeFactory, FfiNode_TO};

use pyannote_rs::{EmbeddingExtractor, EmbeddingManager};

/// Local result alias mirroring the original `crate::error::Result`.
type Result<T> = std::result::Result<T, Error>;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Speaker Diarization Node configuration
///
/// Configuration for the speaker diarization streaming node.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SpeakerDiarizationConfig {
    /// Search threshold for speaker matching (0.0-1.0)
    /// Lower values = more strict matching (fewer speakers detected)
    /// Higher values = looser matching (more speakers detected)
    pub search_threshold: f32,

    /// Expected sample rate (16000 Hz required by pyannote)
    #[serde(alias = "samplingRate")]
    pub sample_rate: u32,

    /// Whether to emit the original audio alongside diarization results
    #[serde(alias = "passthroughAudio")]
    pub passthrough_audio: bool,

    /// Maximum number of speakers to track (prevents unbounded memory growth)
    #[serde(alias = "maxSpeakers")]
    pub max_speakers: usize,
}

impl Default for SpeakerDiarizationConfig {
    fn default() -> Self {
        Self {
            search_threshold: 0.5,
            sample_rate: 16000,
            passthrough_audio: true,
            max_speakers: 10,
        }
    }
}

/// A single speaker segment with timing and speaker ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerSegment {
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
    /// Speaker identifier (e.g., "0", "1", "2")
    pub speaker: String,
}

/// Per-session state for speaker diarization.
struct DiarizationState {
    /// Embedding manager for speaker tracking
    embedding_manager: EmbeddingManager,
    /// Total samples processed (for timestamp calculation)
    total_samples: usize,
}

impl Default for DiarizationState {
    fn default() -> Self {
        Self {
            embedding_manager: EmbeddingManager::new(usize::MAX),
            total_samples: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/// Speaker Diarization Streaming Node.
pub struct SpeakerDiarizationNode {
    /// Search threshold for speaker matching
    search_threshold: f32,
    /// Expected sample rate
    sample_rate: u32,
    /// Whether to passthrough audio
    passthrough_audio: bool,
    /// Maximum number of speakers to track
    max_speakers: usize,
    /// Path to segmentation model
    segmentation_model_path: String,
    /// Path to embedding model
    embedding_model_path: String,
    /// Per-session diarization state
    states: Arc<Mutex<HashMap<String, DiarizationState>>>,
}

impl SpeakerDiarizationNode {
    /// Create a new SpeakerDiarizationNode with the given configuration.
    pub fn with_config(config: SpeakerDiarizationConfig) -> Self {
        // Get model paths from build-time env var (mirrors the host-side
        // node's behaviour).
        let models_dir = option_env!("SPEAKER_DIARIZATION_MODELS_DIR").unwrap_or(".");

        Self {
            search_threshold: config.search_threshold,
            sample_rate: config.sample_rate,
            passthrough_audio: config.passthrough_audio,
            max_speakers: config.max_speakers,
            segmentation_model_path: format!("{}/segmentation-3.0.onnx", models_dir),
            embedding_model_path: format!("{}/wespeaker_en_voxceleb_CAM++.onnx", models_dir),
            states: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create with default configuration.
    pub fn new() -> Self {
        Self::with_config(SpeakerDiarizationConfig::default())
    }

    /// Resample audio to target sample rate using linear interpolation.
    fn resample_audio(&self, audio: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
        if from_sr == to_sr {
            return audio.to_vec();
        }

        let ratio = from_sr as f32 / to_sr as f32;
        let new_len = (audio.len() as f32 / ratio) as usize;

        (0..new_len)
            .map(|i| {
                let pos = i as f32 * ratio;
                let idx = pos as usize;
                let frac = pos - idx as f32;

                if idx + 1 < audio.len() {
                    audio[idx] * (1.0 - frac) + audio[idx + 1] * frac
                } else {
                    audio[idx]
                }
            })
            .collect()
    }

    /// Convert stereo to mono by averaging channels.
    fn to_mono(&self, audio: &[f32], channels: u32) -> Vec<f32> {
        if channels <= 1 {
            return audio.to_vec();
        }

        audio
            .chunks(channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    }

    /// Convert f32 samples to i16 (pyannote-rs expects i16).
    fn f32_to_i16(&self, audio: &[f32]) -> Vec<i16> {
        audio
            .iter()
            .map(|&sample| {
                // Clamp to [-1.0, 1.0] and scale to i16 range
                let clamped = sample.clamp(-1.0, 1.0);
                (clamped * i16::MAX as f32) as i16
            })
            .collect()
    }

    /// Check if this node is stateful (it is — per-session embedding manager).
    pub fn is_stateful(&self) -> bool {
        true
    }

    /// Reset the diarization state for all sessions.
    pub fn reset_state(&mut self) {
        tokio::task::block_in_place(|| {
            let mut states = self.states.blocking_lock();
            states.clear();
        });
        tracing::info!("Speaker diarization states reset");
    }
}

impl Default for SpeakerDiarizationNode {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AsyncStreamingNode for SpeakerDiarizationNode {
    fn node_type(&self) -> &str {
        "SpeakerDiarizationNode"
    }

    async fn process(&self, _data: RuntimeData) -> Result<RuntimeData> {
        Err(Error::Execution(
            "SpeakerDiarizationNode requires streaming mode - use process_streaming() instead"
                .into(),
        ))
    }

    async fn process_streaming<F>(
        &self,
        data: RuntimeData,
        session_id: Option<String>,
        mut callback: F,
    ) -> Result<usize>
    where
        F: FnMut(RuntimeData) -> Result<()> + Send,
    {
        // Extract audio from RuntimeData - pass through non-audio data
        let (audio_samples, audio_sample_rate, audio_channels) = match &data {
            RuntimeData::Audio {
                samples,
                sample_rate,
                channels,
                ..
            } => (samples.clone(), *sample_rate, *channels),
            _ => {
                // Pass through non-audio data (e.g., video frames)
                callback(data)?;
                return Ok(1);
            }
        };

        // Resample to 16kHz if needed (pyannote requirement)
        let resampled = if audio_sample_rate != self.sample_rate {
            tracing::debug!(
                "Resampling from {}Hz to {}Hz",
                audio_sample_rate,
                self.sample_rate
            );
            self.resample_audio(&audio_samples, audio_sample_rate, self.sample_rate)
        } else {
            audio_samples.into_vec()
        };

        // Convert to mono if stereo
        let mono = self.to_mono(&resampled, audio_channels);

        // Get or create state for this session
        let session_key = session_id.clone().unwrap_or_else(|| "default".to_string());
        let mut states = self.states.lock().await;
        let state = states
            .entry(session_key.clone())
            .or_insert_with(DiarizationState::default);

        // Calculate time offset based on previously processed samples
        let time_offset = state.total_samples as f64 / self.sample_rate as f64;

        // Create embedding extractor (per-call, could optimize with caching)
        let mut embedding_extractor = EmbeddingExtractor::new(&self.embedding_model_path)
            .map_err(|e| Error::Execution(format!("Failed to load embedding model: {}", e)))?;

        // Convert f32 samples to i16 (pyannote-rs expects i16)
        let mono_i16 = self.f32_to_i16(&mono);

        // Get segments from audio
        let segments = pyannote_rs::get_segments(
            &mono_i16,
            self.sample_rate,
            &self.segmentation_model_path,
        )
        .map_err(|e| Error::Execution(format!("Failed to get segments: {}", e)))?;

        let mut speaker_segments: Vec<SpeakerSegment> = Vec::new();
        let mut num_speakers = 0;

        for segment_result in segments {
            match segment_result {
                Ok(segment) => {
                    // Compute embedding for this segment
                    let embedding: Vec<f32> = embedding_extractor
                        .compute(&segment.samples)
                        .map_err(|e| {
                            Error::Execution(format!("Failed to compute embedding: {}", e))
                        })?
                        .collect();

                    // Search for matching speaker or create new one
                    let speaker = state
                        .embedding_manager
                        .search_speaker(embedding.clone(), self.search_threshold)
                        .or_else(|| {
                            // No match found, try with threshold 0 to always assign a speaker
                            state.embedding_manager.search_speaker(embedding, 0.0)
                        })
                        .map(|id| id.to_string())
                        .unwrap_or_else(|| "?".to_string());

                    // Parse speaker ID to track max
                    if let Ok(id) = speaker.parse::<usize>() {
                        if id + 1 > num_speakers {
                            num_speakers = id + 1;
                        }
                    }

                    speaker_segments.push(SpeakerSegment {
                        start: time_offset + segment.start as f64,
                        end: time_offset + segment.end as f64,
                        speaker,
                    });

                    tracing::debug!(
                        "Segment: {:.2}s - {:.2}s, speaker: {}",
                        time_offset + segment.start as f64,
                        time_offset + segment.end as f64,
                        speaker_segments.last().unwrap().speaker
                    );
                }
                Err(e) => {
                    tracing::warn!("Failed to process segment: {:?}", e);
                }
            }
        }

        // Update total samples processed
        state.total_samples += mono.len();

        // Check speaker limit
        if num_speakers > self.max_speakers {
            tracing::warn!(
                "Number of speakers ({}) exceeds max_speakers ({})",
                num_speakers,
                self.max_speakers
            );
        }

        drop(states); // Release lock

        let mut outputs = 0;

        if self.passthrough_audio {
            // Emit single annotated audio with speaker segments in metadata
            if let RuntimeData::Audio {
                samples,
                sample_rate: orig_sr,
                channels: orig_ch,
                stream_id,
                timestamp_us,
                arrival_ts_us,
                ..
            } = data
            {
                callback(RuntimeData::Audio {
                    samples,
                    sample_rate: orig_sr,
                    channels: orig_ch,
                    stream_id,
                    timestamp_us,
                    arrival_ts_us,
                    metadata: Some(serde_json::json!({
                        "diarization": {
                            "segments": speaker_segments,
                            "num_speakers": num_speakers,
                            "time_offset": time_offset,
                            "duration": mono.len() as f64 / self.sample_rate as f64,
                        }
                    })),
                })?;
                outputs += 1;
            }
        }

        Ok(outputs)
    }

    async fn process_control_message(
        &self,
        message: RuntimeData,
        _session_id: Option<String>,
    ) -> Result<bool> {
        match message {
            RuntimeData::ControlMessage { message_type, .. } => match message_type {
                ControlMessageType::CancelSpeculation { .. } => {
                    // Diarization doesn't buffer speculatively
                    Ok(true)
                }
                _ => Ok(false),
            },
            _ => Ok(false),
        }
    }
}

// ---------------------------------------------------------------------------
// Factory + plugin registration
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct SpeakerDiarizationNodeFactory;

impl FfiNodeFactory for SpeakerDiarizationNodeFactory {
    fn node_type(&self) -> RString {
        RString::from("SpeakerDiarizationNode")
    }

    fn create(&self, params: RString) -> RResult<FfiNodeBox, RString> {
        let cfg: SpeakerDiarizationConfig =
            serde_json::from_str(params.as_str()).unwrap_or_default();
        ROk(FfiNode_TO::from_value(
            StreamingNodeFfiAdapter::new(SpeakerDiarizationNode::with_config(cfg)),
            TD_Opaque,
        ))
    }
}

remotemedia_plugin_sdk::plugin_export!(SpeakerDiarizationNodeFactory);

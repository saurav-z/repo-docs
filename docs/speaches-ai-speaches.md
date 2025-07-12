# Speaches: Your Open-Source AI Speech Toolkit (STT/TTS)

**Tired of relying on proprietary AI services for speech?** Speaches is an open-source, OpenAI API-compatible server that empowers you to easily transcribe, translate, and generate speech using powerful, customizable models.  [Explore the Speaches project on GitHub](https://github.com/speaches-ai/speaches) to get started!

_Note: This project was previously named `faster-whisper-server`._

## Key Features:

*   **OpenAI API Compatibility:** Seamlessly integrate with existing OpenAI tools and SDKs.
*   **Versatile Speech Generation:**
    *   Generate audio summaries from text.
    *   Perform sentiment analysis on audio recordings.
    *   Enable async speech-to-speech interactions.
*   **Streaming Transcription:** Receive real-time transcriptions via Server-Sent Events (SSE) without waiting for the entire audio to process.
*   **Dynamic Model Management:** Automatically load and unload models based on demand, optimizing resource usage.
*   **High-Quality Text-to-Speech (TTS):** Utilizes the top-rated Kokoro model and Piper for natural-sounding voice generation.
*   **GPU and CPU Support:** Leverage your hardware for optimal performance.
*   **Easy Deployment:** Deploy with Docker Compose or Docker.
*   **Realtime API for interactive audio applications**
*   **Highly Configurable:** Tailor the system to your specific needs.

## Technologies Used:

*   **Speech-to-Text (STT):** Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for accurate and efficient transcription.
*   **Text-to-Speech (TTS):** Utilizes [piper](https://github.com/rhasspy/piper) and the leading [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) model.

## Getting Started:

Refer to the official documentation for installation and usage instructions: [speaches.ai](https://speaches.ai/)

## Demos:

### Realtime API

(Insert video)

### Streaming Transcription

(TODO)

### Speech Generation

(Insert video)

## Contribute:

We welcome your contributions!  Please submit issues for bug reports, questions, or feature suggestions.
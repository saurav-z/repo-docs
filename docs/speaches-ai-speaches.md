# Speaches: Your Open-Source OpenAI-Compatible Server for Speech Processing

**Tired of vendor lock-in? Speaches provides a powerful, open-source alternative to OpenAI, offering streaming transcription, translation, and speech generation, all through a familiar API.** (See the original repo: [https://github.com/speaches-ai/speaches](https://github.com/speaches-ai/speaches))

Speaches is an OpenAI API-compatible server that empowers you to process speech and text seamlessly. It leverages the speed and accuracy of [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech-to-text, and utilizes high-quality Text-to-Speech (TTS) models like [piper](https://github.com/rhasspy/piper) and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M). This project aims to provide an accessible and flexible platform for all your audio processing needs.

## Key Features:

*   **OpenAI API Compatibility:** Works seamlessly with existing OpenAI tools and SDKs.
*   **Streaming Transcription:** Receive transcriptions in real-time via Server-Sent Events (SSE).
*   **Audio Generation:** Utilize the chat completions endpoint for:
    *   Generating audio summaries from text.
    *   Performing sentiment analysis on audio recordings.
    *   Enabling async speech-to-speech interactions.
*   **Text-to-Speech (TTS):** High-quality TTS using `kokoro` (ranked #1 in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) and `piper` models.
*   **Dynamic Model Loading/Offloading:** Automatically loads and unloads models based on your requests to optimize resource usage.
*   **GPU and CPU Support:** Optimized for both GPU and CPU hardware.
*   **Easy Deployment:** Deployable via Docker Compose and Docker.
*   **Realtime API:** For instant audio processing.
*   **Highly Configurable:** Customize the server to fit your specific needs.

## Demos

### Realtime API

**(Demo video embedded - link not provided as it requires a different solution)**

### Streaming Transcription

*(Demo video/example - To be added)*

### Speech Generation

**(Demo video embedded - link not provided as it requires a different solution)**

## Getting Started

For installation instructions and detailed usage guides, visit our documentation: [speaches.ai](https://speaches.ai/)

## Contribute

Encounter a bug, have a question, or a feature suggestion? Please create an issue on our GitHub repository.
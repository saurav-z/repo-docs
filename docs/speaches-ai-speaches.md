# Speaches: Your Open-Source AI Companion for Speech-to-Text, Translation, and Text-to-Speech

**Transform the way you interact with AI with Speaches, an OpenAI API-compatible server offering powerful and versatile speech processing capabilities.** Developed by [speaches-ai](https://github.com/speaches-ai/speaches), Speaches empowers you to easily integrate advanced speech technologies into your applications.

## Key Features:

*   **OpenAI API Compatibility:** Seamlessly integrates with existing OpenAI tools and SDKs for effortless adoption.
*   **Advanced Speech-to-Text (STT):**  Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for high-speed and accurate transcriptions.
*   **High-Quality Text-to-Speech (TTS):** Utilize cutting-edge TTS models including `kokoro` (ranked #1 in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) and `piper`.
*   **Versatile Functionality:**
    *   Audio Generation (via the chat completions endpoint):  Generate spoken audio summaries from text, perform sentiment analysis on audio, and enable async speech-to-speech interactions.
    *   Realtime API support with [realtime API](https://speaches.ai/usage/realtime-api)
*   **Streaming Transcription:** Receive transcriptions in real-time via Server-Sent Events (SSE) for a more responsive experience.
*   **Dynamic Model Management:** Automatic loading and unloading of models based on your requests, optimizing resource utilization.
*   **GPU and CPU Support:**  Leverages the power of your hardware for optimal performance.
*   **Easy Deployment:** Deployable via Docker Compose and Docker, with detailed installation instructions available at [speaches.ai/installation/](https://speaches.ai/installation/).
*   **Highly Configurable:** Tailor Speaches to your specific needs with extensive configuration options, documented at [speaches.ai/configuration/](https://speaches.ai/configuration/).

## Demos

### Realtime API

(Example Demo showing Realtime API, from original README)

### Streaming Transcription

(TODO: Add description of what this demo shows, and then insert a relevant link or description here.)

### Speech Generation

(Example Demo, from original README)

## Get Started

Explore the [speaches.ai](https://speaches.ai/) documentation for detailed installation guides, usage examples, and configuration options.

## Contribute

Find a bug, have a question, or a feature suggestion? Please create an issue on the [GitHub repository](https://github.com/speaches-ai/speaches).
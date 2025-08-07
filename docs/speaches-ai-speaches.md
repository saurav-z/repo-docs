# Speaches: Your Open-Source AI Companion for Speech & Audio

**Looking for a powerful, open-source alternative to OpenAI for speech-to-text, text-to-speech, and audio generation?** Speaches offers an OpenAI API-compatible server, empowering you to easily integrate cutting-edge AI audio capabilities into your projects. [(View the original project on GitHub)](https://github.com/speaches-ai/speaches)

## Key Features:

*   **OpenAI API Compatibility:** Seamlessly integrates with all tools and SDKs designed for OpenAI's API.
*   **Versatile Audio Generation:** Leverage the chat completions endpoint for a variety of use cases:
    *   Generate spoken audio summaries from text.
    *   Perform sentiment analysis on recordings.
    *   Enable asynchronous speech-to-speech interactions with AI models.
*   **Real-time Streaming:** Receive transcription results via Server-Sent Events (SSE) as the audio is processed, eliminating the need to wait for complete transcription.
*   **Dynamic Model Management:** Automatically load and unload models based on your requests, optimizing resource usage.
*   **High-Quality Text-to-Speech (TTS):** Utilizes the top-performing [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) model (as ranked in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) along with `piper` models for exceptional speech synthesis.
*   **GPU & CPU Support:** Benefit from optimized performance regardless of your hardware setup.
*   **Easy Deployment:** Deploy with Docker Compose or Docker.
*   **Real-time API:** Access real-time audio processing capabilities.
*   **Highly Configurable:** Tailor Speaches to your specific needs with extensive configuration options.

## Demos

### Realtime API

![Realtime API Demo](https://github.com/user-attachments/assets/457a736d-4c29-4b43-984b-05cc4d9995bc)

### Streaming Transcription

TODO

### Speech Generation

![Speech Generation Demo](https://github.com/user-attachments/assets/0021acd9-f480-4bc3-904d-831f54c4d45b)

## Get Started

For installation instructions, usage guides, and detailed configuration options, visit the official documentation: [speaches.ai](https://speaches.ai/)

## Contribute

Encounter a bug, have a question, or a feature suggestion? Please open an issue on GitHub!
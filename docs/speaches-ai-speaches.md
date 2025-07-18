# Speaches: Your Open-Source AI Companion for Speech & Audio

**Transform audio and text with ease using Speaches, an OpenAI API-compatible server for streaming transcription, translation, and speech generation.**  This project is your go-to solution for integrating powerful speech and audio functionalities into your applications. (Originally named `faster-whisper-server`, it has evolved to support much more!)

[Visit the original repository on GitHub](https://github.com/speaches-ai/speaches)

## Key Features of Speaches

Speaches provides a flexible and robust platform for all your audio and speech needs:

*   **OpenAI API Compatibility:** Seamlessly integrate with existing OpenAI tools and SDKs.
*   **Versatile Audio Generation:** Leverage the chat completions endpoint for:
    *   Generating spoken audio summaries from text.
    *   Performing sentiment analysis on audio recordings.
    *   Enabling async speech-to-speech interactions.
*   **Real-time Streaming:** Receive transcription results via Server-Sent Events (SSE) as audio is processed, eliminating the need to wait for complete transcription.
*   **Dynamic Model Management:** Automatically load and unload models based on your request, optimizing resource usage.
*   **Text-to-Speech (TTS) Capabilities:** Utilize state-of-the-art TTS models like `kokoro` (ranked #1 in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) and `piper`.
*   **GPU and CPU Support:** Benefit from hardware acceleration for faster processing.
*   **Docker Deployment:** Easily deploy with Docker Compose / Docker ([Installation Guide](https://speaches.ai/installation/)).
*   **Realtime API:** Access a powerful Realtime API ([Realtime API Documentation](https://speaches.ai/usage/realtime-api)).
*   **Highly Configurable:** Customize the platform to meet your specific needs ([Configuration Options](https://speaches.ai/configuration/)).

## Demos

See Speaches in action!

### Realtime API

(Insert embedded demo video here)

### Streaming Transcription

(Placeholder for a future demo)

### Speech Generation

(Insert embedded demo video here)

## Get Started

Explore the documentation for detailed installation and usage instructions: [speaches.ai](https://speaches.ai/)

## Contribute

Encountered a bug, have a question, or suggest a feature?  Please create an issue to help improve Speaches!
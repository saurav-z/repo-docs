# Speaches: Your Open-Source AI Toolkit for Speech and Audio

**Tired of proprietary APIs?** Speaches provides an open-source, API-compatible solution for streaming transcription, translation, and speech generation, giving you complete control over your audio processing. ([View the original repository](https://github.com/speaches-ai/speaches))

## Key Features:

*   **OpenAI API Compatibility:** Seamlessly integrate with existing tools and SDKs designed for the OpenAI API.
*   **Versatile Audio Generation:** Utilize the chat completions endpoint for:
    *   Text-to-speech: Convert text into spoken audio summaries.
    *   Sentiment analysis: Analyze audio recordings to extract sentiment.
    *   Async speech-to-speech interactions.
*   **Streaming Support:** Receive real-time transcription via Server-Sent Events (SSE) as the audio is processed, without waiting for completion.
*   **Dynamic Model Management:** Automatically load and unload models based on your requests, optimizing resource usage.
*   **High-Quality Text-to-Speech:** Powered by top-rated models like `kokoro` (Ranked #1 in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) and `piper`.
*   **GPU & CPU Support:** Leverage the power of your hardware for optimal performance.
*   **Easy Deployment:** Deploy with Docker Compose or Docker.
*   **Real-Time API:** Interact with the API in real-time.
*   **Highly Configurable:** Customize Speaches to meet your specific needs.

## Technologies powering Speaches

*   **Speech-to-Text:**  [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
*   **Text-to-Speech:** [piper](https://github.com/rhasspy/piper) and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)

## Demos

### Realtime API

[![Realtime API Demo](https://github.com/user-attachments/assets/457a736d-4c29-4b43-984b-05cc4d9995bc)](https://github.com/user-attachments/assets/457a736d-4c29-4b43-984b-05cc4d9995bc)

### Streaming Transcription

*   **TODO**

### Speech Generation

[![Speech Generation Demo](https://github.com/user-attachments/assets/0021acd9-f480-4bc3-904d-831f54c4d45b)](https://github.com/user-attachments/assets/0021acd9-f480-4bc3-904d-831f54c4d45b)

## Get Started

*   **Installation and Usage:** [speaches.ai](https://speaches.ai/)
*   **Realtime API Documentation:** [speaches.ai/usage/realtime-api](https://speaches.ai/usage/realtime-api)
*   **Configuration Options:** [speaches.ai/configuration/](https://speaches.ai/configuration/)

## Contribute

Encounter a bug, have a question, or a feature suggestion? Please create an issue.
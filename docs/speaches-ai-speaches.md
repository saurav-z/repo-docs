# Speaches: Your Open-Source AI Toolkit for Speech-to-Text, Translation, and Speech Generation

**Transform audio and text seamlessly with Speaches, a powerful and versatile open-source alternative to OpenAI's API.** ([View on GitHub](https://github.com/speaches-ai/speaches))

## Key Features:

*   **OpenAI API Compatibility:**  Works seamlessly with all tools and SDKs compatible with the OpenAI API, offering a drop-in replacement experience.
*   **Advanced Audio Generation:** Utilize the chat completions endpoint for:
    *   Generating spoken audio summaries from text.
    *   Performing sentiment analysis on audio recordings.
    *   Enabling asynchronous speech-to-speech interactions.
*   **Real-Time Streaming:** Receive transcriptions and translations via Server-Sent Events (SSE) as the audio is processed, eliminating the need to wait for the entire file.
*   **Dynamic Model Management:**  Automatically loads and unloads models based on your requests, optimizing resource usage.
*   **High-Quality Text-to-Speech (TTS):** Leverages top-performing models including `kokoro` (ranked #1 in the TTS Arena) and `piper` for natural-sounding audio output.
*   **GPU & CPU Support:** Optimized performance across both GPU and CPU hardware.
*   **Docker Deployment:** Easily deploy Speaches using Docker Compose or Docker for simplified setup and scaling.
*   **Realtime API:** Access a dedicated [Realtime API](https://speaches.ai/usage/realtime-api) for interactive voice applications.
*   **Highly Configurable:**  Tailor Speaches to your specific needs with extensive [configuration options](https://speaches.ai/configuration/).

## Core Technologies:

*   **Speech-to-Text (STT):** Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for accurate and efficient transcriptions.
*   **Text-to-Speech (TTS):** Utilizes [piper](https://github.com/rhasspy/piper) and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) models for realistic voice generation.

## Demos

### Realtime API

*(Demo video coming soon)*

### Streaming Transcription

*(Demo video coming soon)*

### Speech Generation

*(Demo video coming soon)*

## Get Started:

*   **Installation:**  Refer to the detailed installation instructions on the official documentation: [speaches.ai](https://speaches.ai/)
*   **Support:**  Report bugs, ask questions, or suggest features by opening an issue.
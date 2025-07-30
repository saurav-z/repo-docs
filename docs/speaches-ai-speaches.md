# Speaches: Your Open-Source AI Companion for Speech & Language

**Speaches** is an open-source, OpenAI API-compatible server that brings the power of cutting-edge speech and language models to your fingertips.  **(Check out the original repo for more details: [speaches-ai/speaches](https://github.com/speaches-ai/speaches))**

This project, formerly known as `faster-whisper-server`, has evolved to provide a comprehensive suite of features, including speech-to-text (STT), translation, and text-to-speech (TTS) capabilities. Built on the foundation of robust open-source technologies like [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [piper](https://github.com/rhasspy/piper), and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M), Speaches aims to be the Ollama equivalent for audio-based models.

Explore the possibilities with our detailed documentation: [speaches.ai](https://speaches.ai/)

## Key Features of Speaches:

*   **OpenAI API Compatibility:** Seamlessly integrate with existing OpenAI tools and SDKs.
*   **Versatile Audio Generation (Chat Completions Endpoint):**
    *   Generate spoken audio summaries from text.
    *   Perform sentiment analysis on audio recordings.
    *   Enable asynchronous speech-to-speech interactions.
*   **Real-time Streaming Support:** Receive transcriptions via Server-Sent Events (SSE) as the audio is processed, without waiting for completion.
*   **Dynamic Model Management:** Automatically load and unload models based on your requests, optimizing resource utilization.
*   **High-Quality Text-to-Speech (TTS):**  Utilize both `kokoro` (Top-ranked in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) and `piper` models for diverse voice options.
*   **GPU and CPU Support:**  Leverage your hardware for optimal performance.
*   **Easy Deployment:** Deploy with Docker Compose or Docker ([speaches.ai/installation/](https://speaches.ai/installation/)).
*   **Realtime API:** Interact with real-time audio processing capabilities ([speaches.ai/usage/realtime-api](https://speaches.ai/usage/realtime-api)).
*   **Highly Configurable:** Customize Speaches to meet your specific needs ([speaches.ai/configuration/](https://speaches.ai/configuration/)).

## Demos

### Realtime API

[Demo Video](https://github.com/user-attachments/assets/457a736d-4c29-4b43-984b-05cc4d9995bc)

### Streaming Transcription

*TODO: Add Streaming Transcription demo*

### Speech Generation

[Demo Video](https://github.com/user-attachments/assets/0021acd9-f480-4bc3-904d-831f54c4d45b)

## Contributing

If you encounter any bugs, have feature suggestions, or simply want to ask a question, please create an issue on the [GitHub repository](https://github.com/speaches-ai/speaches).
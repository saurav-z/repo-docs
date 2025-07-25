# Speaches: Your Open-Source AI Voice and Speech Solution 

**Transform text to speech, transcribe audio, and translate in real-time with Speaches, a powerful and versatile OpenAI API-compatible server.**  ([View the original project on GitHub](https://github.com/speaches-ai/speaches))

> **Note:** This project was formerly known as `faster-whisper-server` and has evolved to support a wider range of features.

Speaches provides a flexible and efficient way to interact with cutting-edge AI models for speech-related tasks.  Leveraging the power of [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for Speech-to-Text, and [piper](https://github.com/rhasspy/piper) and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) for Text-to-Speech, Speaches offers a comprehensive solution for audio processing.  Think of it as Ollama, but tailored for TTS/STT models.

[Explore the complete documentation for detailed installation and usage instructions at speaches.ai](https://speaches.ai/)

## Key Features:

*   **OpenAI API Compatibility:** Seamlessly integrate with existing OpenAI tools and SDKs.
*   **Audio Generation (Text-to-Speech):**  Convert text to spoken audio using the chat completions endpoint.  Examples:
    *   Create audio summaries of text.
    *   Perform sentiment analysis on audio recordings.
    *   Enable asynchronous speech-to-speech interactions.
*   **Streaming Support:** Receive transcription results in real-time via Server-Sent Events (SSE) as the audio is processed, eliminating the need to wait for the entire file.
*   **Dynamic Model Loading/Offloading:**  Automatically load and unload models based on your request, optimizing resource usage. Just specify which model you want to use in the request and it will be loaded automatically. It will then be unloaded after a period of inactivity.
*   **Text-to-Speech (TTS) with High-Quality Models:** Utilize `kokoro` (ranked #1 in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) and `piper` for natural-sounding speech generation.
*   **GPU and CPU Support:** Leverage your hardware's capabilities for optimal performance.
*   **Docker and Docker Compose Deployment:** Easily deploy Speaches using Docker Compose or Docker for containerized environments ([Deployment Instructions](https://speaches.ai/installation/)).
*   **Realtime API:** Experience real-time audio processing capabilities ([Realtime API Documentation](https://speaches.ai/usage/realtime-api)).
*   **Highly Configurable:** Customize Speaches to meet your specific needs ([Configuration Options](https://speaches.ai/configuration/)).

## Demos

### Realtime API

[Demo video showcasing the Realtime API](https://github.com/user-attachments/assets/457a736d-4c29-4b43-984b-05cc4d9995bc)

### Streaming Transcription

TODO

### Speech Generation

[Demo video showcasing Speech Generation capabilities](https://github.com/user-attachments/assets/0021acd9-f480-4bc3-904d-831f54c4d45b)

---

We encourage you to report any bugs, ask questions, or suggest new features by [creating an issue](https://github.com/speaches-ai/speaches/issues).
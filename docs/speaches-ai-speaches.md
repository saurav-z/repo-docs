# Speaches: Your OpenAI-Compatible Server for Speech-to-Text, Translation, and Text-to-Speech

**Transform audio and text effortlessly with Speaches, an open-source server designed to emulate the OpenAI API for cutting-edge speech and language processing.**  *(For the original project, see: [Speaches on GitHub](https://github.com/speaches-ai/speaches))*

This project, formerly known as `faster-whisper-server`, has evolved to offer a comprehensive suite of speech and language capabilities, making it a versatile tool for developers and researchers alike. It leverages the power of [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for Speech-to-Text (STT) and utilizes [piper](https://github.com/rhasspy/piper) and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) for Text-to-Speech (TTS) functionalities. Speaches aims to be the Ollama of TTS/STT models.

Explore the detailed documentation for installation and usage at [speaches.ai](https://speaches.ai/).

## Key Features:

*   **OpenAI API Compatibility:**  Seamlessly integrate with all tools and SDKs designed for the OpenAI API.
*   **Audio Generation (Chat Completions Endpoint):**
    *   Generate spoken audio summaries from text.
    *   Perform sentiment analysis on audio recordings.
    *   Enable async speech-to-speech interactions.
*   **Streaming Support:** Receive transcriptions in real-time via Server-Sent Events (SSE) as the audio is processed.
*   **Dynamic Model Loading/Offloading:**  Automatically load and unload models based on request, optimizing resource utilization.
*   **Text-to-Speech (TTS):** Utilizes high-quality `kokoro` (ranked #1 in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) and `piper` models.
*   **GPU and CPU Support:** Optimized for both GPU and CPU environments.
*   **Docker Deployment:** Easy deployment using Docker Compose or Docker.
*   **Realtime API:** Access a robust realtime API.
*   **Highly Configurable:** Customize settings to meet your specific needs.

## Demos

### Realtime API

*(Demo Video)*

### Streaming Transcription

*Coming Soon*

### Speech Generation

*(Demo Video)*

**Get Involved:**
Please create an issue if you find a bug, have a question, or a feature suggestion.
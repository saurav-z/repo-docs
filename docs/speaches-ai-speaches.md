# Speaches: Your Open-Source AI Companion for Speech-to-Text, Translation, and Text-to-Speech

**Speaches** empowers you to easily transcribe audio, translate languages, and generate realistic speech using open-source models, all within an OpenAI API-compatible framework.  *(Link back to original repo: [speaches-ai/speaches](https://github.com/speaches-ai/speaches))*

This project, formerly known as `faster-whisper-server`, has evolved to offer a comprehensive suite of AI audio tools. It leverages the power of [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for Speech-to-Text, and [piper](https://github.com/rhasspy/piper) and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) for Text-to-Speech, providing an open-source alternative to tools like Ollama, but specifically focused on audio processing.

For detailed installation instructions and usage, please refer to the comprehensive documentation at: [speaches.ai](https://speaches.ai/)

## Key Features

*   **OpenAI API Compatibility:** Seamlessly integrates with all OpenAI tools and SDKs, making it easy to use with your existing workflows.
*   **Versatile Audio Generation:** Utilize the chat completions endpoint for a range of applications:
    *   Convert text to spoken audio summaries (text-in, audio-out).
    *   Analyze sentiment from audio recordings (audio-in, text-out).
    *   Enable async speech-to-speech interactions with AI models (audio-in, audio-out).
*   **Real-time Streaming:** Receive transcriptions via Server-Sent Events (SSE) as the audio is processed, eliminating the need to wait for complete transcription.
*   **Dynamic Model Management:** Automatically load and unload models based on your requests, optimizing resource usage.
*   **High-Quality Text-to-Speech:** Utilizes the top-ranked [Kokoro](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena) and `piper` models for natural-sounding speech generation.
*   **GPU and CPU Support:**  Offers flexibility in hardware utilization for optimal performance.
*   **Easy Deployment:** Deployable via Docker Compose and Docker.
*   **Realtime API:**  Provides a dedicated [realtime API](https://speaches.ai/usage/realtime-api).
*   **Highly Configurable:** Customize the application to meet your specific needs with extensive [configuration options](https://speaches.ai/configuration/).

## Demos

### Realtime API

![Realtime API Demo](https://github.com/user-attachments/assets/457a736d-4c29-4b43-984b-05cc4d9995bc)

### Streaming Transcription

(Coming Soon)

### Speech Generation

![Speech Generation Demo](https://github.com/user-attachments/assets/0021acd9-f480-4bc3-904d-831f54c4d5b)

## Get Involved

Please feel free to create an issue if you find a bug, have a question, or a feature suggestion!
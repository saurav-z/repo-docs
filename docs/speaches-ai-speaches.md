# Speaches: Your Open-Source AI Toolkit for Speech & Audio

**Speaches empowers you with powerful, open-source AI capabilities for speech-to-text, text-to-speech, and real-time audio applications, all through an OpenAI-compatible API.**  ([View the original repo on GitHub](https://github.com/speaches-ai/speaches))

Note: This project was previously known as `faster-whisper-server`. It has evolved to support a broader range of features beyond just ASR.

## Key Features:

*   **OpenAI API Compatibility:** Seamlessly integrate with existing OpenAI tools, SDKs, and applications.
*   **Audio Generation:**
    *   Generate spoken audio summaries from text.
    *   Perform sentiment analysis on audio recordings.
    *   Enable asynchronous speech-to-speech interactions.
*   **Streaming Support:** Receive real-time transcription via Server-Sent Events (SSE) as audio is processed.
*   **Dynamic Model Management:** Automatically load and unload models based on request, optimizing resource usage.
*   **Text-to-Speech:** Utilize `kokoro` (ranked #1 in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) and `piper` models for high-quality audio generation.
*   **Speech-to-Text:** Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for accurate and efficient transcription.
*   **GPU and CPU Support:** Leverage the power of your hardware for optimal performance.
*   **Easy Deployment:** Deployable using Docker Compose and Docker.
*   **Realtime API:** Access a dedicated realtime API for interactive audio applications.
*   **Highly Configurable:** Customize settings to meet your specific needs.

## Demos

### Realtime API

![Realtime API Demo](https://github.com/user-attachments/assets/457a736d-4c29-4b43-984b-05cc4d9995bc)

### Streaming Transcription

(Coming Soon)

### Speech Generation

![Speech Generation Demo](https://github.com/user-attachments/assets/0021acd9-f480-4bc3-904d-831f54c4d45b)

## Resources

*   **Installation and Usage:** [speaches.ai](https://speaches.ai/)
*   **Configuration:** [speaches.ai/configuration/](https://speaches.ai/configuration/)

## Get Involved

Please report bugs, ask questions, or suggest new features by [creating an issue](https://github.com/speaches-ai/speaches/issues).
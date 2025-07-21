# Speaches: Your Open-Source AI Companion for Speech & Audio Processing

Tired of proprietary AI services? **Speaches** provides a powerful, open-source alternative for transcription, translation, and speech generation, compatible with the OpenAI API. [Explore the Speaches repository on GitHub](https://github.com/speaches-ai/speaches)

## Key Features:

*   **OpenAI API Compatibility:** Seamlessly integrate with existing tools and SDKs designed for the OpenAI API.
*   **Versatile Audio Generation (Text-to-Speech):**
    *   Convert text to spoken audio summaries.
    *   Analyze audio recordings for sentiment.
    *   Enable asynchronous speech-to-speech interactions.
*   **Real-time Streaming:** Receive streaming transcriptions via Server-Sent Events (SSE) as the audio is processed, without waiting for completion.
*   **Dynamic Model Management:** Automatically load and unload models based on your requests, optimizing resource usage.
*   **High-Quality Text-to-Speech:** Utilize the top-ranked `kokoro` model (based on the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) and `piper` for realistic voice generation.
*   **GPU and CPU Support:** Leverage the power of your hardware for optimal performance.
*   **Docker Deployment:** Easily deployable with Docker Compose and Docker.
*   **Realtime API:**  Interact with the API for dynamic applications.
*   **Highly Configurable:** Customize Speaches to meet your specific needs.

## Get Started:

*   **Installation and Usage:** Find detailed instructions on [speaches.ai](https://speaches.ai/)
*   **Realtime API Demo:** [Video Demo](https://github.com/user-attachments/assets/457a736d-4c29-4b43-984b-05cc4d9995bc)
*   **Speech Generation Demo:** [Video Demo](https://github.com/user-attachments/assets/0021acd9-f480-4bc3-904d-831f54c4d45b)
*   **Streaming Transcription Demo:** (Coming Soon!)

## Powered By:

*   **Speech-to-Text:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
*   **Text-to-Speech:** `piper` and `kokoro` ([Hugging Face](https://huggingface.co/hexgrad/Kokoro-82M))

## Contribute and Connect:

Report bugs, ask questions, or suggest new features by [creating an issue on GitHub](https://github.com/speaches-ai/speaches/issues).
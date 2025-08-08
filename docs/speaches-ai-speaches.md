# Speaches: Your Open-Source AI Speech & Transcription Toolkit

**Unlock the power of AI-powered speech with Speaches, an OpenAI API-compatible server for streaming transcription, translation, and speech generation.**

[View the original repository on GitHub](https://github.com/speaches-ai/speaches)

**Key Features:**

*   **OpenAI API Compatibility:** Seamlessly integrates with all tools and SDKs designed for the OpenAI API, making it easy to adopt.
*   **Versatile Audio Generation:**
    *   Generate spoken audio summaries from text inputs (text-to-speech).
    *   Perform sentiment analysis on audio recordings (speech-to-text).
    *   Enable async speech-to-speech interactions with AI models.
*   **Streaming Transcription:** Receive transcriptions in real-time via Server-Sent Events (SSE) as audio is processed, enhancing user experience.
*   **Dynamic Model Management:** Automatically loads and unloads models based on your request, optimizing resource utilization.
*   **Text-to-Speech (TTS) Capabilities:** Utilizes cutting-edge TTS models, including `kokoro` (Top-ranked in TTS Arena) and `piper`, delivering high-quality voice synthesis.
*   **GPU & CPU Support:** Flexible hardware support for optimal performance.
*   **Deployment Options:** Easily deployable using Docker Compose and Docker.
*   **Realtime API:** Provides a dedicated realtime API for instant speech interactions.
*   **Highly Configurable:** Tailor `speaches` to your specific needs with extensive configuration options.

**Powered By:**

*   **Speech-to-Text:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
*   **Text-to-Speech:** [piper](https://github.com/rhasspy/piper) and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)

**Get Started:**

Explore the detailed documentation for installation and usage: [speaches.ai](https://speaches.ai/)

**Demos:**

*   **Realtime API:** [Link to demo video]
*   **Streaming Transcription:** (Coming Soon)
*   **Speech Generation:** [Link to demo video]

**Contribute & Support:**

If you encounter any issues, have questions, or suggestions, please [create an issue](https://github.com/speaches-ai/speaches/issues) in the repository.
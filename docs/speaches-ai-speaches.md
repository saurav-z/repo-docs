# Speaches: Your Open-Source AI Companion for Speech & Text

**Transform audio and text with ease using Speaches, an OpenAI API-compatible server that brings the power of speech-to-text, translation, and text-to-speech to your fingertips.**  (Original repo: [https://github.com/speaches-ai/speaches](https://github.com/speaches-ai/speaches))

## Key Features:

*   **OpenAI API Compatibility:** Seamlessly integrates with existing OpenAI tools and SDKs.
*   **Versatile Audio Generation:**
    *   Generate spoken audio summaries from text (Text-to-Speech).
    *   Perform sentiment analysis on audio recordings (Speech-to-Text).
    *   Enable asynchronous speech-to-speech interactions with AI models.
*   **Real-Time Streaming:** Receive transcriptions via Server-Sent Events (SSE) as the audio is processed, without waiting for completion.
*   **Dynamic Model Management:** Automatically loads and unloads models based on your requests, optimizing resource usage.
*   **High-Quality Text-to-Speech (TTS):** Utilizes `kokoro` (ranked #1 in the TTS Arena) and `piper` models for realistic voice generation.
*   **Flexible Hardware Support:** Runs efficiently on both GPUs and CPUs.
*   **Easy Deployment:** Deployable via Docker Compose and Docker.
*   **Real-Time API:** [Explore the Real-time API](https://speaches.ai/usage/realtime-api).
*   **Highly Configurable:** [Customize Speaches to your needs](https://speaches.ai/configuration/).

## Powered By:

*   **Speech-to-Text (STT):** [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
*   **Text-to-Speech (TTS):** [piper](https://github.com/rhasspy/piper) and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)

## Demos

*   **Realtime API:** [Insert the demo video link here]
*   **Streaming Transcription:** [TODO - Insert placeholder or description here]
*   **Speech Generation:** [Insert the demo video link here]

## Get Started

Visit our documentation for detailed installation instructions and usage guides: [speaches.ai](https://speaches.ai/)

## Contribute

Please create an issue to report bugs, ask questions, or suggest new features.
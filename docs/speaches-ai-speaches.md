# Speaches: Your Open-Source AI Companion for Speech & Audio Processing

**Tired of the limitations of proprietary speech APIs?** Speaches is an open-source, OpenAI API-compatible server empowering you with streaming transcription, translation, and speech generation, giving you complete control over your audio processing needs.  ([See the original repository](https://github.com/speaches-ai/speaches))

## Key Features:

*   **OpenAI API Compatibility:** Seamlessly integrates with existing tools and SDKs designed for the OpenAI API, making it easy to get started.
*   **Versatile Audio Generation:**
    *   Generate spoken audio summaries from text inputs.
    *   Perform sentiment analysis on audio recordings.
    *   Enable asynchronous speech-to-speech interactions with AI models.
*   **Real-Time Streaming:** Receive transcription results via Server-Sent Events (SSE) as audio is processed, eliminating wait times.
*   **Dynamic Model Management:** Automatically loads and unloads models based on request, optimizing resource usage.
*   **Text-to-Speech Capabilities:** Utilizes the top-rated `kokoro` (from the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) and `piper` models for high-quality speech generation.
*   **GPU & CPU Support:** Optimized to run efficiently on both CPU and GPU hardware.
*   **Docker Deployment:** Easily deployable using Docker Compose and Docker for streamlined setup.
*   **Real-time API:** Implement real-time audio processing.
*   **Highly Configurable:** Fine-tune the server to meet your specific requirements.

## Demos

### Realtime API

<img src="https://github.com/user-attachments/assets/457a736d-4c29-4b43-984b-05cc4d9995bc" alt="Realtime API Demo" width="500"/>

*(Note: Excuse the audio quality in the demo.)*

### Streaming Transcription

*   [TODO: Streaming Transcription Demo - Coming Soon]

### Speech Generation

<img src="https://github.com/user-attachments/assets/0021acd9-f480-4bc3-904d-831f54c4d45b" alt="Speech Generation Demo" width="500"/>

## Get Started

Explore the [speaches.ai](https://speaches.ai/) documentation for detailed installation instructions, usage guides, and configuration options.

## Contribute

Encounter a bug, have a question, or want to suggest a new feature? Please open an issue on the [GitHub repository](https://github.com/speaches-ai/speaches).
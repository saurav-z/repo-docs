# Speaches: Your Open-Source AI Companion for Speech & Audio

**Tired of being locked into proprietary AI services?** Speaches is an open-source, OpenAI API-compatible server that empowers you with cutting-edge speech-to-text (STT), text-to-speech (TTS), and audio generation capabilities, all running locally or in the cloud.  For the original source code, visit the [Speaches GitHub Repository](https://github.com/speaches-ai/speaches).

## Key Features:

*   **OpenAI API Compatibility:** Seamlessly integrate with existing tools and SDKs designed for OpenAI's API, simplifying development and deployment.
*   **Versatile Audio Generation:**
    *   Convert text to spoken audio summaries (text-in, audio-out).
    *   Perform sentiment analysis on audio recordings (audio-in, text-out).
    *   Enable asynchronous, speech-to-speech interactions (audio-in, audio-out).
*   **Real-time Streaming:** Receive transcription results via Server-Sent Events (SSE) as audio is processed, providing a responsive user experience.
*   **Dynamic Model Management:** Automatically load and unload STT and TTS models based on your requests, optimizing resource utilization.
*   **Multi-Model Support:** Utilize top-performing TTS models including `kokoro` (Ranked #1 in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) and `piper`.
*   **GPU and CPU Support:** Leverage the power of your hardware, with support for both GPUs and CPUs.
*   **Easy Deployment:** Deploy using Docker Compose or Docker for simplified setup and management.
*   **Realtime API:**  Take advantage of a Realtime API.
*   **Highly Configurable:** Customize Speaches to fit your specific needs.

## Technologies powering Speaches

*   **Speech-to-Text:** Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for high-performance transcription.
*   **Text-to-Speech:** Utilizing [piper](https://github.com/rhasspy/piper) and `kokoro` ([Hugging Face Link](https://huggingface.co/hexgrad/Kokoro-82M)) for generating natural-sounding speech.

## Get Started:

*   For installation instructions and usage guides, explore the official documentation at [speaches.ai](https://speaches.ai/).

## Demos

### Realtime API
<!-- Replace with the actual video, not just the link -->
[![Realtime API Demo](https://img.youtube.com/vi/YOUR_YOUTUBE_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_YOUTUBE_VIDEO_ID)

### Streaming Transcription
<!--  Add a demo video/gif when available -->
TODO

### Speech Generation
<!-- Replace with the actual video, not just the link -->
[![Speech Generation Demo](https://img.youtube.com/vi/YOUR_YOUTUBE_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_YOUTUBE_VIDEO_ID)


## Contribute & Support

If you encounter any issues, have questions, or want to suggest new features, please [open an issue](https://github.com/speaches-ai/speaches/issues) on the GitHub repository.
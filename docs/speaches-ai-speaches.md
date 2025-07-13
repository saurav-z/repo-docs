# Speaches: Your Open-Source AI Toolkit for Speech and Audio Processing

**Speaches empowers you to easily transcribe, translate, and generate speech using open-source AI models, offering an OpenAI API-compatible experience.** (Check out the original repository on GitHub: [speaches-ai/speaches](https://github.com/speaches-ai/speaches))

## Key Features:

*   **OpenAI API Compatibility:** Seamlessly integrate with existing OpenAI tools and SDKs.
*   **Versatile Audio Generation:**
    *   Text-to-speech (text input, audio output) for summarizing text or creating spoken content.
    *   Sentiment analysis on recordings (audio input, text output).
    *   Async speech-to-speech interactions.
*   **Real-time Streaming:** Get transcription results via Server-Sent Events (SSE) as audio is processed.
*   **Dynamic Model Management:** Automatically loads and unloads models based on demand, optimizing resource usage.
*   **Text-to-Speech (TTS) Support:** Utilizes the top-performing `kokoro` (as ranked in the [TTS Arena](https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena)) and `piper` models for high-quality speech synthesis.
*   **GPU and CPU Support:** Runs efficiently on both GPU and CPU hardware.
*   **Easy Deployment:** Deployable via Docker Compose and Docker.
*   **Realtime API:** Offers a real-time API for interactive applications.
*   **Highly Configurable:** Customize Speaches to fit your specific needs.

## Technologies Used:

*   **Speech-to-Text (STT):** Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
*   **Text-to-Speech (TTS):** Uses `piper` and `kokoro` models.

## Demos

### Realtime API

*(Video Demo will be added here)*

### Streaming Transcription

*(Video Demo will be added here)*

### Speech Generation

*(Video Demo will be added here)*

## Documentation

Detailed instructions for installation and usage can be found at [speaches.ai](https://speaches.ai/).

## Get Involved

Please create an issue on the [GitHub repository](https://github.com/speaches-ai/speaches) if you encounter any bugs, have questions, or have feature suggestions.
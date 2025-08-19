# KittenTTS: Realistic, Lightweight Text-to-Speech for Everyone

**KittenTTS is an open-source text-to-speech (TTS) model, delivering high-quality, realistic voices with exceptional performance and minimal resource requirements.**

[Visit the original repository on GitHub](https://github.com/KittenML/KittenTTS)

*Currently in developer preview*

**Join the KittenTTS Community:** [Discord](https://discord.com/invite/VJ86W4SURW)

**Need Custom Support?** [Fill out this form](https://docs.google.com/forms/d/e/1FAIpQLSc49erSr7jmh3H2yeqH4oZyRRuXm0ROuQdOgWguTzx6SMdUnQ/viewform?usp=preview)

**Questions?** Contact the creators at info@stellonlabs.com

## Key Features of KittenTTS

*   **Ultra-Lightweight Design:** The model size is under 25MB, perfect for resource-constrained environments.
*   **CPU-Optimized Performance:** Runs efficiently on any device without requiring a GPU.
*   **High-Quality Voice Options:** Experience realistic and engaging voices.
*   **Fast Inference:** Optimized for real-time speech synthesis, providing quick turnaround times.

## Getting Started: Quick Installation & Usage

### Installation

Install KittenTTS using pip:

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

### Basic Usage Example

Here's how to generate speech using KittenTTS:

```python
from kittentts import KittenTTS
m = KittenTTS("KittenML/kitten-tts-nano-0.1")

audio = m.generate("This high quality TTS model works without a GPU", voice='expr-voice-2-f' )

# available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

# Save the audio
import soundfile as sf
sf.write('output.wav', audio, 24000)
```

## System Requirements

KittenTTS is designed to run on a wide range of systems, making it accessible to everyone.

## Future Development Roadmap

*   [x] Release a preview model
*   [ ] Release the fully trained model weights
*   [ ] Release mobile SDK
*   [ ] Release web version
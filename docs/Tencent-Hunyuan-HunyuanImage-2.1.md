<p align="center">
  <img src="./assets/logo.png"  height=100>
</p>

<div align="center">
  <h1>HunyuanImage-2.1: Generate Stunning 2K Images from Text with an Open-Source Champion</h1>
</div>

<p align="center">Effortlessly transform your text prompts into breathtaking 2K (2048x2048) resolution images with HunyuanImage-2.1, the leading open-source text-to-image model! 🖼️</p>

<p align="center"> &nbsp&nbsp🤗 <a href="https://huggingface.co/tencent/HunyuanImage-2.1">HuggingFace</a>&nbsp&nbsp | 
💻 <a href="https://hunyuan.tencent.com/modelSquare/home/play?modelId=286&from=/visual">Official Website (Try the Model!)</a>&nbsp&nbsp
</p>

<p align="center">
    👏 Join our <a href="assets/WECHAT.md" target="_blank">WeChat</a> and <a href="https://discord.gg/ehjWMqF5wY">Discord</a>
</p>

**[Visit the original repository on GitHub](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1)**

---

HunyuanImage-2.1 is a state-of-the-art text-to-image model designed for high-resolution image generation.  This repository provides everything you need to utilize this powerful model, including model definitions, pre-trained weights, and inference code.

<div align="center">
  <img src="./assets/demo.jpg" width=100% alt="HunyuanImage 2.1 Demo">
</div>

## Key Features

*   **High-Resolution (2K) Generation:** Create stunning images at 2048x2048 resolution, perfect for detailed and cinematic visuals.
*   **Top-Tier Performance:** Ranked #1 on the Arena's leaderboard for text-to-image open-source models.
*   **Multilingual Support:**  Supports both English and Chinese prompts for diverse creative possibilities.
*   **Advanced Architecture:**  Built on a robust multi-modal DiT (Diffusion Transformer) backbone for superior image quality.
*   **Prompt Enhancement:**  Improve your results automatically with our built-in prompt enhancement.  Use the [PromptEnhancer-32B model](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B) for even greater detail.
*   **Flexible Aspect Ratios:** Generate images in various aspect ratios, including 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, and 2:3.
*   **Optimized for Efficiency:** Offers FP8 quantized models, enabling 2K image generation with as little as 24GB of GPU memory.

## Latest Updates

*   **September 18, 2025:** ✨ Enhanced prompt generation using the [PromptEnhancer-32B model](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B)
*   **September 18, 2025:** ✨ Explore the [ComfyUI workflow](https://github.com/KimbingNg/ComfyUI-HunyuanImage2.1) for easier model usage.
*   **September 16, 2025:** 👑  Achieved #1 rank on the Arena leaderboard for text-to-image open-source models ([Leaderboard](https://artificialanalysis.ai/text-to-image/arena/leaderboard-text)).
*   **September 12, 2025:** 🚀 Released FP8 quantized models for efficient 2K image generation on limited GPU memory.
*   **September 8, 2025:** 🚀 Released inference code and model weights for HunyuanImage-2.1.

## Architecture Overview

HunyuanImage-2.1 leverages a two-stage architecture for exceptional image generation:

1.  **Base Text-to-Image Model:** Employs a multimodal large language model (MLLM) and a multilingual, character-aware encoder to improve text-image alignment and text rendering accuracy.
2.  **Refiner Model:** Enhances image quality, clarity, and detail while minimizing artifacts.

## System Requirements

### Hardware and OS Requirements
*   NVIDIA GPU with CUDA support.
*   **Minimum:** 24 GB GPU memory for 2048x2048 image generation (with model CPU offloading and FP8 quantization enabled).
*   Supported operating system: Linux.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Tencent-Hunyuan/HunyuanImage-2.1.git
    cd HunyuanImage-2.1
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install flash-attn==2.7.3 --no-build-isolation
    ```

## Download Pretrained Models

Download pre-trained models following the instructions [here](ckpts/checkpoints-download.md).

## Usage

### Prompt Enhancement:  The Key to Quality

Crafting detailed and descriptive prompts is crucial for generating the best possible images.  We highly encourage you to leverage the [PromptEnhancer-32B model](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B) to enhance your prompts.

### Text-to-Image Generation

HunyuanImage-2.1 supports 2K image generation (e.g., 2048x2048 for 1:1, 2560x1536 for 16:9). Generating with 1K resolution will result in artifacts. We *highly recommend* using the full generation pipeline with prompt enhancement and the refiner model for optimal results.

| Model Type                    | Model Name                 | Description                                     | num_inference_steps | guidance_scale | shift |
|-------------------------------|----------------------------|-------------------------------------------------|---------------------|----------------|-------|
| Base Text-to-Image Model      | hunyuanimage2.1            | Undistilled model for the best quality          | 50                  | 3.5            | 5     |
| Distilled Text-to-Image Model | hunyuanimage2.1-distilled  | Distilled model for faster inference           | 8                   | 3.25           | 4     |
| Refiner                       | hunyuanimage-refiner       | Enhances image quality                         | N/A                 | N/A            | N/A   |

```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline

# Supported model_name: hunyuanimage-v2.1, hunyuanimage-v2.1-distilled
model_name = "hunyuanimage-v2.1"
pipe = HunyuanImagePipeline.from_pretrained(model_name=model_name, use_fp8=True)
pipe = pipe.to("cuda")

# The input prompt
prompt = "A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, standing in a painting studio, wearing a red knitted scarf and a red beret with the word \"Tencent\" on it, holding a paintbrush with a focused expression as it paints an oil painting of the Mona Lisa, rendered in a photorealistic photographic style."

# Generate with different aspect ratios
aspect_ratios = {
    "16:9": (2560, 1536),
    "4:3": (2304, 1792),
    "1:1": (2048, 2048),
    "3:4": (1792, 2304),
    "9:16": (1536, 2560),
}

width, height = aspect_ratios["1:1"]

image = pipe(
    prompt=prompt,
    width=width,
    height=height,
    # disable the reprompt if you already use the prompt enhancement to enhance the prompt
    use_reprompt=False,  # Enable prompt enhancement (which may result in higher GPU memory usage)
    use_refiner=True,   # Enable refiner model
    # For the distilled model, use 8 steps for faster inference.
    # For the non-distilled model, use 50 steps for better quality.
    num_inference_steps=8 if "distilled" in model_name else 50,
    guidance_scale=3.25 if "distilled" in model_name else 3.5,
    shift=4 if "distilled" in model_name else 5,
    seed=649151,
)

image.save("generated_image.png")
```

## More Examples

Explore the versatility of HunyuanImage-2.1 with these example prompts and generated images. Longer, more detailed prompts often yield the best results.  Experiment with the examples provided.

<div align="center">
  <img src="./assets/more_cases.jpg" width=100% alt="HunyuanImage 2.1 Demo">
</div>

<p align="center">
<table>
<thead>
<tr>
    <th>Index</th>  <th>User Prompt</th> <th>Image</th>
</tr>
</thead>
<tbody>
<tr>
    <td>1</td> <td>宏伟教堂的内部，穹顶下方的中央矗立着一尊小巧的维纳斯雕像，微微侧对镜头。雕像没有双手，布满裂纹，表面若干古老的水泥片剥落，露出内部真人质感的牛奶肌肤。雕像穿着薄薄的白色婚纱，在雕像的身后，一只浮空水泥断手轻轻提起长长的婚纱拖尾；在雕像的头顶上方，另一只浮空水泥断手正为她戴上一个由白色花朵组成的花环，雕像本身是没有双手的。教堂穹顶上布满彩色玻璃窗，一束阳光从上往下照射到雕像上，形成丁达尔效应，光斑点点洒在雕像的脸庞和胸前。充满神性的光辉，背景微微虚化，物体的边缘模糊柔和。拉斐尔前派的梦幻朦胧美学风格。</td> <td><img src="./assets/demo_case1.png" width=100%></td>
</tr>
<tr>
    <td>2</td> <td>A hyper-realistic photograph of a crystal ball diorama sitting atop fluffy forest moss and surrounded by scattered sunlight. Inside, detailed diorama features a Tencent meeting room, an animated chat bubble sculpture, and several joyful penguins—one wearing a graduation cap, others playing soccer and waving tiny banners. The base of the crystal sphere boldly presents ""Tencent"" in large, crisp, white 3D letters. Background is softly blurred and bokeh-rich, emphasizing the cute, vibrant details of the sphere.</td>  <td><img src="./assets/demo_case2.png" width=100%></td>
</tr>
<tr>
    <td>3</td> <td>A close-up portrait of an elderly Italian man with deeply wrinkled skin, expressive hazel eyes, and a neatly trimmed white mustache. His olive-toned complexion shows the marks of sun and age, and he wears a flat cap slightly tilted to the side. He smiles faintly, revealing warmth and wisdom, while holding a small espresso cup in one hand. The softly blurred background shows a rustic stone wall with climbing ivy, captured in a realistic photography style.</td> <td><img src="./assets/demo_case3.png" width=100%></td>
</tr>
<tr>
    <td>4</td> <td>An open vintage suitcase on a neutral, softly lit background. The suitcase is made of deep brown, worn leather with visible scuffs and creases, and its interior is lined with dark, plush fabric. Inside the suitcase is a meticulously crafted miniature landscape of China, featuring the Great Wall of China winding across model mountains, the pagoda roofs of the Forbidden City, and a representation of the terracotta army, all interwoven with vibrant green rice paddies.  On the side of the suitcase, a text "China" is labeled. The entire diorama is bathed in warm, ethereal light, with a dreamy lens bloom and soft, glowing highlights. Photorealistic style, ultra-detailed textures, cinematic lighting.</td> <td><img src="./assets/demo_case4.png" width=100%></td>
</tr>
</tbody>
</table>
</p>

### Prompt Enhancement Showcase

HunyuanImage-2.1 automatically rewrites prompts to improve image quality.  See the difference!

<p align="center">
<table>
<thead>
<tr>
    <th>Index</th>  <th>User Prompt</th> <th>Prompt Enhanced</th> <th>Image</th>
</tr>
</thead>
<tbody>
<tr>
    <td>1</td> <td>Wildlife poster for Serengeti plains. Wide-eyed chibi explorer riding friendly lion cub. 'Serengeti: Roar of Adventure' in whimsical font. 'Where Dreams Run Wild' tagline. Warm yellows and soft browns.</td> <td> A wildlife poster design for the Serengeti plains features a central illustration of a chibi-style explorer riding a lion cub, set against a backdrop of rolling hills. At the top of the composition, the title "Serengeti: Roar of Adventure" is displayed in a large, whimsical font with decorative, swirling letters. The main scene depicts a wide-eyed chibi explorer, characterized by a large head and a small body, sitting atop a friendly lion cub. The explorer wears a green explorer's hat, a backpack, and holds onto the cub's mane, looking forward with a look of wonder. The lion cub, with a light brown mane and a smiling expression, strides forward, its body rendered in warm orange tones. In the background, the Serengeti plains are illustrated with rolling hills and savanna grass, all in shades of warm yellow and soft brown. Below the main illustration, the tagline "Where Dreams Run Wild" is written in a smaller, elegant script. The overall presentation is that of a poster design, combining a cute chibi illustration style with playful, whimsical typography.</td> <td><img src="./assets/demo_case5.png" width=100%></td>
</tr>
<tr>
    <td>2</td> <td>Energetic poster for New York City. Anime businesswoman hailing a taxi with skyscrapers and Times Square signs around. 'NYC: Bright Ambitions' in urban graffiti font. 'Own Every Dream' tagline. Saturated yellows, reds, and sharp blues.</td> <td>An energetic poster for New York City unfolds, featuring a dynamic scene with an anime-style businesswoman in the midst of hailing a taxi. The central figure is a young woman with large, expressive eyes and dark hair styled in a bob, wearing a professional blue business suit with motion lines indicating movement. She stands on a bustling street, her arms outstretched as she calls for a classic yellow taxi cab that is approaching. In the background, towering skyscrapers with sleek, anime-inspired architecture rise into the sky, adorned with vibrant, glowing billboards and neon signs characteristic of Times Square. Across the top of the poster, the text "NYC: Bright Ambitions" is displayed in a large, stylized urban graffiti font, with spray-paint-like edges. Below this main title, the tagline "Own Every Dream" is written in a smaller, clean font. The entire composition is rendered with saturated colors, dominated by bright yellows, reds, and sharp blues. The overall presentation is a fusion of anime illustration and graphic design.</td> <td><img src="./assets/demo_case6.png" width=100%></td>
</tr>
<tr>
    <td>3</td> <td>An artistic studio portrait captures a high fashion model in a striking, dynamic pose. Her face is a canvas for avant-garde makeup, defined by bold, geometric applications of primary colors. She wears a sculptural, unconventional garment, emphasizing clean lines and form. The scene is illuminated by dramatic studio lighting, creating sharp contrasts and highlighting her features against an abstract, blurred background of colors. The image is presented in a realistic photography style.</td> <td> An artistic studio portrait captures a high fashion model in a striking, dynamic pose, her body twisted with one arm raised high to convey energy and movement. Her face serves as a canvas for avant-garde makeup, featuring bold, geometric applications of primary colors; vibrant yellow triangles are painted on her forehead, and electric blue lines accentuate her eye sockets. She wears a sculptural, unconventional garment made of a stiff, matte white fabric, with asymmetrical panels that wrap around her torso, emphasizing clean lines and form. Illuminated by dramatic studio lighting, with a strong beam from the side casting sharp shadows and highlighting the contours of her face and body against an abstract, blurred background of purples and oranges, creating a bokeh effect. Realistic photography style. </td> <td><img src="./assets/demo_case7.png" width=100%></td>
</tr>
<tr>
    <td>4</td> <td>An environmental portrait of a chef, captured with a focused expression in a bustling kitchen. He holds culinary tools, his gaze fixed on his work, embodying passion and creativity. The background is a blur of motion with stainless steel counters, all illuminated by a warm ambient light. The image is presented in a realistic photography style.</td> <td> An environmental portrait of a male chef in the midst of work within a bustling kitchen. The chef, as the central subject and viewed from the chest up, has a focused expression with a furrowed brow, his gaze directed downward at the culinary tools he holds. He wears a professional white chef‘s jacket and a traditional toque, with flour lightly dusting his face and clothes. In his hands, he grips a large chef’s knife and a metal spatula, poised over an unseen cooking surface. The background is a dynamic blur of motion, with out-of-focus shapes of stainless steel counters, pots, and other kitchen equipment suggesting a busy environment. Warm ambient light from overhead fixtures casts a golden hue, creating highlights on the chef‘s jacket and the tools. Realistic photography style, characterized by a shallow depth of field that emphasizes the subject while conveying the energy and creativity of the kitchen. </td>  <td><img src="./assets/demo_case8.png" width=100%></td>
</tr>
</tbody>
</table>
</p>

## Comparisons

### SSAE Evaluation

SSAE (Structured Semantic Alignment Evaluation) is an intelligent evaluation metric for image-text alignment based on advanced multimodal large language models (MLLMs). We extracted 3500 key points across 12 categories, then used multimodal large language models to automatically evaluate and score by comparing the generated images with these key points based on the visual content of the images. Mean Image Accuracy represents the image-wise average score across all key points, while Global Accuracy directly calculates the average score across all key points.

<p align="center">
<table>
<thead>
<tr>
    <th rowspan="2">Model</th>  <th rowspan="2">Open Source</th> <th rowspan="2">Mean Image Accuracy</th> <th rowspan="2">Global Accuracy</th> <th colspan="4" style="text-align: center;">Primary Subject</th> <th colspan="3" style="text-align: center;">Secondary Subject</th> <th colspan="2" style="text-align: center;">Scene</th> <th colspan="3" style="text-align: center;">Other</th>
</tr>
<tr>
    <th>Noun</th> <th>Key Attributes</th> <th>Other Attributes</th> <th>Action</th> <th>Noun</th> <th>Attributes</th> <th>Action</th> <th>Noun</th> <th>Attributes</th> <th>Shot</th> <th>Style</th> <th>Composition</th>
</tr>
</thead>
<tbody>
<tr>
    <td>FLUX-dev</td> <td>✅</td> <td>0.7122</td> <td>0.6995</td> <td>0.7965</td> <td>0.7824</td> <td>0.5993</td> <td>0.5777</td> <td>0.7950</td> <td>0.6826</td> <td>0.6923</td> <td>0.8453</td> <td>0.8094</td> <td>0.6452</td> <td>0.7096</td> <td>0.6190</td>
</tr>
<tr>
    <td>Seedream-3.0</td> <td>❌</td> <td>0.8827</td> <td>0.8792</td> <td>0.9490</td> <td>0.9311</td> <td>0.8242</td> <td>0.8177</td> <td>0.9747</td> <td>0.9103</td> <td>0.8400</td> <td>0.9489</td> <td>0.8848</td> <td>0.7582</td> <td>0.8726</td> <td>0.7619</td>
</tr>
<tr>
    <td>Qwen-Image</td> <td>✅</td> <td>0.8854</td> <td>0.8828</td> <td>0.9502</td> <td>0.9231</td> <td>0.8351</td> <td>0.8161</td> <td>0.9938</td> <td>0.9043</td> <td>0.8846</td> <td>0.9613</td> <td>0.8978</td> <td>0.7634</td> <td>0.8548</td> <td>0.8095</td>
</tr>
<tr>
    <td>GPT-Image</td>  <td>❌</td> <td> 0.8952</td> <td>0.8929</td> <td>0.9448</td> <td>0.9289</td> <td>0.8655</td> <td>0.8445</td> <td>0.9494</td> <td>0.9283</td> <td>0.8800</td> <td>0.9432</td> <td>0.9017</td> <td>0.7253</td> <td>0.8582</td> <td>0.7143</td>
</tr>
<tr>
    <td><strong>HunyuanImage 2.1</strong></td> <td>✅</td> <td><strong>0.8888</strong></td> <td><strong>0.8832</strong></td> <td>0.9339</td> <td>0.9341</td> <td>0.8363</td> <td>0.8342</td> <td>0.9627</td> <td>0.8870</td> <td>0.9615</td> <td>0.9448</td> <td>0.9254</td> <td>0.7527</td> <td>0.8689</td> <td>0.7619</td>
</tr>
</tbody>
</table>
</p>

HunyuanImage-2.1 outperforms other open-source models and comes very close to the performance of closed-source commercial models in terms of semantic alignment, according to SSAE evaluations.

### GSB Evaluation

<p align="center">
  <img src="./assets/gsb.png" width=70% alt="Human Evaluation with Other Models">
</p>

HunyuanImage 2.1 demonstrates competitive image generation quality compared to commercial models, and shows advantages over similar open-source models. The results validate the model's technical advancement in text-to-image generation.

### Contact

Join our Discord server or WeChat groups for community support, collaboration, and to ask any questions. Your feedback is crucial in helping us drive HunyuanImage forward! Feel free to open an issue or submit a pull request on GitHub.

## BibTeX

```BibTeX
@misc{HunyuanImage-2.1,
  title={HunyuanImage 2.1: An Efficient Diffusion Model for High-Resolution (2K) Text-to-Image Generation},
  author={Tencent Hunyuan Team},
  year={2025},
  howpublished={\url{https://github.com/Tencent-Hunyuan/HunyuanImage-2.1}},
}
```

## Acknowledgements

We appreciate the contributions of the following open-source projects and communities: [Qwen](https://huggingface.co/Qwen), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co).

## Github Star History
<a href="https://star-history.com/#Tencent-Hunyuan/HunyuanImage-2.1&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanImage-2.1&type=Date1&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanImage-2.1&type=Date1" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanImage-2.1&type=Date1" />
 </picture>
</a>
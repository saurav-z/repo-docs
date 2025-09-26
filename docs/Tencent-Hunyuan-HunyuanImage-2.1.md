<p align="center">
  <img src="./assets/logo.png"  height=100>
</p>

<div align="center">

# HunyuanImage-2.1: Generate Stunning 2K Images with an Open-Source Diffusion Model

</div>

<p align="center">Create breathtaking 2K resolution images from text prompts using HunyuanImage-2.1, an open-source model with state-of-the-art performance.  <br> Explore the model on the <a href="https://huggingface.co/tencent/HunyuanImage-2.1">Hugging Face Hub</a> and the <a href="https://hunyuan.tencent.com/modelSquare/home/play?modelId=286&from=/visual">official website</a>.</p>


<p align="center">
    👏 Join our <a href="assets/WECHAT.md" target="_blank">WeChat</a> and <a href="https://discord.gg/ehjWMqF5wY">Discord</a>
</p>

---

Harness the power of **HunyuanImage-2.1**, a 17B parameter text-to-image model, to bring your creative visions to life in stunning 2K (2048 x 2048) resolution.  This repository contains everything you need to run the model, including model definitions, pre-trained weights, and inference code.  Explore the model directly on the [official website](https://hunyuan.tencent.com/modelSquare/home/play?modelId=286&from=/visual) and view impressive examples on our [project page](https://hunyuan.tencent.com/image/en?tabIndex=0).  Find the original source code and documentation on the original [GitHub repository](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1).

<div align="center">
  <img src="./assets/demo.jpg" width=100% alt="HunyuanImage 2.1 Demo">
</div>

## ✨ **Key Features**

*   **High-Resolution Imagery:** Generate detailed 2K (2048x2048) images.
*   **Multilingual Support:**  Works natively with both Chinese and English prompts.
*   **Advanced Architecture:** Based on a multi-modal DiT backbone.
*   **Glyph-Aware Text Rendering:** Utilizes ByT5 for improved text generation.
*   **Flexible Aspect Ratios:** Supports various aspect ratios (1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3).
*   **Prompt Enhancement:** Integrated prompt rewriting for superior visual results.

## 📰 **What's New**

*   **[September 18, 2025]:** ✨ Try the [PromptEnhancer-32B model](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B) for higher-quality prompt enhancement!​.
*   **[September 18, 2025]:** ✨ [ComfyUI workflow of HunyuanImage-2.1](https://github.com/KimbingNg/ComfyUI-HunyuanImage2.1) is available now!
*   **[September 16, 2025]:** 👑 Top1 on the Arena leaderboard for text-to-image open-source models. [Leaderboard](https://artificialanalysis.ai/text-to-image/arena/leaderboard-text)
*   **[September 12, 2025]:** 🚀 Released FP8 quantized models!  Generate 2K images with just 24GB GPU memory!
*   **[September 8, 2025]:** 🚀 Inference code and model weights for HunyuanImage-2.1 released.

## 💻 **Introduction**

HunyuanImage-2.1 is a powerful text-to-image model capable of generating high-resolution (2K) images.  It leverages advanced techniques for superior text-image alignment and detail control, leading to improved consistency and realism.

Our architecture consists of two key stages:

1.  **Base Text-to-Image Model:** This stage uses a multimodal large language model (MLLM) and a character-aware encoder to enhance image-text alignment and text rendering across multiple languages.
2.  **Refiner Model:** This second stage further enhances image quality, reduces artifacts, and refines the generated images.

👑 HunyuanImage-2.1 achieved the **Top1** on Arena's leaderboard for open-source text-to-image models.

<div align="center">
  <img src="./assets/leaderboard.png" width=70% alt="HunyuanImage 2.1 Demo">
</div>

## 🚀 **System Requirements**

**Hardware and OS:**

*   NVIDIA GPU with CUDA support.
    *   **Minimum Requirement:** 24 GB GPU memory for 2048x2048 image generation.
        *   **Note:** The memory requirements are measured with model CPU offloading and FP8 quantization enabled. You can disable offloading for faster inference if your GPU has enough memory.
*   Supported operating system: Linux.

## ⚙️ **Installation**

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

## 💾 **Download Pretrained Models**

Find instructions on how to download the pretrained models [here](ckpts/checkpoints-download.md).

## 🎬 **Usage**

###  **Prompt Enhancement**

For the best results, craft detailed and descriptive prompts.  We recommend using the [PromptEnhancer-32B model](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B) to further improve your prompts.

###  **Text-to-Image Generation**

HunyuanImage-2.1 supports 2K image generation (e.g., 2048x2048 for 1:1 images, 2560x1536 for 16:9 images). Using other resolutions may result in artifacts.

**Highly recommended:** Utilize the full generation pipeline (prompt enhancement and refiner) for optimal image quality.

| Model Type                | Model Name                | Description                                       | `num_inference_steps` | `guidance_scale` | `shift` |
| ------------------------- | ------------------------- | ------------------------------------------------- | --------------------- | ---------------- | ------- |
| Base Text-to-Image Model  | `hunyuanimage2.1`         | Undistilled model (best quality).                 | 50                    | 3.5              | 5       |
| Distilled Text-to-Image Model | `hunyuanimage2.1-distilled` | Distilled model (faster inference).               | 8                     | 3.25             | 4       |
| Refiner                   | `hunyuanimage-refiner`      | The refiner model to improve image quality         | N/A                   | N/A              | N/A     |

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

## 🖼️ **More Examples**

HunyuanImage-2.1 can generate diverse and creative images from complex instructions.

<div align="center">
  <img src="./assets/more_cases.jpg" width=100% alt="HunyuanImage 2.1 Demo">
</div>

We recommend using longer, more detailed prompts.  You can also use the example prompts provided in the original README.

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

## 🏆 **Comparisons & Benchmarks**

###  SSAE Evaluation

SSAE (Structured Semantic Alignment Evaluation) is an intelligent metric for image-text alignment based on advanced multimodal large language models (MLLMs). We extracted 3500 key points across 12 categories, then used multimodal large language models to automatically evaluate and score by comparing the generated images with these key points based on the visual content of the images. Mean Image Accuracy represents the image-wise average score across all key points, while Global Accuracy directly calculates the average score across all key points.
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

HunyuanImage 2.1 achieves the top performance among open-source models for semantic alignment, nearing the performance of closed-source models like GPT-Image.

###  GSB Evaluation

<p align="center">
  <img src="./assets/gsb.png" width=70% alt="Human Evaluation with Other Models">
</p>

HunyuanImage-2.1's image generation quality is comparable to closed-source commercial models.  It outperforms the open-source Qwen-Image, demonstrating its technical advancements.

## 💬 **Contact and Community**

Join our Discord server or WeChat groups to connect with other users, share ideas, and ask questions! You are also welcome to open an issue or submit a pull request on GitHub. Your feedback is valued!

## 📚 **BibTeX**

```BibTeX
@misc{HunyuanImage-2.1,
  title={HunyuanImage 2.1: An Efficient Diffusion Model for High-Resolution (2K) Text-to-Image Generation},
  author={Tencent Hunyuan Team},
  year={2025},
  howpublished={\url{https://github.com/Tencent-Hunyuan/HunyuanImage-2.1}},
}
```

## 🙏 **Acknowledgements**

We thank the following open-source projects: [Qwen](https://huggingface.co/Qwen), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co).

## ⭐ **GitHub Star History**

[<img src="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanImage-2.1&type=Date1" alt="Star History Chart" />](https://star-history.com/#Tencent-Hunyuan/HunyuanImage-2.1&Date)
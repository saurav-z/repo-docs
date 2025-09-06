<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
</p>

<p align="center">
    💜 <a href="https://chat.qwen.ai/">Qwen Chat</a> |
    🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace (T2I)</a> |
    🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace (Edit)</a> |
    🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a> |
    🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a> |
    📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> |
    📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog (T2I)</a> |
    📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog (Edit)</a>
    <br>
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a> |
    🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a> |
    💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a> |
    🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>
</p>

---

## **Qwen-Image: Unleash Your Creativity with Advanced Image Generation and Editing**

Qwen-Image is a powerful 20B MMDiT image foundation model offering cutting-edge capabilities in text-to-image generation and precise image editing. ([See the original repo](https://github.com/QwenLM/Qwen-Image))

### **Key Features:**

*   **Exceptional Text Rendering:** Generate images with unparalleled accuracy in rendering complex text, especially for Chinese.
*   **Advanced Image Editing:** Edit images with style transfer, object manipulation, and text editing capabilities.
*   **Versatile Artistic Styles:** Generate images in a wide range of styles, from photorealistic to artistic.
*   **Image Understanding:** Support for object detection, semantic segmentation, and more, enabling intelligent image editing.
*   **Multi-Language Support:**  Optimized for both English and Chinese prompts.

### **News & Updates:**

*   **2025.08.19:** Update to the latest diffusers commit for improved Qwen-Image-Edit performance, especially regarding identity preservation and instruction following.
*   **2025.08.18:** Qwen-Image-Edit is now open-sourced!
*   **2025.08.09:** Support for LoRA models, such as MajicBeauty LoRA.
*   **2025.08.05:** Native support in ComfyUI and integration with Qwen Chat.
*   **2025.08.05:** Technical Report released.
*   **2025.08.04:** Qwen-Image weights released on Hugging Face and ModelScope.
*   **2025.08.04:** Qwen-Image launched!

### **Quick Start**

1.  **Prerequisites:** Ensure you have transformers >= 4.51.3
2.  **Install Dependencies:**

    ```bash
    pip install git+https://github.com/huggingface/diffusers
    ```

    (See the original repo for detailed code examples on Text-to-Image and Image Editing.)

### **Advanced Usage**

*   **Prompt Enhancement:** Leverage Qwen-Plus and Qwen-VL-Max for enhanced prompt optimization and stability. Example scripts are available in the repository.

### **Deploy Qwen-Image**

*   **Multi-GPU API Server:** Supports multi-GPU parallel processing, queue management, and automatic prompt optimization. Configuration via environment variables is available.

    ```bash
    export NUM_GPUS_TO_USE=4
    export TASK_QUEUE_SIZE=100
    export TASK_TIMEOUT=300
    ```

    To start the demo server:

    ```bash
    cd src
    DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxx python examples/demo.py
    ```

### **Show Cases & Tutorials**

*   **General Cases:**  Showcases examples of high-fidelity text rendering, various artistic styles, and image editing capabilities.
*   **Image Editing Tutorial:**  Step-by-step guide demonstrating semantic and appearance editing techniques with Qwen-Image-Edit, including text editing and novel view synthesis.

### **AI Arena**

*   **AI Arena:** An open benchmarking platform ( [AI Arena](https://aiarena.alibaba-inc.com) ) built on the Elo rating system for fair and transparent model evaluation.  View the latest leaderboard at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

### **Community Support**

*   **Hugging Face:** Supports Qwen-Image directly in Diffusers.
*   **ModelScope:** Offers comprehensive support with DiffSynth-Studio, DiffSynth-Engine, and ModelScope AIGC Central.
*   **WaveSpeedAI:** Deployed Qwen-Image on their platform.
*   **LiblibAI:** Provides native support for Qwen-Image.
*   **cache-dit:** Supports cache acceleration for Qwen-Image.

### **License**

*   Qwen-Image is licensed under Apache 2.0.

### **Citation**

```bibtex
@misc{wu2025qwenimagetechnicalreport,
      title={Qwen-Image Technical Report}, 
      author={Chenfei Wu and Jiahao Li and Jingren Zhou and Junyang Lin and Kaiyuan Gao and Kun Yan and Sheng-ming Yin and Shuai Bai and Xiao Xu and Yilei Chen and Yuxiang Chen and Zecheng Tang and Zekai Zhang and Zhengyi Wang and An Yang and Bowen Yu and Chen Cheng and Dayiheng Liu and Deqing Li and Hang Zhang and Hao Meng and Hu Wei and Jingyuan Ni and Kai Chen and Kuan Cao and Liang Peng and Lin Qu and Minggang Wu and Peng Wang and Shuting Yu and Tingkun Wen and Wensen Feng and Xiaoxiao Xu and Yi Wang and Yichang Zhang and Yongqiang Zhu and Yujia Wu and Yuxuan Cai and Zenan Liu},
      year={2025},
      eprint={2508.02324},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.02324}, 
}
```

### **Contact and Join Us**

*   Join the Discord and WeChat groups for discussions and collaboration.
*   Contribute via issues and pull requests on GitHub.
*   Contact fulai.hr@alibaba-inc.com for FTE and research intern opportunities.

### **Star History**

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)
```
Key improvements and explanations:

*   **SEO-Optimized Hook:**  A concise, keyword-rich sentence opens the README, grabbing attention and hinting at key capabilities.
*   **Clear Headings:**  Organized the information with clear, descriptive headings.
*   **Bulleted Key Features:**  Highlights the core functionalities for quick understanding.
*   **Concise Descriptions:** Simplified the descriptions, focusing on what's important.
*   **Emphasis on Benefits:**  Focuses on what the model *does* for the user (e.g., "Unleash Your Creativity").
*   **Direct Links:** Included links to relevant resources for easy navigation.
*   **Clear Instructions:** Streamlined the Quick Start section.
*   **Contact Information:** Emphasized how to connect with the team.
*   **Code Snippets:** Kept the essential code snippets
*   **Removed Duplication:** Avoided repetitive information.
*   **Readability:** Used formatting (bold, italics, lists) to improve readability.
*   **Consistent Formatting:** Applied consistent formatting for headings, links, and lists.
*   **Contextual Information**: Improved the context.
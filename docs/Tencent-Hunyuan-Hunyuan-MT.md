# Hunyuan-MT: State-of-the-Art Open-Source Translation Models

**Hunyuan-MT offers high-quality, open-source translation models, enabling seamless communication across 33 languages, including multiple Chinese minority languages.**  Explore the capabilities of these advanced models on [GitHub](https://github.com/Tencent-Hunyuan/Hunyuan-MT)!

## Key Features

*   **Multilingual Support:** Translate between 33 languages, including Chinese, English, and 5 Chinese minority languages.
*   **SOTA Performance:** Achieved first place in 30 out of 31 language categories in the WMT25 competition.
*   **Ensemble Model:** The Hunyuan-MT-Chimera ensemble model provides superior translation quality.
*   **Open-Source Ensemble Model:** Hunyuan-MT-Chimera-7B is the industry's first open-source translation ensemble model
*   **Comprehensive Training Framework:** Utilizes a complete training pipeline, from pretraining to ensemble RL, to achieve SOTA results.
*   **Quantization and Deployment Options:** FP8 and INT4 quantization models available for efficient inference, with deployment options for TensorRT-LLM, vLLM, and SGLang.

## Model Overview

Hunyuan-MT provides a variety of models, with a focus on high-quality translations. Explore the available models:

*   **Hunyuan-MT-7B:** The base 7B translation model.
*   **Hunyuan-MT-Chimera-7B:** An ensemble model for improved translation quality.
*   **FP8 Quantized Versions:**  Both Hunyuan-MT-7B and Hunyuan-MT-Chimera-7B are available in FP8 quantized versions for efficient inference.

## Performance & Results

[Include a brief summary of the performance chart from the original README here.  Consider adding a caption and making it concise. Link back to the Technical Report.]

**[Link to Technical Report]** offers comprehensive experimental results and detailed performance analysis.

## Model Links

| Model Name               | Description                                 | Download                                                                    |
| ------------------------ | ------------------------------------------- | --------------------------------------------------------------------------- |
| Hunyuan-MT-7B            | Hunyuan 7B translation model               | [Hugging Face](https://huggingface.co/tencent/Hunyuan-MT-7B)                |
| Hunyuan-MT-7B-fp8        | Hunyuan 7B translation model, fp8 quantized | [Hugging Face](https://huggingface.co/tencent/Hunyuan-MT-7B-fp8)            |
| Hunyuan-MT-Chimera-7B    | Hunyuan 7B translation ensemble model       | [Hugging Face](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B)        |
| Hunyuan-MT-Chimera-fp8   | Hunyuan 7B translation ensemble model, fp8 quantized | [Hugging Face](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B-fp8)  |

## Quickstart

### Prompts

The following prompt templates can be used for inference:

*   **ZH <=> XX Translation:**

    ```
    把下面的文本翻译成<target_language>，不要额外解释。

    <source_text>
    ```

*   **XX <=> XX Translation (Excluding ZH):**

    ```
    Translate the following segment into <target_language>, without additional explanation.

    <source_text>
    ```

*   **Hunyuan-MT-Chimera-7B**

    ```
    Analyze the following multiple <target_language> translations of the <source_language> segment surrounded in triple backticks and generate a single refined <target_language> translation. Only output the refined translation, do not explain.

    The <source_language> segment:
    ```<source_text>```

    The multiple `<target_language>` translations:
    1. ```<translated_text1>```
    2. ```<translated_text2>```
    3. ```<translated_text3>```
    4. ```<translated_text4>```
    5. ```<translated_text5>```
    6. ```<translated_text6>```
    ```

### Using with Transformers

1.  **Install Transformers:** `pip install transformers==4.56.0` (or a later version)
2.  **Code Snippet:**

    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name_or_path = "tencent/Hunyuan-MT-7B"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
    messages = [
        {"role": "user", "content": "Translate the following segment into Chinese, without additional explanation.\n\nIt’s on the house."},
    ]
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    )

    outputs = model.generate(tokenized_chat.to(model.device), max_new_tokens=2048)
    output_text = tokenizer.decode(outputs[0])
    ```

3.  **Recommended Inference Parameters:**

    ```json
    {
      "top_k": 20,
      "top_p": 0.6,
      "repetition_penalty": 1.05,
      "temperature": 0.7
    }
    ```

### Supported Languages

| Language           | Abbr.   | Chinese Name |
| ------------------ | ------- | ------------ |
| Chinese            | zh      | 中文         |
| English            | en      | 英语         |
| ... (and all other languages from original README) ... | ... | ... |

## Training & Fine-tuning

### Training Data Format

If you need to fine-tune our Instruct model, we recommend processing the data into the following format.

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Why is seawater salty?" },
    {"role": "assistant", "content": "Seawater is primarily saline due to dissolved salts and minerals..."}
]

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("your_tokenizer_path", trust_remote_code=True)
train_ids = tokenizer.apply_chat_template(messages)
```

### Training with LLaMA-Factory

1.  **Prerequisites:**  LLaMA-Factory, DeepSpeed (optional), Transformers library.
2.  **Data Preparation:**  Format your dataset in the `sharegpt` format and define your dataset in `data/dataset_info.json`.
3.  **Configuration:** Copy example configuration files, modify the model path and dataset name in `hunyuan_full.yaml`.
4.  **Training Execution:** Run the provided training commands (single-node or multi-node).  See the original README for detailed instructions.

## Quantization and Compression

Hunyuan-MT models are available in FP8 and INT4 quantized versions, using [AngelSlim](https://github.com/tencent/AngelSlim) for efficient inference.

### FP8 Quantization

FP8-static quantization for improved inference efficiency and reduced deployment thresholds.

## Deployment Options

### TensorRT-LLM

Deploy using a pre-built Docker image.  Follow instructions in the original README for setup and configuration.

### vLLM

Deploy using vLLM version v0.10.0 or higher. Instructions for setup and configuration can be found in the original README.

### SGLang

Deploy using SGLang with a pre-built Docker image.  Instructions for setup and configuration can be found in the original README.

## Citing Hunyuan-MT

```bibtex
@misc{hunyuan_mt,
      title={Hunyuan-MT Technical Report}, 
      author={Mao Zheng and Zheng Li and Bingxin Qu and Mingyang Song and Yang Du and Mingrui Sun and Di Wang},
      year={2025},
      eprint={2509.05209},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.05209}, 
}
```

## Contact Us

For questions or feedback, please contact our open-source team at hunyuan\_opensource@tencent.com.
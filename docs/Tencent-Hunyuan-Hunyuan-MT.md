# Hunyuan-MT: State-of-the-Art Multilingual Translation Models

**Hunyuan-MT offers advanced, open-source multilingual translation capabilities, achieving top performance across numerous language pairs.** [Access the GitHub Repository](https://github.com/Tencent-Hunyuan/Hunyuan-MT)

*   [Hugging Face](https://huggingface.co/collections/tencent/hunyuan-mt-68b42f76d473f82798882597)
*   [ModelScope](https://modelscope.cn/collections/Hunyuan-MT-2ca6b8e1b4934f)
*   [Official Website](https://hunyuan.tencent.com)
*   [Demo](https://hunyuan.tencent.com/chat/HunyuanDefault?from=modelSquare&modelId=hunyuan-mt-7b)
*   [Technical Report](https://www.arxiv.org/pdf/2509.05209)

## Key Features

*   **Industry-Leading Performance:** Achieved first place in 30 out of 31 language categories in the WMT25 competition.
*   **Open-Source Ensemble Model:** Hunyuan-MT-Chimera-7B is the industry's first open-source translation ensemble model, improving translation quality.
*   **Comprehensive Training Framework:** Utilizes a cutting-edge training pipeline (Pretrain -> CPT -> SFT -> Translation RL -> Ensemble RL) for SOTA results.
*   **Extensive Language Support:** Supports mutual translation for 33 languages, including five Chinese ethnic minority languages.
*   **Quantization and Deployment:** Provides optimized models (FP8, INT4) and deployment options using TensorRT-LLM, vLLM, and SGLang.

## Model Overview

Hunyuan-MT includes two primary model types:

*   **Hunyuan-MT-7B:** A base translation model.
*   **Hunyuan-MT-Chimera:** An ensemble model that combines multiple translations for enhanced accuracy.

## Model Performance

<div align='center'>
<img src="imgs/overall_performance.png" width = "80%" />
</div>

*For detailed performance metrics and analysis, please refer to our [technical report](https://www.arxiv.org/pdf/2509.05209).*

## Quick Links

| Model Name                    | Description                                         | Download                                                                   |
| ----------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------- |
| Hunyuan-MT-7B                 | Hunyuan 7B translation model                       | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B)                    |
| Hunyuan-MT-7B-fp8             | Hunyuan 7B translation model, fp8 quant             | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B-fp8)                |
| Hunyuan-MT-Chimera            | Hunyuan 7B translation ensemble model              | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B)           |
| Hunyuan-MT-Chimera-fp8        | Hunyuan 7B translation ensemble model, fp8 quant    | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B-fp8)       |

## Prompt Templates

### ZH <=> XX Translation

```
把下面的文本翻译成<target_language>，不要额外解释。

<source_text>
```

### XX <=> XX Translation (Excluding ZH)

```
Translate the following segment into <target_language>, without additional explanation.

<source_text>
```

### Hunyuan-MT-Chimera-7B

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

## Usage with Transformers

1.  **Install Transformers:**

    ```shell
    pip install transformers==4.56.0
    ```

    *Note: If you want to load fp8 model with transformers, you need to change the name "ignored_layers" in config.json to "ignore" and upgrade the compressed-tensors to compressed-tensors-0.11.0.*

2.  **Example Code:**

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

## Supported Languages

| Languages         | Abbr.   | Chinese Names   |
|-------------------|---------|-----------------|
| Chinese           | zh      | 中文            |
| English           | en      | 英语            |
| French            | fr      | 法语            |
| Portuguese        | pt      | 葡萄牙语        |
| Spanish           | es      | 西班牙语        |
| Japanese          | ja      | 日语            |
| Turkish           | tr      | 土耳其语        |
| Russian           | ru      | 俄语            |
| Arabic            | ar      | 阿拉伯语        |
| Korean            | ko      | 韩语            |
| Thai              | th      | 泰语            |
| Italian           | it      | 意大利语        |
| German            | de      | 德语            |
| Vietnamese        | vi      | 越南语          |
| Malay             | ms      | 马来语          |
| Indonesian        | id      | 印尼语          |
| Filipino          | tl      | 菲律宾语        |
| Hindi             | hi      | 印地语          |
| Traditional Chinese | zh-Hant| 繁体中文        |
| Polish            | pl      | 波兰语          |
| Czech             | cs      | 捷克语          |
| Dutch             | nl      | 荷兰语          |
| Khmer             | km      | 高棉语          |
| Burmese           | my      | 缅甸语          |
| Persian           | fa      | 波斯语          |
| Gujarati          | gu      | 古吉拉特语      |
| Urdu              | ur      | 乌尔都语        |
| Telugu            | te      | 泰卢固语        |
| Marathi           | mr      | 马拉地语        |
| Hebrew            | he      | 希伯来语        |
| Bengali           | bn      | 孟加拉语        |
| Tamil             | ta      | 泰米尔语        |
| Ukrainian         | uk      | 乌克兰语        |
| Tibetan           | bo      | 藏语            |
| Kazakh            | kk      | 哈萨克语        |
| Mongolian         | mn      | 蒙古语          |
| Uyghur            | ug      | 维吾尔语        |
| Cantonese         | yue     | 粤语            |

## Fine-tuning Guidelines

To fine-tune the Instruct model, format your data as follows:

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

## Training with LLaMA-Factory

Instructions are provided for training using LLaMA-Factory.  Please consult the original README for detailed steps, including data preparation, dataset configuration, and training execution.

## Quantization and Compression

Hunyuan-MT models are available in FP8 and INT4 quantized versions, offering improved efficiency.

### FP8 Quantization

*   Uses FP8-static quantization.
*   Weights and activation values are converted to FP8 format, enhancing inference speed and reducing deployment overhead.
*   Utilize AngelSlim tool for quantization or directly download pre-quantized models from Hugging Face.

## Deployment Options

### TensorRT-LLM

*   **Docker Image:** Pre-built Docker image provided.
*   **Get Started:** `docker pull docker.cnb.cool/tencent/hunyuan/hunyuan-7b:hunyuan-7b-trtllm`
*   **Run Container:**  Instructions provided to prepare the config file and run the container.

### vLLM

*   **Requirements:** vLLM version v0.10.0 or higher and Transformers (v4.56.0).
*   **Model Download:** Hugging Face or ModelScope.
*   **API Server Startup:** Command provided to start the API server.
*   **Testing:**  `curl` command to test the API.
*   **Quantized Model Deployment:** Guides for INT8, INT4 (GPTQ), and FP8 model deployment using vLLM.

### SGLang

*   **Docker Image:** Pre-built Docker image available.
*   **Get Started:**  `docker pull lmsysorg/sglang:latest`
*   **API Server Startup:** Command to start the API server.

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

## Contact

For questions, feedback, or to reach the development team, please contact us at hunyuan\_opensource@tencent.com.
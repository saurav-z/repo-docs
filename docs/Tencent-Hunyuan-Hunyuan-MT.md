# Hunyuan-MT: State-of-the-Art Multilingual Translation Models

**Unlock seamless communication across 33 languages with the powerful Hunyuan-MT translation models from Tencent, achieving top performance in the WMT25 competition.**  Learn more and explore the models on the [original repository](https://github.com/Tencent-Hunyuan/Hunyuan-MT).

<p align="center">
 <img src="https://dscache.tencent-cloud.cn/upload/uploader/hunyuan-64b418fd052c033b228e04bc77bbc4b54fd7f5bc.png" width="400"/> <br>
</p><p></p>

<p align="center">
    🤗&nbsp;<a href="https://huggingface.co/collections/tencent/hunyuan-mt-68b42f76d473f82798882597"><b>Hugging Face</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    <img src="https://avatars.githubusercontent.com/u/109945100?s=200&v=4" width="16"/>&nbsp;<a href="https://modelscope.cn/collections/Hunyuan-MT-2ca6b8e1b4934f"><b>ModelScope</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
</p>

<p align="center">
    🖥️&nbsp;<a href="https://hunyuan.tencent.com" style="color: red;"><b>Official Website</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    🕹️&nbsp;<a href="https://hunyuan.tencent.com/chat/HunyuanDefault?from=modelSquare&modelId=hunyuan-mt-7b"><b>Demo</b></a>&nbsp;&nbsp;&nbsp;&nbsp;
</p>

<p align="center">
    <a href="https://github.com/Tencent-Hunyuan/Hunyuan-MT"><b>GITHUB</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="https://www.arxiv.org/pdf/2509.05209"><b>Technical Report</b> </a>
</p>

## Key Features

*   **Superior Performance:** Achieved first place in 30 out of 31 language categories in the WMT25 competition.
*   **High-Quality Translations:**  Hunyuan-MT-7B delivers industry-leading performance for its size, while Hunyuan-MT-Chimera-7B sets a new standard with its ensemble approach.
*   **Open-Source Ensemble Model:**  Benefit from the industry's first open-source translation ensemble model for enhanced accuracy.
*   **Comprehensive Training Framework:** Utilizes an advanced framework, including pretraining, continual pretraining, supervised fine-tuning, and ensemble RL for optimal results.
*   **Extensive Language Support:** Supports mutual translation among 33 languages, including several Chinese ethnic minority languages.

## Models Available

Explore the available Hunyuan-MT models:

| Model Name                | Description                                      | Download                                                                       |
| :------------------------ | :----------------------------------------------- | :----------------------------------------------------------------------------- |
| Hunyuan-MT-7B             | Hunyuan 7B translation model                     | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B)                       |
| Hunyuan-MT-7B-fp8         | Hunyuan 7B translation model, FP8 quantized      | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B-fp8)                   |
| Hunyuan-MT-Chimera       | Hunyuan 7B translation ensemble model             | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B)               |
| Hunyuan-MT-Chimera-fp8    | Hunyuan 7B translation ensemble model, FP8 quantized | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B-fp8)           |

## Usage and Examples

### Prompt Templates

Use these templates for your translation tasks:

*   **ZH<=>XX Translation:**

```
把下面的文本翻译成<target_language>，不要额外解释。

<source_text>
```

*   **XX<=>XX Translation (excluding ZH<=>XX):**

```
Translate the following segment in <target_language>, without additional explanation.

<source_text>
```

*   **Hunyuan-MT-Chimera-7B:**

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

### Quick Start with Transformers

1.  **Install Transformers (v4.56.0 Recommended):**

    ```bash
    pip install transformers==4.56.0
    ```

    *   *Note:*  If loading the FP8 model, modify "ignored\_layers" to "ignore" in `config.json` and upgrade `compressed-tensors` to version 0.11.0.

2.  **Example Code:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name_or_path = "tencent/Hunyuan-MT-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")  # You may want to use bfloat16 and/or move to GPU here
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

### Training Data Format

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Why is seawater salty?" },
    {"role": "assistant", "content": "Seawater is primarily saline due to dissolved salts and minerals. These substances come from the chemical materials in rocks and soil on the Earth's surface, which are carried into the ocean over time. When seawater evaporates, the water vapor leaves, but the salts and minerals remain, making the seawater saltier. Therefore, the salinity of seawater is determined by the amount of salts and minerals it contains."}
]

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("your_tokenizer_path", trust_remote_code=True)
train_ids = tokenizer.apply_chat_template(messages)
```

## Fine-tuning with LLaMA-Factory

*   **Prerequisites:**  Ensure you have LLaMA-Factory and its dependencies (including DeepSpeed - optional, and the specific Transformers branch - see README) installed as per the linked documentation.
*   **Data Preparation:** Format your training data in a `sharegpt` format within the `data` directory of LLaMA-Factory (see examples in the original README).  Define the dataset in `data/dataset_info.json`.
*   **Configuration:** Copy configurations from the `llama_factory_support/example_configs` to `example/hunyuan` in `LLaMA-Factory`. Edit `hunyuan_full.yaml` with your model path and dataset name.
*   **Training:** Use the provided single-node or multi-node training commands (example in original README), setting the `DISABLE_VERSION_CHECK` environment variable if needed.

## Quantization and Compression

Hunyuan-MT models support FP8 and INT4 quantization using AngelSlim, a tool for user-friendly model compression.  You can either use AngelSlim directly or download the pre-quantized models.

## Deployment Options

Deploy Hunyuan-MT using frameworks like **TensorRT-LLM**, **vLLM**, or **SGLang** to create an OpenAI-compatible API.

*   **TensorRT-LLM:** Docker image and example commands. (See original README for detailed steps)
*   **vLLM:** Instructions on how to deploy and use the vLLM server.
    *   **Installation:** You can install vLLM by following the guidance in the original repo.
    *   **Starting API Server:** You can start the API server using the provided example code.
    *   **Quantitative Model Deployment:** You can quantitative model deployment, including Int8, Int4, and FP8 in the original README.
*   **SGLang:** Docker image and example server commands. (See original README for detailed steps)

## Citation

If you use Hunyuan-MT, please cite the following:

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

For questions and feedback, contact the open-source team at hunyuan\_opensource@tencent.com.
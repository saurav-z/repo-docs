# Hunyuan-MT: State-of-the-Art Open-Source Translation Models

**Hunyuan-MT empowers seamless multilingual communication with cutting-edge translation technology.**  Check out the original repository [here](https://github.com/Tencent-Hunyuan/Hunyuan-MT).

<p align="center">
 <img src="https://dscache.tencent-cloud.cn/upload/uploader/hunyuan-64b418fd052c033b228e04bc77bbc4b54fd7f5bc.png" width="400"/> <br>
</p>

## Key Features

*   🏆 **Industry-Leading Performance:** Achieved first place in 30 out of 31 language categories in the WMT25 competition.
*   ✨ **High-Quality Translation:**  Offers superior translation quality, especially with the Hunyuan-MT-Chimera ensemble model.
*   🌐 **Multilingual Support:** Supports mutual translation among 33 languages, including Chinese and five Chinese ethnic minority languages.
*   🛠️ **Open-Source Ensemble Model:**  Hunyuan-MT-Chimera is the first open-source translation ensemble model in the industry.
*   🔄 **Comprehensive Training Framework:**  Employs an advanced training framework (pretrain -> CPT -> SFT -> translation RL -> ensemble RL) for optimal results.
*   🚀 **Model Variety:** Available models include both a base translation model and an ensemble model (Hunyuan-MT-Chimera).

## Key Advantages

*   **Hunyuan-MT-7B:**  Industry-leading performance among models of a similar size.
*   **Hunyuan-MT-Chimera-7B:** Elevates translation quality to a new level, setting a new benchmark for open-source models.

## Quick Links

*   🤗 **Hugging Face:** [Hugging Face Link](https://huggingface.co/collections/tencent/hunyuan-mt-68b42f76d473f82798882597)
*   🖥️ **Official Website:** [Official Website](https://hunyuan.tencent.com)
*   🕹️ **Demo:** [Demo](https://hunyuan.tencent.com/chat/HunyuanDefault?from=modelSquare&modelId=hunyuan-mt-7b)
*   📝 **Technical Report:** [Technical Report](https://www.arxiv.org/pdf/2509.05209)

## Model Introduction

Hunyuan-MT encompasses both a single translation model, Hunyuan-MT-7B, and an ensemble model, Hunyuan-MT-Chimera. The Hunyuan-MT-7B model is designed for direct source-to-target language translation, while the Hunyuan-MT-Chimera model combines multiple translation outputs for enhanced accuracy. The models provide support for mutual translation across 33 languages.

## Model Links

| Model Name                  | Description                                         | Download                                                                           |
| :-------------------------- | :-------------------------------------------------- | :--------------------------------------------------------------------------------- |
| Hunyuan-MT-7B               | Hunyuan 7B translation model                      | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B)                        |
| Hunyuan-MT-7B-fp8           | Hunyuan 7B translation model, fp8 quant            | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B-fp8)                       |
| Hunyuan-MT-Chimera          | Hunyuan 7B translation ensemble model            | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B)                  |
| Hunyuan-MT-Chimera-fp8      | Hunyuan 7B translation ensemble model, fp8 quant | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B-fp8)             |

## Prompts

### Prompt Templates

*   **ZH<=>XX Translation:**

```
把下面的文本翻译成<target_language>，不要额外解释。

<source_text>
```

*   **XX<=>XX Translation (Excluding ZH<=>XX):**

```
Translate the following segment into <target_language>, without additional explanation.

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

## Usage

### Install

```bash
pip install transformers==4.56.0
```

### Python Example

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

### Recommended Parameters for Inference

```json
{
  "top_k": 20,
  "top_p": 0.6,
  "repetition_penalty": 1.05,
  "temperature": 0.7
}
```

## Training

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

### Train with LLaMA-Factory

*   **Prerequisites**: Install LLaMA-Factory, DeepSpeed (optional), and the transformers library with the specified commit.
*   **Data Preparation**:
    *   Organize your data in the `sharegpt` format.
    *   Define your dataset in `data/dataset_info.json`.
*   **Training Execution**:
    *   Copy configuration files.
    *   Modify the model path and dataset name in the configuration file `hunyuan_full.yaml`.
    *   Run the training command (single-node or multi-node).

## Quantization Compression

### FP8 Quantization

Hunyuan-MT models use FP8-static quantization to improve inference efficiency. The [AngelSlim](https://github.com/tencent/AngelSlim) tool is utilized for model compression.  FP8 quantization converts model weights and activations to an 8-bit floating-point format.

## Deployment

### Frameworks

You can deploy the model using:

*   TensorRT-LLM
*   vLLM
*   SGLang

### TensorRT-LLM

*   **Docker Image:** Use pre-built Docker images.
*   **Configuration:** Create an extra configuration file.
*   **Start the API Server:** Use the provided command to launch the server.

### vLLM

*   **Install:** Install the transformers library.
*   **Model Download:** Download model files (from Hugging Face or ModelScope).
*   **API Server:** Start the API server.
*   **Request:** Send requests to the API endpoint.

### SGLang

*   **Docker Image:** Use pre-built Docker images.
*   **Start the API Server:** Use the provided command to launch the server.

## Citation

If you use Hunyuan-MT in your research, please cite the following paper:

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

For inquiries, feedback, or to connect with the R&D and product teams, please contact the open-source team at hunyuan\_opensource@tencent.com.
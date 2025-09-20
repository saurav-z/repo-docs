# Hunyuan-MT: State-of-the-Art Multilingual Translation Models

Hunyuan-MT is a suite of advanced multilingual translation models offering superior performance across numerous languages.  **Explore the power of high-quality translation with Hunyuan-MT!**  [See the original repository](https://github.com/Tencent-Hunyuan/Hunyuan-MT).

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Hunyuan--MT-blue?style=flat-square&logo=huggingface)](https://huggingface.co/collections/tencent/hunyuan-mt-68b42f76d473f82798882597)
[![ModelScope](https://img.shields.io/badge/ModelScope-Hunyuan--MT-green?style=flat-square&logo=data-visualization)](https://modelscope.cn/collections/Hunyuan-MT-2ca6b8e1b4934f)
[![Official Website](https://img.shields.io/badge/Official%20Website-Hunyuan-red?style=flat-square&logo=tencent)](https://hunyuan.tencent.com)
[![Demo](https://img.shields.io/badge/Demo-Hunyuan--MT-orange?style=flat-square)](https://hunyuan.tencent.com/chat/HunyuanDefault?from=modelSquare&modelId=hunyuan-mt-7b)
[![GitHub](https://img.shields.io/badge/GitHub-Hunyuan--MT-gray?style=flat-square&logo=github)](https://github.com/Tencent-Hunyuan/Hunyuan-MT)
[![Technical Report](https://img.shields.io/badge/Technical%20Report-arXiv-purple?style=flat-square&logo=arxiv)](https://www.arxiv.org/pdf/2509.05209)

## Key Features and Advantages

*   **Industry-Leading Performance:** Achieved first place in 30 out of 31 language categories in the WMT25 competition.
*   **High-Quality Translation:** Hunyuan-MT-Chimera, the first open-source ensemble translation model, sets a new standard for translation accuracy.
*   **Superior Model:** Hunyuan-MT-7B delivers industry-leading performance among models of comparable scale.
*   **Comprehensive Framework:**  Employs a robust training framework (pretrain → CPT → SFT → translation RL → ensemble RL) for SOTA results.
*   **Wide Language Support:**  Supports mutual translation among 33 languages, including five Chinese ethnic minority languages.

## Models

| Model Name  | Description | Download |
| ----------- | ----------- |-----------
| Hunyuan-MT-7B  | Hunyuan 7B translation model |🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B)|
| Hunyuan-MT-7B-fp8 | Hunyuan 7B translation model，fp8 quant    | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B-fp8)|
| Hunyuan-MT-Chimera | Hunyuan 7B translation ensemble model    | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B)|
| Hunyuan-MT-Chimera-fp8 | Hunyuan 7B translation ensemble model，fp8 quant     | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B-fp8)|

## Performance

<div align='center'>
<img src="imgs/overall_performance.png" width = "80%" />
</div>

For detailed experimental results and analysis, please refer to our [Technical Report](https://www.arxiv.org/pdf/2509.05209).

## Prompts

### Prompt Template for ZH<=>XX Translation
```
把下面的文本翻译成<target_language>，不要额外解释。

<source_text>
```

### Prompt Template for XX<=>XX Translation, excluding ZH<=>XX
```
Translate the following segment to <target_language>, without additional explanation.

<source_text>
```

### Prompt Template for Hunyuan-MT-Chimera-7B
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

## Usage and Deployment

### Installation

```shell
pip install transformers==4.56.0
```

*Note: If you want to load the fp8 model with transformers, you need to change the name "ignored_layers" in config.json to "ignore" and upgrade the compressed-tensors to compressed-tensors-0.11.0.*

### Inference

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

### Recommended Inference Parameters

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
| Urdu            | ur      | 乌尔都语        |
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

### Fine-tuning with LLaMA-Factory

*   **Prerequisites:** [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), DeepSpeed (optional), and a compatible branch of the `transformers` library.
*   **Data Preparation:** Prepare your data in the `sharegpt` format.
*   **Configuration:** Modify the model path and dataset name in `hunyuan_full.yaml`.
*   **Training:** Execute the training commands (single-node or multi-node) as specified in the original README.

## Quantization and Compression

Utilizing the [AngelSlim](https://github.com/tencent/AngelSlim) tool, we offer FP8 and INT4 quantized models.  These compressed models enhance inference efficiency and reduce deployment requirements.

### FP8 Quantization

Hunyuan-MT supports FP8-static quantization for improved inference.

## Deployment

Hunyuan-MT models can be deployed using frameworks like:

*   **TensorRT-LLM:**  Pre-built Docker images are provided.  See the original README for detailed setup and usage instructions.
*   **vLLM:** (version v0.10.0 or higher)  Follow the instructions in the original README for model download and API server setup, and instructions for quantitative deployment with Int8, Int4, and FP8 models.
*   **SGLang:**  Pre-built Docker images are also available. Consult the original README for setup.

### Citing Hunyuan-MT

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

For inquiries or feedback, please contact the open-source team or email us at  [hunyuan\_opensource@tencent.com](mailto:hunyuan_opensource@tencent.com).
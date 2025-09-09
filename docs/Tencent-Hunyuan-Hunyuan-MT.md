# Hunyuan-MT: State-of-the-Art Multilingual Translation Models

**Hunyuan-MT offers superior multilingual translation capabilities, achieving top performance and supporting 33 languages, including minority languages.** [Explore the Hunyuan-MT Repository](https://github.com/Tencent-Hunyuan/Hunyuan-MT).

---

## Key Features and Advantages:

*   🥇 **Top Performance:** Achieved first place in 30 out of 31 language categories in the WMT25 competition.
*   🚀 **Industry-Leading:** Hunyuan-MT-7B delivers exceptional performance compared to similarly sized models.
*   🧠 **Ensemble Model:** Hunyuan-MT-Chimera-7B is the first open-source translation ensemble model, enhancing translation quality.
*   ⚙️ **Comprehensive Training:** Implements a robust training pipeline: pretrain → CPT → SFT → translation RL → ensemble RL, leading to SOTA results.
*   🌐 **Multilingual Support:** Translates between 33 languages, including Chinese and five Chinese minority languages.
*   ⚡ **Quantization and Deployment options**: The models can be deployed via TensorRT-LLM, vLLM, or SGLang.

---

## Model Details and Resources

### Model Links

| Model Name | Description | Download |
| ----------- | ----------- | ----------- |
| Hunyuan-MT-7B | Hunyuan 7B translation model | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B) |
| Hunyuan-MT-7B-fp8 | Hunyuan 7B translation model，fp8 quant | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B-fp8) |
| Hunyuan-MT-Chimera | Hunyuan 7B translation ensemble model | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B) |
| Hunyuan-MT-Chimera-fp8 | Hunyuan 7B translation ensemble model，fp8 quant | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B-fp8) |

### Prompts

#### Prompt Template for ZH<=>XX Translation:

```
把下面的文本翻译成<target_language>，不要额外解释。

<source_text>
```

#### Prompt Template for XX<=>XX Translation, excluding ZH<=>XX:

```
Translate the following segment into <target_language>, without additional explanation.

<source_text>
```

#### Prompt Template for Hunyuan-MT-Chimera-7B:

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

### Supported Languages

| Languages | Abbr. | Chinese Names |
|---|---|---|
| Chinese | zh | 中文 |
| English | en | 英语 |
| French | fr | 法语 |
| Portuguese | pt | 葡萄牙语 |
| Spanish | es | 西班牙语 |
| Japanese | ja | 日语 |
| Turkish | tr | 土耳其语 |
| Russian | ru | 俄语 |
| Arabic | ar | 阿拉伯语 |
| Korean | ko | 韩语 |
| Thai | th | 泰语 |
| Italian | it | 意大利语 |
| German | de | 德语 |
| Vietnamese | vi | 越南语 |
| Malay | ms | 马来语 |
| Indonesian | id | 印尼语 |
| Filipino | tl | 菲律宾语 |
| Hindi | hi | 印地语 |
| Traditional Chinese | zh-Hant | 繁体中文 |
| Polish | pl | 波兰语 |
| Czech | cs | 捷克语 |
| Dutch | nl | 荷兰语 |
| Khmer | km | 高棉语 |
| Burmese | my | 缅甸语 |
| Persian | fa | 波斯语 |
| Gujarati | gu | 古吉拉特语 |
| Urdu | ur | 乌尔都语 |
| Telugu | te | 泰卢固语 |
| Marathi | mr | 马拉地语 |
| Hebrew | he | 希伯来语 |
| Bengali | bn | 孟加拉语 |
| Tamil | ta | 泰米尔语 |
| Ukrainian | uk | 乌克兰语 |
| Tibetan | bo | 藏语 |
| Kazakh | kk | 哈萨克语 |
| Mongolian | mn | 蒙古语 |
| Uyghur | ug | 维吾尔语 |
| Cantonese | yue | 粤语 |

### Usage and Deployment

*   **Transformers Library:** Integrate Hunyuan-MT using the `transformers` library. See the example code in the original README for details.
*   **Inference Parameters:** Recommended inference parameters are included.
*   **Training Data Format:** Guidance on data format for fine-tuning the model.
*   **Training with LLaMA-Factory:** Step-by-step instructions for fine-tuning with LLaMA-Factory, including prerequisites, data preparation, and execution commands.
*   **Quantization Compression:** Learn about using FP8 and INT4 quantization, with links to quantization models and the AngelSlim compression tool.
*   **Deployment Options:**
    *   **TensorRT-LLM:**  Provides a Docker image and configuration examples for deployment.
    *   **vLLM:**  Instructions for deploying with vLLM, including example API server setups and quantitative model deployment.
    *   **SGLang:** Includes details on launching a server with SGLang and using a Docker image.

---

## Example Code (Transformers)

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

---

## Contact Us

For questions or feedback, contact the open-source team or send an email to hunyuan\_opensource@tencent.com.

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
# Hunyuan-MT: State-of-the-Art Machine Translation Models

**Hunyuan-MT offers high-quality machine translation across 33 languages, including Chinese minority languages, delivering leading performance and open-source solutions.**  [Visit the Original Repository](https://github.com/Tencent-Hunyuan/Hunyuan-MT)

---

## Key Features & Advantages

*   **Top Performance:** Achieved first place in 30 out of 31 language categories in the WMT25 competition.
*   **Industry-Leading Models:** Hunyuan-MT-7B sets a high bar for models of its size, and Hunyuan-MT-Chimera-7B is the first open-source ensemble translation model.
*   **Comprehensive Framework:** Implements a complete training process from pretraining to ensemble reinforcement learning, SOTA results are acheived.
*   **Multilingual Support:** Supports mutual translation between 33 languages.
*   **FP8 & INT4 Quantization:**  Support for models quantized with AngelSlim for reduced resource usage.

---

## Quick Links

*   🤗 **Hugging Face:** [Hunyuan-MT Models](https://huggingface.co/collections/tencent/hunyuan-mt-68b42f76d473f82798882597)
*   🖥️ **Official Website:** [Tencent Hunyuan](https://hunyuan.tencent.com)
*   🕹️ **Demo:** [Hunyuan Chat Demo](https://hunyuan.tencent.com/chat/HunyuanDefault?from=modelSquare&modelId=hunyuan-mt-7b)
*   📄 **Technical Report:** [arXiv](https://www.arxiv.org/pdf/2509.05209)

---

## Model Details

Hunyuan-MT provides the following models:

| Model Name                | Description                                      | Download                                                              |
| ------------------------- | ------------------------------------------------ | --------------------------------------------------------------------- |
| Hunyuan-MT-7B             | Hunyuan 7B translation model                   | [Hugging Face](https://huggingface.co/tencent/Hunyuan-MT-7B)        |
| Hunyuan-MT-7B-fp8         | Hunyuan 7B translation model, FP8 quantized        | [Hugging Face](https://huggingface.co/tencent/Hunyuan-MT-7B-fp8)    |
| Hunyuan-MT-Chimera-7B     | Hunyuan 7B translation ensemble model            | [Hugging Face](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B) |
| Hunyuan-MT-Chimera-fp8    | Hunyuan 7B translation ensemble model, FP8 quantized | [Hugging Face](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B-fp8) |

---

## Usage

### Prompts

Use the appropriate prompt template based on your translation needs:

*   **ZH <-> XX Translation:**

    ```
    把下面的文本翻译成<target_language>，不要额外解释。

    <source_text>
    ```

*   **XX <-> XX Translation (excluding ZH <-> XX):**

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

### Quickstart with Transformers

1.  **Install Transformers:**
    ```bash
    pip install transformers==4.56.0
    ```

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

---

## Supported Languages

| Languages           | Abbr.   | Chinese Names   |
| ------------------- | ------- | --------------- |
| Chinese             | zh      | 中文            |
| English             | en      | 英语            |
| French              | fr      | 法语            |
| Portuguese          | pt      | 葡萄牙语        |
| Spanish             | es      | 西班牙语        |
| Japanese            | ja      | 日语            |
| Turkish             | tr      | 土耳其语        |
| Russian             | ru      | 俄语            |
| Arabic              | ar      | 阿拉伯语        |
| Korean              | ko      | 韩语            |
| Thai                | th      | 泰语            |
| Italian             | it      | 意大利语        |
| German              | de      | 德语            |
| Vietnamese          | vi      | 越南语          |
| Malay               | ms      | 马来语          |
| Indonesian          | id      | 印尼语          |
| Filipino            | tl      | 菲律宾语        |
| Hindi               | hi      | 印地语          |
| Traditional Chinese | zh-Hant | 繁体中文        |
| Polish              | pl      | 波兰语          |
| Czech               | cs      | 捷克语          |
| Dutch               | nl      | 荷兰语          |
| Khmer               | km      | 高棉语          |
| Burmese             | my      | 缅甸语          |
| Persian             | fa      | 波斯语          |
| Gujarati            | gu      | 古吉拉特语      |
| Urdu                | ur      | 乌尔都语        |
| Telugu              | te      | 泰卢固语        |
| Marathi             | mr      | 马拉地语        |
| Hebrew              | he      | 希伯来语        |
| Bengali             | bn      | 孟加拉语        |
| Tamil               | ta      | 泰米尔语        |
| Ukrainian           | uk      | 乌克兰语        |
| Tibetan             | bo      | 藏语            |
| Kazakh            | kk      | 哈萨克语        |
| Mongolian           | mn      | 蒙古语          |
| Uyghur              | ug      | 维吾尔语        |
| Cantonese           | yue     | 粤语            |

---

## Fine-tuning

### Training Data Format

*   If you need to fine-tune our Instruct model, we recommend processing the data into the following format:

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

1.  **Prerequisites**:
    *   Install **LLaMA-Factory** (follow the [official guide](https://github.com/hiyouga/LLaMA-Factory)).
    *   Install **DeepSpeed** (optional, follow the [official guide](https://github.com/deepspeedai/DeepSpeed#installation)).
    *   **Transformer Library**: Install the correct branch:
        ```bash
        pip install git+https://github.com/huggingface/transformers@4970b23cedaf745f963779b4eae68da281e8c6ca
        ```

2.  **Data Preparation:**
    *   Organize your data in `json` format, and place it in the `data` directory in `LLaMA-Factory`
    *   Define your dataset in the `data/dataset_info.json` file.

3.  **Training Execution:**

    *   Copy config files from the `llama_factory_support/example_configs` directory to the `example/hunyuan` directory in `LLaMA-Factory`.
    *   Modify the model path and dataset name in the configuration file `hunyuan_full.yaml`.
    *   Execute training commands (single or multi-node).

---

## Quantization & Compression

Hunyuan-MT models are available in both FP8 and INT4 quantized versions for efficient inference, created using [AngelSlim](https://github.com/tencent/AngelSlim).

### FP8 Quantization

Use FP8-static quantization.

---

## Deployment

Deploy your Hunyuan-MT models using:

*   **TensorRT-LLM**
*   **vLLM**
*   **SGLang**

### TensorRT-LLM

*   Provides a pre-built Docker image.
*   Follow the instructions to pull the image, configure, and run the API server.

### vLLM

*   Supports serving the model with bfloat16, and quantized weights.
*   Follow the instructions to install dependencies, and set environment variables.
*   Deploy the API server

### SGLang

*   Provides a pre-built Docker image.
*   Follow the instructions to pull the image, and start the API server.

---

## Citing Hunyuan-MT

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

---

## Contact

For inquiries, contact our open-source team or via email: hunyuan_opensource@tencent.com.
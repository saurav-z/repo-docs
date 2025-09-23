# Hunyuan-MT: Revolutionizing Machine Translation

**Hunyuan-MT offers state-of-the-art machine translation, excelling in 33 languages and delivering superior performance.**  ([Original Repo](https://github.com/Tencent-Hunyuan/Hunyuan-MT))

<p align="center">
 <img src="https://dscache.tencent-cloud.cn/upload/uploader/hunyuan-64b418fd052c033b228e04bc77bbc4b54fd7f5bc.png" width="400"/> <br>
</p><p></p>

## Key Features

*   **Leading Performance:** Achieved first place in 30 out of 31 language categories in the WMT25 competition.
*   **High-Quality Translation:**  Hunyuan-MT-Chimera-7B, an open-source ensemble model, enhances translation quality.
*   **Extensive Language Support:**  Supports mutual translation among 33 languages, including Chinese and 5 Chinese ethnic minority languages.
*   **Comprehensive Training Framework:**  Employs a robust training pipeline (pretrain -> CPT -> SFT -> translation RL -> ensemble RL) to achieve SOTA results.
*   **Open Source and Accessible:**  Hunyuan-MT-7B and Hunyuan-MT-Chimera-7B are available on Hugging Face and ModelScope.

## Quick Links

*   🤗 [Hugging Face](https://huggingface.co/collections/tencent/hunyuan-mt-68b42f76d473f82798882597)
*   🖥️ [Official Website](https://hunyuan.tencent.com)
*   🕹️ [Demo](https://hunyuan.tencent.com/chat/HunyuanDefault?from=modelSquare&modelId=hunyuan-mt-7b)
*   <img src="https://avatars.githubusercontent.com/u/109945100?s=200&v=4" width="16"/> [ModelScope](https://modelscope.cn/collections/Hunyuan-MT-2ca6b8e1b4934f)
*   [Technical Report](https://www.arxiv.org/pdf/2509.05209)

## Model Overview

Hunyuan-MT comprises two main models:

*   **Hunyuan-MT-7B:** A translation model.
*   **Hunyuan-MT-Chimera:** An ensemble model that combines multiple translation outputs.

## Model Links

| Model Name             | Description                                          | Download                                                                   |
| ---------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------- |
| Hunyuan-MT-7B          | Hunyuan 7B translation model                       | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B)                    |
| Hunyuan-MT-7B-fp8      | Hunyuan 7B translation model, fp8 quantized         | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B-fp8)                |
| Hunyuan-MT-Chimera     | Hunyuan 7B translation ensemble model                | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B)           |
| Hunyuan-MT-Chimera-fp8 | Hunyuan 7B translation ensemble model, fp8 quantized | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B-fp8)       |

## Prompts

Provides clear prompt templates for various translation scenarios, including Chinese and other languages, and for the Hunyuan-MT-Chimera-7B ensemble model.

### Prompt Templates

*   **ZH<=>XX Translation:**
    ```
    把下面的文本翻译成<target_language>，不要额外解释。

    <source_text>
    ```
*   **XX<=>XX Translation (Excluding ZH):**
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

## Usage with Transformers

Instructions and code snippets for using the Transformers library.  This includes how to install the necessary packages and example code for loading and using the model,  along with recommended inference parameters.

**Install Transformers (v4.56.0 recommended):**

```shell
pip install transformers==4.56.0
```

**Example Code:**

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

**Recommended Inference Parameters:**

```json
{
  "top_k": 20,
  "top_p": 0.6,
  "repetition_penalty": 1.05,
  "temperature": 0.7
}
```

## Supported Languages

Comprehensive list of supported languages with their abbreviations and Chinese names.

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

## Training Data Format

Guidelines for preparing data for fine-tuning, including an example of data formatting.

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

Comprehensive instructions for fine-tuning the Hunyuan model using LLaMA-Factory, including:

*   Prerequisites (LLaMA-Factory, DeepSpeed, Transformers).
*   Data preparation (data format and dataset definition).
*   Training execution (single-node and multi-node).

## Quantization and Compression

Information on how to use the AngelSlim tool for FP8 and INT4 quantization, including links to quantized models.

### FP8 Quantization

Details on using FP8-static quantization with AngelSlim, to improve inference efficiency.

## Deployment

Instructions for deploying Hunyuan-MT using different frameworks:

### TensorRT-LLM

*   Docker image setup instructions.
*   Configuration file examples.
*   API server startup commands.

```bash
docker pull docker.cnb.cool/tencent/hunyuan/hunyuan-7b:hunyuan-7b-trtllm
docker run --privileged --user root --name hunyuanLLM_infer --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all hunyuaninfer/hunyuan-7b:hunyuan-7b-trtllm
cat >/path/to/extra-llm-api-config.yml <<EOF
use_cuda_graph: true
cuda_graph_padding_enabled: true
cuda_graph_batch_sizes:
- 1
- 2
- 4
- 8
- 16
- 32
print_iter_log: true
EOF
trtllm-serve \
  /path/to/HunYuan-7b \
  --host localhost \
  --port 8000 \
  --backend pytorch \
  --max_batch_size 32 \
  --max_num_tokens 16384 \
  --tp_size 2 \
  --kv_cache_free_gpu_memory_fraction 0.6 \
  --trust_remote_code \
  --extra_llm_api_options /path/to/extra-llm-api-config.yml
```

### vLLM

*   Instructions to set up the vLLM environment.
*   Model download steps from Hugging Face or ModelScope.
*   API server start command.
*   Example curl command for testing the API.
```bash
pip install git+https://github.com/huggingface/transformers@4970b23cedaf745f963779b4eae68da281e8c6ca
export MODEL_PATH=tencent/Hunyuan-MT-7B-Instruct
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --model ${MODEL_PATH} \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --quantization experts_int8 \
    --served-model-name hunyuan \
    2>&1 | tee log_server.txt
curl http://0.0.0.0:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
"model": "hunyuan",
"messages": [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "请按面积大小对四大洋进行排序，并给出面积最小的洋是哪一个？直接输出结果。"}]
    }
],
"max_tokens": 2048,
"temperature":0.7,
"top_p": 0.6,
"top_k": 20,
"repetition_penalty": 1.05,
"stop_token_ids": [127960]
}'
```

#### Quantitative Model Deployment in vLLM

Provides instructions for deploying int8, int4 and FP8 quantized versions of the model using vLLM.

##### Int8
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --model ${MODEL_PATH} \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --served-model-name hunyuan \
    --quantization experts_int8 \
    2>&1 | tee log_server.txt
```

##### Int4
```bash
export MODEL_PATH=PATH_TO_INT4_MODEL
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --model ${MODEL_PATH} \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --served-model-name hunyuan \
    --quantization gptq_marlin \
    2>&1 | tee log_server.txt
```

##### FP8
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --model ${MODEL_PATH} \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --served-model-name hunyuan \
    --kv-cache-dtype fp8 \
    2>&1 | tee log_server.txt
```

### SGLang

*   Docker image pull and run commands.
*   API server startup command.
```bash
docker pull lmsysorg/sglang:latest
docker run --entrypoint="python3" --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    --ulimit nproc=10000 \
    --privileged \
    --ipc=host \
     lmsysorg/sglang:latest \
    -m sglang.launch_server --model-path hunyuan/huanyuan_7B --tp 4 --trust-remote-code --host 0.0.0.0 --port 30000
```

## Citation

Provides the BibTeX citation for referencing the Hunyuan-MT technical report.

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

Contact information for the R&D and product teams, including an email address for inquiries.
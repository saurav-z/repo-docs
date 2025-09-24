# Hunyuan-MT: State-of-the-Art Translation Models

**Hunyuan-MT delivers high-quality, open-source translation across 33 languages, pushing the boundaries of multilingual communication.** ([Original Repository](https://github.com/Tencent-Hunyuan/Hunyuan-MT))

<p align="center">
 <img src="https://dscache.tencent-cloud.cn/upload/uploader/hunyuan-64b418fd052c033b228e04bc77bbc4b54fd7f5bc.png" width="400"/> <br>
</p>

**Key Features:**

*   🏆 **Industry-Leading Performance:** Achieved first place in 30 out of 31 language pairs in the WMT25 competition.
*   🧠 **High-Quality Translation:**  Hunyuan-MT-Chimera, an open-source ensemble model, significantly improves translation quality.
*   🌐 **Multilingual Support:** Supports mutual translation among 33 languages, including several Chinese ethnic minority languages.
*   🛠️ **Comprehensive Framework:** Trained using a cutting-edge framework, spanning pretraining, continual pretraining, supervised fine-tuning, and reinforcement learning.
*   ⚡ **Quantization Options:** Offers FP8 and INT4 quantized models for efficient deployment using AngelSlim.

**Key Resources:**

*   🤗 [Hugging Face](https://huggingface.co/collections/tencent/hunyuan-mt-68b42f76d473f82798882597)
*   💻 [ModelScope](https://modelscope.cn/collections/Hunyuan-MT-2ca6b8e1b4934f)
*   🖥️ [Official Website](https://hunyuan.tencent.com)
*   🕹️ [Demo](https://hunyuan.tencent.com/chat/HunyuanDefault?from=modelSquare&modelId=hunyuan-mt-7b)
*   📄 [Technical Report](https://www.arxiv.org/pdf/2509.05209)

## Model Overview

Hunyuan-MT features both the Hunyuan-MT-7B translation model and the ensemble model, Hunyuan-MT-Chimera. The 7B model focuses on direct translation, while Chimera combines outputs for superior results.

## Model Performance

<div align='center'>
<img src="imgs/overall_performance.png" width = "80%" />
</div>

See the technical report for more experimental results.

## Available Models

| Model Name                   | Description                                       | Download                                                                |
| :--------------------------- | :------------------------------------------------ | :---------------------------------------------------------------------- |
| Hunyuan-MT-7B               | Hunyuan 7B translation model                      | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B)                   |
| Hunyuan-MT-7B-fp8           | Hunyuan 7B translation model, FP8 quantization    | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B-fp8)                |
| Hunyuan-MT-Chimera          | Hunyuan 7B translation ensemble model           | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B)             |
| Hunyuan-MT-Chimera-fp8      | Hunyuan 7B translation ensemble model, FP8 quant | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B-fp8)          |

## Prompting Guidelines

### ZH <-> XX Translation

```
把下面的文本翻译成<target_language>，不要额外解释。

<source_text>
```

### XX <-> XX Translation (Excluding ZH)

```
Translate the following segment into <target_language>, without additional explanation.

<source_text>
```

### Hunyuan-MT-Chimera-7B Prompt Template

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

Install the transformers library:
```bash
pip install transformers==4.56.0
```

**Example Code:**
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

| Language              | Abbr.   | Chinese Name  |
| :-------------------- | :------ | :------------ |
| Chinese               | zh      | 中文          |
| English               | en      | 英语          |
| French                | fr      | 法语          |
| Portuguese            | pt      | 葡萄牙语      |
| Spanish               | es      | 西班牙语      |
| Japanese              | ja      | 日语          |
| Turkish               | tr      | 土耳其语      |
| Russian               | ru      | 俄语          |
| Arabic                | ar      | 阿拉伯语      |
| Korean                | ko      | 韩语          |
| Thai                  | th      | 泰语          |
| Italian               | it      | 意大利语      |
| German                | de      | 德语          |
| Vietnamese            | vi      | 越南语        |
| Malay                 | ms      | 马来语        |
| Indonesian            | id      | 印尼语        |
| Filipino              | tl      | 菲律宾语      |
| Hindi                 | hi      | 印地语        |
| Traditional Chinese   | zh-Hant | 繁体中文      |
| Polish                | pl      | 波兰语        |
| Czech                 | cs      | 捷克语        |
| Dutch                 | nl      | 荷兰语        |
| Khmer                 | km      | 高棉语        |
| Burmese               | my      | 缅甸语        |
| Persian               | fa      | 波斯语        |
| Gujarati              | gu      | 古吉拉特语    |
| Urdu                  | ur      | 乌尔都语      |
| Telugu                | te      | 泰卢固语      |
| Marathi               | mr      | 马拉地语      |
| Hebrew                | he      | 希伯来语      |
| Bengali               | bn      | 孟加拉语      |
| Tamil                 | ta      | 泰米尔语      |
| Ukrainian             | uk      | 乌克兰语      |
| Tibetan               | bo      | 藏语          |
| Kazakh                | kk      | 哈萨克语      |
| Mongolian             | mn      | 蒙古语        |
| Uyghur                | ug      | 维吾尔语      |
| Cantonese             | yue     | 粤语          |

## Fine-tuning

### Training Data Format

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

#### Prerequisites

*   LLaMA-Factory ([installation guide](https://github.com/hiyouga/LLaMA-Factory))
*   DeepSpeed (optional, [installation guide](https://github.com/deepspeedai/DeepSpeed#installation))
*   Transformers (companion branch):  `pip install git+https://github.com/huggingface/transformers@4970b23cedaf745f963779b4eae68da281e8c6ca`

#### Data Preparation

1.  **Dataset Format:** Organize your data in `json` format in the `data` directory in `LLaMA-Factory`. Use the `sharegpt` format:
    ```json
    [
      {
        "messages": [
          {
            "role": "system",
            "content": "System prompt (optional)"
          },
          {
            "role": "user",
            "content": "Human instruction"
          },
          {
            "role": "assistant",
            "content": "Model response"
          }
        ]
      }
    ]
    ```
2.  **Dataset Definition (dataset_info.json):**
    ```json
    "dataset_name": {
      "file_name": "dataset.json",
      "formatting": "sharegpt",
      "columns": {
        "messages": "messages"
      },
      "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system"
      }
    }
    ```

#### Training Execution

1.  Copy configuration files from `llama_factory_support/example_configs` to `example/hunyuan` within `LLaMA-Factory`.
2.  Modify the model path and dataset name in the configuration file `hunyuan_full.yaml`.
    ```yaml
    ### model
    model_name_or_path: [!!!add the model path here!!!]

    ### dataset
    dataset: [!!!add the dataset name here!!!]
    ```
3.  **Training Commands:**
    *   **Single-node:**
        ```bash
        export DISABLE_VERSION_CHECK=1
        llamafactory-cli train examples/hunyuan/hunyuan_full.yaml
        ```
    *   **Multi-node:**
        ```bash
        export DISABLE_VERSION_CHECK=1
        FORCE_TORCHRUN=1 NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \
        llamafactory-cli train examples/hunyuan/hunyuan_full.yaml
        ```

## Quantization and Compression

Utilizing the AngelSlim tool for FP8 and INT4 model compression.

### FP8 Quantization

FP8 static quantization converts weights and activations to an 8-bit floating-point format. Use our quantized models from [AngelSlim](https://huggingface.co/AngelSlim).

## Deployment Options

### TensorRT-LLM

#### Docker Image

Pre-built Docker images based on the latest TensorRT-LLM are available.

```bash
docker pull docker.cnb.cool/tencent/hunyuan/hunyuan-7b:hunyuan-7b-trtllm
```

```bash
docker run --privileged --user root --name hunyuanLLM_infer --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all hunyuaninfer/hunyuan-7b:hunyuan-7b-trtllm
```

Prepare Configuration File:

```bash
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
```

Start the API Server:

```bash
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

#### Prerequisites

```bash
pip install git+https://github.com/huggingface/transformers@4970b23cedaf745f963779b4eae68da281e8c6ca
```

Choose your model path:
```bash
export MODEL_PATH=tencent/Hunyuan-7B-Instruct
```

OR, if using ModelScope (downloaded locally):
```bash
export MODEL_PATH=/root/.cache/modelscope/hub/models/Tencent-Hunyuan/Hunyuan-7B-Instruct/
```

Start the API Server:

```bash
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
```

Test the API:
```bash
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

#### Quantized Model Deployment

Deploying quantized models with vLLM:

##### Int8 (weight-only)
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

##### Int4 (weight-only)
```bash
export MODEL_PATH=PATH_TO_INT4_MODEL # Set this
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

##### FP8 (W8A8C8)
```bash
export MODEL_PATH=PATH_TO_FP8_MODEL  # Set this
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

#### Docker Image

```bash
docker pull lmsysorg/sglang:latest
```

Start API Server:

```bash
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

For inquiries, contact the open-source team or email hunyuan_opensource@tencent.com.
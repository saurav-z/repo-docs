# Hunyuan-MT: State-of-the-Art Multilingual Translation Models

**Hunyuan-MT offers high-quality, open-source multilingual translation, achieving top performance in numerous language pairs. [See the original repository](https://github.com/Tencent-Hunyuan/Hunyuan-MT).**

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

*   **Exceptional Performance:** Achieved first place in 30 out of 31 language categories in the WMT25 competition.
*   **Leading-Edge Models:** Includes Hunyuan-MT-7B, a high-performing translation model, and Hunyuan-MT-Chimera-7B, the first open-source ensemble translation model.
*   **Comprehensive Training Framework:**  Utilizes a complete training pipeline from pretraining to ensemble RL, achieving state-of-the-art results.
*   **Supports 33 Languages:** Offers mutual translation across a broad range of languages, including five Chinese ethnic minority languages.

## Model Overview

Hunyuan-MT provides both a standalone translation model (Hunyuan-MT-7B) and an ensemble model (Hunyuan-MT-Chimera) to deliver superior translation quality.

## Performance

<div align='center'>
<img src="imgs/overall_performance.png" width = "80%" />
</div>
For detailed performance analysis and results, please refer to the [Technical Report](https://www.arxiv.org/pdf/2509.05209).

## Model Links

| Model Name                 | Description                                | Download                                                               |
| -------------------------- | ------------------------------------------ | ---------------------------------------------------------------------- |
| Hunyuan-MT-7B              | Hunyuan 7B translation model              | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B)                 |
| Hunyuan-MT-7B-fp8          | Hunyuan 7B translation model, FP8 quantized | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B-fp8)             |
| Hunyuan-MT-Chimera         | Hunyuan 7B translation ensemble model      | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B)        |
| Hunyuan-MT-Chimera-fp8     | Hunyuan 7B translation ensemble model, FP8 quantized      | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B-fp8)   |

## Quickstart - Prompts

### Prompt Template for ZH<=>XX Translation

```
把下面的文本翻译成<target_language>，不要额外解释。

<source_text>
```

### Prompt Template for XX<=>XX Translation (excluding ZH<=>XX)

```
Translate the following segment into <target_language>, without additional explanation.

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

## How to Use with Transformers

Install Transformers (version 4.56.0 recommended):

```shell
pip install transformers==4.56.0
```

*   *Important*:  To load FP8 models, change "ignored\_layers" to "ignore" in the `config.json` and upgrade `compressed-tensors` to `compressed-tensors-0.11.0`.

Example code using `tencent/Hunyuan-MT-7B`:

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

Recommended inference parameters:

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

## Training Data Format

If you want to fine-tune the Instruct model, format your data as follows:

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

### Prerequisites

*   **LLaMA-Factory:**  Install as per the [official guide](https://github.com/hiyouga/LLaMA-Factory).
*   **DeepSpeed** (optional): Install as per the [official guide](https://github.com/deepspeedai/DeepSpeed#installation).
*   **Transformers Library:**  Use the companion branch  (`pip install git+https://github.com/huggingface/transformers@4970b23cedaf745f963779b4eae68da281e8c6ca`).

### Data Preparation

1.  Create a `json` file in the `data` directory of  `LLaMA-Factory` with the following structure (using `sharegpt` format):

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
    Refer to [Training Data Format](#training-data-format) section for details.

2.  Define your dataset in the `data/dataset_info.json` file using the following format:

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

### Training Execution

1.  Copy all files from the `llama_factory_support/example_configs` directory to the `example/hunyuan` directory in `LLaMA-Factory`.
2.  Modify the model path and dataset name in the configuration file `hunyuan_full.yaml`. Adjust other configurations as needed:

    ```yaml
    ### model
    model_name_or_path: [!!!add the model path here!!!]

    ### dataset
    dataset: [!!!add the dataset name here!!!]
    ```

3.  Execute training commands:

    *   **Single-node training:**

        ```bash
        export DISABLE_VERSION_CHECK=1
        llamafactory-cli train examples/hunyuan/hunyuan_full.yaml
        ```

    *   **Multi-node training:** (Configure `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` according to your environment)

        ```bash
        export DISABLE_VERSION_CHECK=1
        FORCE_TORCHRUN=1 NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \
        llamafactory-cli train examples/hunyuan/hunyuan_full.yaml
        ```

## Quantization and Compression

The project leverages the [AngelSlim](https://github.com/tencent/AngelSlim) compression tool for FP8 and INT4 quantization.

### FP8 Quantization

*   FP8 static quantization is used. Weights and activations are converted to FP8 format using calibration data (no training required). This improves inference efficiency.

## Deployment

Deploy Hunyuan-MT using frameworks like **TensorRT-LLM**, **vLLM**, or **SGLang**.

### TensorRT-LLM

#### Docker Image

Pre-built Docker images are available. For example, for tencent/Hunyuan-7B-Instruct:

*   Pull the image:

    ```bash
    docker pull docker.cnb.cool/tencent/hunyuan/hunyuan-7b:hunyuan-7b-trtllm
    ```

*   Run the container:

    ```bash
    docker run --privileged --user root --name hunyuanLLM_infer --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all hunyuaninfer/hunyuan-7b:hunyuan-7b-trtllm
    ```

*   Prepare a configuration file:

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

*   Start the API server:

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

#### Installation

Install Transformers:
```shell
pip install git+https://github.com/huggingface/transformers@4970b23cedaf745f963779b4eae68da281e8c6ca
```

Example using tencent/Hunyuan-7B-Instruct:

*   Download the model (Hugging Face or ModelScope):

    *   Hugging Face (downloads automatically):
        ```bash
        export MODEL_PATH=tencent/Hunyuan-7B-Instruct
        ```
    *   ModelScope:
        ```bash
        export MODEL_PATH=/root/.cache/modelscope/hub/models/Tencent-Hunyuan/Hunyuan-7B-Instruct/
        ```

*   Start the API server:

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

*   Test the API:

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

*   **Int8 (Weight-Only):**

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

*   **Int4 (GPTQ):**

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

*   **FP8 (W8A8C8):**

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

#### Docker Image

*   Pull the Docker image:

    ```bash
    docker pull lmsysorg/sglang:latest
    ```

*   Run the API server (example with tencent/Hunyuan-7B-Instruct):

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

For questions, feedback, or collaboration, contact our open-source team at [hunyuan_opensource@tencent.com].
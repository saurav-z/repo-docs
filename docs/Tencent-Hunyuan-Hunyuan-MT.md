# Hunyuan-MT: State-of-the-Art Machine Translation Models

**Harness the power of Hunyuan-MT, Tencent's cutting-edge machine translation models, delivering superior translation quality across 33 languages, including Chinese minority languages.** [View the original repository](https://github.com/Tencent-Hunyuan/Hunyuan-MT)

**Key Features:**

*   **Multilingual Support:** Excels in translating between 33 languages, including Chinese, English, and five Chinese minority languages.
*   **Industry-Leading Performance:** Achieves top-tier results, including first place in 30 out of 31 language categories in the WMT25 competition.
*   **Open-Source Ensemble Model:** Features Hunyuan-MT-Chimera, the industry's first open-source ensemble translation model, for enhanced accuracy.
*   **Advanced Training Framework:** Employs a comprehensive training pipeline (pretrain -> CPT -> SFT -> translation RL -> ensemble RL) for SOTA performance.
*   **Quantization Support:** Offers FP8 and INT4 quantized models for efficient deployment and reduced resource usage.

## Quick Links

*   🤗 [Hugging Face](https://huggingface.co/collections/tencent/hunyuan-mt-68b42f76d473f82798882597)
*   🖥️ [Official Website](https://hunyuan.tencent.com)
*   🕹️ [Demo](https://hunyuan.tencent.com/chat/HunyuanDefault?from=modelSquare&modelId=hunyuan-mt-7b)
*   <img src="https://avatars.githubusercontent.com/u/109945100?s=200&v=4" width="16"/> [ModelScope](https://modelscope.cn/collections/Hunyuan-MT-2ca6b8e1b4934f)

## Model Details

Hunyuan-MT includes two primary models:

*   **Hunyuan-MT-7B:** A translation model.
*   **Hunyuan-MT-Chimera:** An ensemble model designed to improve translation quality.

## Performance Highlights

*   **WMT25 Competition:** Achieved first place in 30 out of 31 language pairs.

<div align='center'>
<img src="imgs/overall_performance.png" width = "80%" />
</div>
  [Read the Technical Report](Hunyuan_MT_Technical_Report.pdf) for details on performance results and evaluations.

## Model Availability

| Model Name                 | Description                                     | Download                                                                   |
| :------------------------- | :---------------------------------------------- | :------------------------------------------------------------------------- |
| Hunyuan-MT-7B              | Hunyuan 7B translation model                   | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B)                 |
| Hunyuan-MT-7B-fp8          | Hunyuan 7B translation model, FP8 quantized     | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B-fp8)             |
| Hunyuan-MT-Chimera         | Hunyuan 7B translation ensemble model           | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B)        |
| Hunyuan-MT-Chimera-fp8     | Hunyuan 7B translation ensemble model, FP8 quant | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B-fp8)     |

## Prompt Templates

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

## Usage

### With Transformers

1.  **Install Transformers:**

```shell
pip install transformers==4.56.0
```

*   **Important:** To load FP8 models, modify "ignored\_layers" to "ignore" in `config.json` and upgrade `compressed-tensors` to `compressed-tensors-0.11.0`.

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

## Supported Languages

| Languages           | Abbr.   | Chinese Names   |
| :------------------ | :------ | :-------------- |
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
| Thai              | th      | 泰语            |
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

## Training Data Format

If you want to fine-tune our Instruct model, we recommend processing the data into the following format.

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

## Training with LLaMA-Factory

Follow these steps to fine-tune the Hunyuan model using LLaMA-Factory:

### Prerequisites

*   **LLaMA-Factory:**  Follow the [official installation guide](https://github.com/hiyouga/LLaMA-Factory).
*   **DeepSpeed** (Optional): Follow the [official installation guide](https://github.com/deepspeedai/DeepSpeed#installation).
*   **Transformers:** Use the companion branch.

```shell
pip install git+https://github.com/huggingface/transformers@4970b23cedaf745f963779b4eae68da281e8c6ca
```

### Data Preparation

1.  **Data Format:** Prepare your data in `json` format using the `sharegpt` format in the `data` directory in `LLaMA-Factory`:

```
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

2.  **Dataset Definition:** Define your dataset in the `data/dataset_info.json` file:

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

1.  **Configuration:** Copy files from `llama_factory_support/example_configs` to the `example/hunyuan` directory in `LLaMA-Factory`.
2.  **Modify Configuration:**  Edit the `hunyuan_full.yaml` configuration file with your model path and dataset name. Adjust other settings as needed.

```yaml
### model
model_name_or_path: [!!!add the model path here!!!]

### dataset
dataset: [!!!add the dataset name here!!!]
```

3.  **Training Commands:**
    *   **Single-Node:**

```shell
export DISABLE_VERSION_CHECK=1
llamafactory-cli train examples/hunyuan/hunyuan_full.yaml
```

    *   **Multi-Node:**  Configure `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` according to your environment.

```shell
export DISABLE_VERSION_CHECK=1
FORCE_TORCHRUN=1 NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \
llamafactory-cli train examples/hunyuan/hunyuan_full.yaml
```

## Quantization Compression

Hunyuan-MT offers FP8 and INT4 quantized models using the AngelSlim tool for efficient model compression.

### FP8 Quantization

FP8 static quantization is used. Calibration data is used to pre-determine the quantization scale.
You can download and utilize these directly from [AngelSlim](https://huggingface.co/AngelSlim).

## Deployment Options

Deploy your models using frameworks such as **TensorRT-LLM**, **vLLM**, or **SGLang**.

### TensorRT-LLM

#### Docker Image

Pre-built Docker images are available.
*Example: hunyuaninfer/hunyuan-7B:hunyuan-7b-trtllm*

1.  **Pull the Image:**

```shell
docker pull docker.cnb.cool/tencent/hunyuan/hunyuan-7b:hunyuan-7b-trtllm
```

2.  **Run the Container:**

```shell
docker run --privileged --user root --name hunyuanLLM_infer --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all hunyuaninfer/hunyuan-7B:hunyuan-7b-trtllm
```

3.  **Prepare Configuration:**

```shell
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

4.  **Start the API Server:**

```shell
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

```shell
pip install git+https://github.com/huggingface/transformers@4970b23cedaf745f963779b4eae68da281e8c6ca
```

#### Model Download

*   **Hugging Face:**  The model will download automatically.
*   **ModelScope:** `modelscope download --model Tencent-Hunyuan/Hunyuan-7B-Instruct`

#### Environment variables

```shell
export MODEL_PATH=tencent/Hunyuan-7B-Instruct   # Or  /root/.cache/modelscope/hub/models/Tencent-Hunyuan/Hunyuan-7B-Instruct/
```

#### Start API Server

```shell
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

#### Run Request Script

```shell
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

#### Quantization Model Deployment

*   **Int8 Deployment:**

```shell
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

*   **Int4 Deployment:**

```shell
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

*   **FP8 Deployment:**

```shell
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

*   Pull the Docker Image

```shell
docker pull lmsysorg/sglang:latest
```

*   Start API server

```shell
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

## Contact Us

For inquiries or feedback, please contact the open-source team at  `hunyuan_opensource@tencent.com`.
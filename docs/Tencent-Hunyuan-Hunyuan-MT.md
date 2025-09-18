# Hunyuan-MT: State-of-the-Art Machine Translation Models

Hunyuan-MT is a powerful suite of machine translation models from Tencent, achieving top performance in 30 out of 31 WMT25 language categories. [Visit the original repository](https://github.com/Tencent-Hunyuan/Hunyuan-MT) for more details.

**Key Features:**

*   **Leading Performance:** Achieved first place in most WMT25 language categories.
*   **Open-Source Ensemble Model:** Includes Hunyuan-MT-Chimera, the industry's first open-source ensemble model for improved translation quality.
*   **Multi-Language Support:** Supports mutual translation among 33 languages, including Chinese and five Chinese ethnic minority languages.
*   **Comprehensive Training Framework:** Utilizes a cutting-edge training pipeline from pretraining to ensemble RL.
*   **FP8 and INT4 Quantization Support:** Offers models optimized for faster inference and reduced resource usage.

**Quick Links:**

*   🤗 [Hugging Face](https://huggingface.co/collections/tencent/hunyuan-mt-68b42f76d473f82798882597)
*   🖥️ [Official Website](https://hunyuan.tencent.com)
*   🕹️ [Demo](https://hunyuan.tencent.com/chat/HunyuanDefault?from=modelSquare&modelId=hunyuan-mt-7b)
*   📑 [Technical Report](https://www.arxiv.org/pdf/2509.05209)

## Model Overview

Hunyuan-MT comprises two primary model types:

*   **Hunyuan-MT-7B:**  A 7-billion parameter translation model.
*   **Hunyuan-MT-Chimera:** An ensemble model built to improve translation quality.

## Performance Highlights

[Include a brief, eye-catching summary of performance. For example, "Hunyuan-MT-7B achieves industry-leading performance among models of comparable scale."]

<div align='center'>
<img src="imgs/overall_performance.png" width = "80%" />
</div>

For detailed experimental results and analysis, please refer to the [Technical Report](https://www.arxiv.org/pdf/2509.05209).

## Model Links

| Model Name                      | Description                                          | Download                                                              |
| ------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------- |
| Hunyuan-MT-7B                   | Hunyuan 7B translation model                         | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B)               |
| Hunyuan-MT-7B-fp8               | Hunyuan 7B translation model, FP8 quantized         | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B-fp8)           |
| Hunyuan-MT-Chimera              | Hunyuan 7B translation ensemble model                | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B)      |
| Hunyuan-MT-Chimera-fp8          | Hunyuan 7B translation ensemble model, FP8 quantized | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B-fp8)  |

## Prompts

### Prompt Templates

*   **ZH<=>XX Translation:**
    ```
    把下面的文本翻译成<target_language>，不要额外解释。

    <source_text>
    ```
*   **XX<=>XX Translation (excluding ZH):**
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

## Usage

### Installation

```shell
pip install transformers==4.56.0
```
*!!! If you want to load fp8 model with transformers, you need to change the name"ignored_layers" in config.json to "ignore" and upgrade the compressed-tensors to compressed-tensors-0.11.0.*

### Example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

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

### Inference Parameters (Recommended):

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

If you need to fine-tune our Instruct model, we recommend processing the data into the following format.

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

[Instructions and examples to use LLaMA-Factory for Fine-tuning]

## Quantization & Compression

### FP8 Quantization

Hunyuan-MT offers models quantized to FP8, and AngelSlim compression tool can be used.

## Deployment

[Deployment instructions and options, including TensorRT-LLM, vLLM, and SGLang.]

### TensorRT-LLM

#### Docker Image

We provide a pre-built Docker image based on the latest version of TensorRT-LLM.

```
docker pull docker.cnb.cool/tencent/hunyuan/hunyuan-7b:hunyuan-7b-trtllm
```
```
docker run --privileged --user root --name hunyuanLLM_infer --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all hunyuaninfer/hunyuan-7b:hunyuan-7b-trtllm
```

- Prepare Configuration file:

```
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

- Start the API server:

```
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

### vllm

#### Start

Please use vLLM version v0.10.0 or higher for inference.

First, please install transformers. We will merge it into the main branch later.
```SHELL
pip install git+https://github.com/huggingface/transformers@4970b23cedaf745f963779b4eae68da281e8c6ca
```

We use tencent/Hunyuan-7B-Instruct for example
- Download Model file:
  - Huggingface:  will download automicly by vllm.
  - ModelScope: `modelscope download --model Tencent-Hunyuan/Hunyuan-7B-Instruct`

- model download by huggingface:
```shell
export MODEL_PATH=tencent/Hunyuan-7B-Instruct
```

- model downloaded by modelscope:
```shell
export MODEL_PATH=/root/.cache/modelscope/hub/models/Tencent-Hunyuan/Hunyuan-7B-Instruct/
```

- Start the API server:

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
- After running service script successfully, run the request script
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
#### Quantitative model deployment
This section describes the process of deploying a post-quantization model using vLLM.

Default server in BF16.

##### Int8 quantitative model deployment
Deploying the Int8-weight-only version of the HunYuan-7B model only requires setting the environment variables

Next we start the Int8 service. Run:
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


##### Int4 quantitative model deployment
Deploying the Int4-weight-only version of the HunYuan-7B model only requires setting the environment variables , using the GPTQ method
```shell
export MODEL_PATH=PATH_TO_INT4_MODEL
```
Next we start the Int4 service. Run
```shell
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

##### FP8 quantitative model deployment
Deploying the W8A8C8 version of the HunYuan-7B model only requires setting the environment variables


Next we start the FP8 service. Run
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

We also provide a pre-built Docker image based on the latest version of SGLang.

We use tencent/Hunyuan-7B-Instruct for example

To get started:

- Pull the Docker image

```
docker pull lmsysorg/sglang:latest
```

- Start the API server:

```
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

For inquiries, contact our open-source team or send an email to hunyuan_opensource@tencent.com.
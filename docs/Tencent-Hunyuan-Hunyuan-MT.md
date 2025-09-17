# Hunyuan-MT: High-Performance Multilingual Translation Models

Unlock the power of AI-driven translation with Hunyuan-MT, a suite of state-of-the-art translation models from Tencent, achieving top rankings and offering exceptional performance. **[View the Original Repository](https://github.com/Tencent-Hunyuan/Hunyuan-MT)**

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

*   **Industry-Leading Performance:** Achieved 1st place in 30 out of 31 language categories in the WMT25 competition.
*   **Open-Source Ensemble Model:** Hunyuan-MT-Chimera-7B is the first open-source translation ensemble model.
*   **Comprehensive Training Framework:** Utilizes a cutting-edge training pipeline (Pretrain -> CPT -> SFT -> Translation RL -> Ensemble RL) for SOTA results.
*   **Wide Language Support:** Supports mutual translation among 33 languages, including several Chinese ethnic minority languages.
*   **Model Variants:** Offers a variety of models including 7B and fp8 quantized versions for diverse applications.

## Models

| Model Name                        | Description                                   | Download                                                                  |
| :-------------------------------- | :-------------------------------------------- | :------------------------------------------------------------------------ |
| Hunyuan-MT-7B                     | Hunyuan 7B translation model                  | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B)                     |
| Hunyuan-MT-7B-fp8                 | Hunyuan 7B translation model, fp8 quantized | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-7B-fp8)                |
| Hunyuan-MT-Chimera                | Hunyuan 7B translation ensemble model           | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B)           |
| Hunyuan-MT-Chimera-fp8            | Hunyuan 7B translation ensemble model, fp8 quantized | 🤗 [Model](https://huggingface.co/tencent/Hunyuan-MT-Chimera-7B-fp8)      |

## Performance

<div align='center'>
<img src="imgs/overall_performance.png" width = "80%" />
</div>
For detailed experimental results and analysis, please refer to the [Technical Report](https://www.arxiv.org/pdf/2509.05209).

## Quickstart

### Prompt Templates

Utilize these templates for optimal translation results:

**ZH<=>XX Translation:**

```
把下面的文本翻译成<target_language>，不要额外解释。

<source_text>
```

**XX<=>XX Translation (excluding ZH<=>XX):**

```
Translate the following segment to <target_language>, without additional explanation.

<source_text>
```

**Hunyuan-MT-Chimera-7B:**

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

### Installation and Usage with Transformers

1.  **Install Transformers:**

    ```bash
    pip install transformers==4.56.0
    ```
    *!!! If you want to load fp8 model with transformers, you need to change the name"ignored_layers" in config.json to "ignore" and upgrade the compressed-tensors to compressed-tensors-0.11.0.*

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
| Hebrew          | he      | 希伯来语        |
| Bengali         | bn      | 孟加拉语        |
| Tamil         | ta      | 泰米尔语        |
| Ukrainian         | uk      | 乌克兰语        |
| Tibetan         | bo      | 藏语            |
| Kazakh         | kk      | 哈萨克语        |
| Mongolian         | mn      | 蒙古语          |
| Uyghur         | ug      | 维吾尔语        |
| Cantonese         | yue     | 粤语            |

## Training

### Training Data Format

To fine-tune the model, format your data as follows:

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

1.  **Prerequisites:**
    *   LLaMA-Factory: Follow the [official installation guide](https://github.com/hiyouga/LLaMA-Factory)
    *   DeepSpeed (optional): Follow the [official installation guide](https://github.com/deepspeedai/DeepSpeed#installation)
    *   Transformers: Use the companion branch (Hunyuan-submitted code is pending review)
    ```bash
    pip install git+https://github.com/huggingface/transformers@4970b23cedaf745f963779b4eae68da281e8c6ca
    ```

2.  **Data Preparation:**
    *   Organize your data in `json` format within the `data` directory. The current implementation uses the `sharegpt` dataset format.
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
    Refer to the [Training Data Format](#training-data-format) section.
    *   Define your dataset in the `data/dataset_info.json` file:
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

3.  **Training Execution:**
    *   Copy files from the `llama_factory_support/example_configs` directory to the `example/hunyuan` directory.
    *   Modify the model path and dataset name in the configuration file `hunyuan_full.yaml`.
    ```yaml
    ### model
    model_name_or_path: [!!!add the model path here!!!]

    ### dataset
    dataset: [!!!add the dataset name here!!!]
    ```
    *   Run the following commands (single-node and multi-node instructions provided in original README).

## Quantization and Compression

Hunyuan-MT models are available in both FP8 and INT4 quantized versions using [AngelSlim](https://github.com/tencent/AngelSlim).

### FP8 Quantization

*   FP8-static quantization converts weights and activations to an 8-bit floating-point format.

## Deployment

Deploy Hunyuan-MT models using frameworks like TensorRT-LLM, vLLM, or SGLang.

### TensorRT-LLM

1.  **Docker Image:**
    ```bash
    docker pull docker.cnb.cool/tencent/hunyuan/hunyuan-7b:hunyuan-7b-trtllm
    ```
    ```bash
    docker run --privileged --user root --name hunyuanLLM_infer --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all hunyuaninfer/hunyuan-7b:hunyuan-7b-trtllm
    ```

2.  **Configuration:**

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

3.  **Start API Server:**

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

1.  **Install:**
    ```bash
    pip install git+https://github.com/huggingface/transformers@4970b23cedaf745f963779b4eae68da281e8c6ca
    ```

2.  **Model Path:**
    ```bash
    # From Hugging Face
    export MODEL_PATH=tencent/Hunyuan-7B-Instruct

    # From ModelScope
    export MODEL_PATH=/root/.cache/modelscope/hub/models/Tencent-Hunyuan/Hunyuan-7B-Instruct/
    ```

3.  **Start API Server:**

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

4.  **Request Example:**

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
5.  **Quantized Model Deployment (vLLM):**
    *   INT8: `quantization experts_int8`
    *   INT4: `quantization gptq_marlin`, set `export MODEL_PATH=PATH_TO_INT4_MODEL`
    *   FP8:  `--kv-cache-dtype fp8`

### SGLang

1.  **Docker Image:**
    ```bash
    docker pull lmsysorg/sglang:latest
    ```
2.  **Start API Server:**
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
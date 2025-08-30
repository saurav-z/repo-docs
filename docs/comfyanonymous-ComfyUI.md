<div>
  <div align="center">
    <h1>ComfyUI: Unleash Your AI Creativity with a Powerful Visual Workflow Engine</h1>
    <p><b>ComfyUI offers a revolutionary node-based interface for designing and executing advanced Stable Diffusion pipelines, making AI image, video, and audio generation more accessible and customizable.</b></p>
    <!-- Badges -->
    <p>
        <a href="https://www.comfy.org/" target="_blank"><img src="https://img.shields.io/badge/ComfyOrg-4285F4?style=flat" alt="Website"></a>
        <a href="https://discord.com/invite/comfyorg" target="_blank"><img src="https://img.shields.io/badge/Discord-Join-7289DA?style=flat&logo=discord&logoColor=white" alt="Discord"></a>
        <a href="https://x.com/ComfyUI" target="_blank"><img src="https://img.shields.io/twitter/follow/ComfyUI?style=flat&logo=x&logoColor=white" alt="Twitter"></a>
        <a href="https://app.element.io/#/room/%23comfyui_space%3Amatrix.org" target="_blank"><img src="https://img.shields.io/badge/Matrix-Join-000000?style=flat&logo=matrix&logoColor=white" alt="Matrix"></a>
        <br/>
        <a href="https://github.com/comfyanonymous/ComfyUI/releases" target="_blank"><img src="https://img.shields.io/github/release/comfyanonymous/ComfyUI?style=flat&sort=semver" alt="GitHub Release"></a>
        <a href="https://github.com/comfyanonymous/ComfyUI/releases" target="_blank"><img src="https://img.shields.io/github/release-date/comfyanonymous/ComfyUI?style=flat" alt="Release Date"></a>
        <a href="https://github.com/comfyanonymous/ComfyUI/releases" target="_blank"><img src="https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/total?style=flat" alt="Total Downloads"></a>
        <a href="https://github.com/comfyanonymous/ComfyUI/releases" target="_blank"><img src="https://img.shields.io/github/downloads/comfyanonymous/ComfyUI/latest/total?style=flat&label=downloads%40latest" alt="Latest Downloads"></a>
    </p>
    <!-- Screenshot (Optional) -->
    <img src="https://github.com/user-attachments/assets/7ccaf2c1-9b72-41ae-9a89-5688c94b7abe" alt="ComfyUI Screenshot">
  </div>

  <p>ComfyUI is a cutting-edge, modular visual AI engine designed for creating and executing sophisticated Stable Diffusion workflows. It empowers users to generate images, videos, and audio using a flexible, node-based interface, making it a powerful tool for both beginners and experienced AI artists. Compatible with Windows, Linux, and macOS.</p>

  <h2>Key Features</h2>
  <ul>
    <li><b>Node-Based Workflow:</b> Design complex Stable Diffusion pipelines visually using a graph/nodes/flowchart interface, eliminating the need for coding.</li>
    <li><b>Broad Model Support:</b> Compatible with a wide range of image, video, and audio models, including SD1.x, SD2.x, SDXL, Stable Cascade, Stable Video Diffusion, and more.  Support for various image editing models like Omnigen 2, Flux Kontext, and more.</li>
    <li><b>Extensive Model and Format Support:</b>  Loads ckpt, safetensors, and various other file formats, as well as support for embeddings, LoRAs, and Hypernetworks.</li>
    <li><b>Advanced Optimization:</b> Features an asynchronous queue system and smart memory management to handle large models, even on GPUs with limited VRAM.</li>
    <li><b>Flexible Installation:</b> Offers a Desktop Application for easy setup and a Windows Portable Package for portability, along with Manual Install options to support all operating systems and GPU types (NVIDIA, AMD, Intel, Apple Silicon, Ascend, and Cambricon).</li>
    <li><b>Workflow Management:</b> Load and save workflows as JSON files, and load workflows from PNG, WebP, and FLAC files.</li>
    <li><b>Integration with External Services:</b> Optional API nodes allow you to use paid models from external providers.</li>
    <li><b>High-Quality Previews:</b> Supports TAESD for high-quality previews with the `--preview-method taesd` command.</li>
  </ul>

  <h2>Getting Started</h2>
  <ul>
    <li><b>Desktop Application:</b> <a href="https://www.comfy.org/download" target="_blank">Download</a> the easiest way to get started, available on Windows & macOS.</li>
    <li><b>Windows Portable Package:</b> Get the latest commits and completely portable version on Windows.  <a href="https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z" target="_blank">Download</a></li>
    <li><b>Manual Install:</b>  Comprehensive instructions for Windows, Linux, and other platforms <a href="#manual-install-windows-linux">below</a>.</li>
  </ul>

  <h2><a href="https://comfyanonymous.github.io/ComfyUI_examples/" target="_blank">Examples</a></h2>
  <p>Explore what ComfyUI can do with the [example workflows](https://comfyanonymous.github.io/ComfyUI_examples/).</p>


  <h2>Installation</h2>
  <details open>
  <summary>Windows Portable</summary>
  <p>
  The easiest way to start on Windows.<br/>
  Simply download, extract with [7-Zip](https://7-zip.org) and run. Make sure you put your Stable Diffusion checkpoints/models (the huge ckpt/safetensors files) in: ComfyUI\models\checkpoints<br/>
  If you have trouble extracting it, right click the file -> properties -> unblock<br/>
  </p>
  <p>
      <b>How do I share models between another UI and ComfyUI?</b><br/>
      See the [Config file](extra_model_paths.yaml.example) to set the search paths for models. In the standalone windows build you can find this file in the ComfyUI directory. Rename this file to extra_model_paths.yaml and edit it with your favorite text editor.
  </p>
  <a href="#installing">Back to Top</a>
  </details>

  <details open>
  <summary>Comfy-cli</summary>
    You can install and start ComfyUI using comfy-cli:<br/>
    <pre>
      pip install comfy-cli<br/>
      comfy install
    </pre>
  <a href="#installing">Back to Top</a>
  </details>

  <details open>
  <summary>Manual Install (Windows, Linux)</summary>

  <p>Python 3.13 is very well supported. If you have trouble with some custom node dependencies you can try 3.12</p>

  <p>Git clone this repo.</p>

  <p>Put your SD checkpoints (the huge ckpt/safetensors files) in: models/checkpoints</p>

  <p>Put your VAE in: models/vae</p>

  <p><b>AMD GPUs (Linux only)</b></p>
  <p>AMD users can install rocm and pytorch with pip if you don't have it already installed, this is the command to install the stable version:</p>
  <pre>
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
  </pre>
  <p>This is the command to install the nightly with ROCm 6.4 which might have some performance improvements:</p>
  <pre>
  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
  </pre>

  <p><b>Intel GPUs (Windows and Linux)</b></p>
  <p><b>Option 1)</b> Intel Arc GPU users can install native PyTorch with torch.xpu support using pip. More information can be found <a href="https://pytorch.org/docs/main/notes/get_start_xpu.html">here</a></p>
  <p>To install PyTorch xpu, use the following command:</p>
  <pre>
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
  </pre>
  <p>This is the command to install the Pytorch xpu nightly which might have some performance improvements:</p>
  <pre>
  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
  </pre>
  <p><b>Option 2)</b> Alternatively, Intel GPUs supported by Intel Extension for PyTorch (IPEX) can leverage IPEX for improved performance.</p>
  <p>visit <a href="https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu">Installation</a> for more information.</p>

  <p><b>NVIDIA</b></p>
  <p>Nvidia users should install stable pytorch using this command:</p>
  <pre>
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
  </pre>
  <p>This is the command to install pytorch nightly instead which might have performance improvements.</p>
  <pre>
  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
  </pre>

  <p><b>Troubleshooting</b></p>
  <p>If you get the "Torch not compiled with CUDA enabled" error, uninstall torch with:</p>
  <pre>
  pip uninstall torch
  </pre>
  <p>And install it again with the command above.</p>

  <p><b>Dependencies</b></p>
  <p>Install the dependencies by opening your terminal inside the ComfyUI folder and:</p>
  <pre>
  pip install -r requirements.txt
  </pre>

  <p>After this you should have everything installed and can proceed to running ComfyUI.</p>

  <p><b>Others:</b></p>
  <p><b>Apple Mac silicon</b></p>

  <p>You can install ComfyUI in Apple Mac silicon (M1 or M2) with any recent macOS version.</p>
  <ol>
    <li>Install pytorch nightly. For instructions, read the <a href="https://developer.apple.com/metal/pytorch/">Accelerated PyTorch training on Mac</a> Apple Developer guide (make sure to install the latest pytorch nightly).</li>
    <li>Follow the <a href="#manual-install-windows-linux">ComfyUI manual installation</a> instructions for Windows and Linux.</li>
    <li>Install the ComfyUI <a href="#dependencies">dependencies</a>. If you have another Stable Diffusion UI <a href="#i-already-have-another-ui-for-stable-diffusion-installed-do-i-really-have-to-install-all-of-these-dependencies">you might be able to reuse the dependencies</a>.</li>
    <li>Launch ComfyUI by running <code>python main.py</code></li>
  </ol>

  <p>
  <b>Note</b>: Remember to add your models, VAE, LoRAs etc. to the corresponding Comfy folders, as discussed in <a href="#manual-install-windows-linux">ComfyUI manual installation</a>.
  </p>

  <p><b>DirectML (AMD Cards on Windows)</b></p>
  <p>This is very badly supported and is not recommended. There are some unofficial builds of pytorch ROCm on windows that exist that will give you a much better experience than this. This readme will be updated once official pytorch ROCm builds for windows come out.</p>
  <pre>
  pip install torch-directml
  </pre>
  <p>Then you can launch ComfyUI with:</p>
  <pre>
  python main.py --directml
  </pre>
  <p><b>Ascend NPUs</b></p>
  <p>For models compatible with Ascend Extension for PyTorch (torch_npu). To get started, ensure your environment meets the prerequisites outlined on the <a href="https://ascend.github.io/docs/sources/ascend/quick_install.html">installation</a> page. Here's a step-by-step guide tailored to your platform and installation method:</p>
  <ol>
    <li>Begin by installing the recommended or newer kernel version for Linux as specified in the Installation page of torch-npu, if necessary.</li>
    <li>Proceed with the installation of Ascend Basekit, which includes the driver, firmware, and CANN, following the instructions provided for your specific platform.</li>
    <li>Next, install the necessary packages for torch-npu by adhering to the platform-specific instructions on the <a href="https://ascend.github.io/docs/sources/pytorch/install.html#pytorch">Installation</a> page.</li>
    <li>Finally, adhere to the <a href="#manual-install-windows-linux">ComfyUI manual installation</a> guide for Linux. Once all components are installed, you can run ComfyUI as described earlier.</li>
  </ol>
  <p><b>Cambricon MLUs</b></p>
  <p>For models compatible with Cambricon Extension for PyTorch (torch_mlu). Here's a step-by-step guide tailored to your platform and installation method:</p>
  <ol>
    <li>Install the Cambricon CNToolkit by adhering to the platform-specific instructions on the <a href="https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cntoolkit_install_3.7.2/index.html">Installation</a></li>
    <li>Next, install the PyTorch(torch_mlu) following the instructions on the <a href="https://www.cambricon.com/docs/sdk_1.15.0/cambricon_pytorch_1.17.0/user_guide_1.9/index.html">Installation</a></li>
    <li>Launch ComfyUI by running <code>python main.py</code></li>
  </ol>
  <p><b>Iluvatar Corex</b></p>
  <p>For models compatible with Iluvatar Extension for PyTorch. Here's a step-by-step guide tailored to your platform and installation method:</p>
  <ol>
    <li>Install the Iluvatar Corex Toolkit by adhering to the platform-specific instructions on the <a href="https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=520117912052801536">Installation</a></li>
    <li>Launch ComfyUI by running <code>python main.py</code></li>
  </ol>
  <a href="#installing">Back to Top</a>
  </details>

  <h2>Running</h2>
  <pre>
  python main.py
  </pre>

  <details open>
  <summary>For AMD cards not officially supported by ROCm</summary>
  <p>
  Try running it with this command if you have issues:
  </p>
  <p>
  For 6700, 6600 and maybe other RDNA2 or older:
  <pre>
  HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py
  </pre>
  </p>
  <p>
  For AMD 7600 and maybe other RDNA3 cards:
  <pre>
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py
  </pre>
  </p>
  </details>

  <details open>
  <summary>AMD ROCm Tips</summary>
  <p>
  You can enable experimental memory efficient attention on recent pytorch in ComfyUI on some AMD GPUs using this command, it should already be enabled by default on RDNA3. If this improves speed for you on latest pytorch on your GPU please report it so that I can enable it by default.
  </p>
  <pre>
  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python main.py --use-pytorch-cross-attention
  </pre>
  <p>
  You can also try setting this env variable <code>PYTORCH_TUNABLEOP_ENABLED=1</code> which might speed things up at the cost of a very slow initial run.
  </p>
  </details>

  <h2>Notes</h2>
  <ul>
    <li>Only parts of the graph that have an output with all the correct inputs will be executed.</li>
    <li>Only parts of the graph that change from each execution to the next will be executed, if you submit the same graph twice only the first will be executed. If you change the last part of the graph only the part you changed and the part that depends on it will be executed.</li>
    <li>Dragging a generated png on the webpage or loading one will give you the full workflow including seeds that were used to create it.</li>
    <li>You can use () to change emphasis of a word or phrase like: (good code:1.2) or (bad code:0.8). The default emphasis for () is 1.1. To use () characters in your actual prompt escape them like \\( or \\).</li>
    <li>You can use {day|night}, for wildcard/dynamic prompts. With this syntax "{wild|card|test}" will be randomly replaced by either "wild", "card" or "test" by the frontend every time you queue the prompt. To use {} characters in your actual prompt escape them like: \\{ or \\}.</li>
    <li>Dynamic prompts also support C-style comments, like <code>// comment</code> or <code>/* comment */</code>.</li>
    <li>To use a textual inversion concepts/embeddings in a text prompt put them in the models/embeddings directory and use them in the CLIPTextEncode node like this (you can omit the .pt extension):</li>
  </ul>
  <pre>
  embedding:embedding_filename.pt
  </pre>

  <h2>How to show high-quality previews?</h2>
  <p>Use <code>--preview-method auto</code> to enable previews.</p>
  <p>The default installation includes a fast latent preview method that's low-resolution. To enable higher-quality previews with <a href="https://github.com/madebyollin/taesd">TAESD</a>, download the [taesd_decoder.pth, taesdxl_decoder.pth, taesd3_decoder.pth and taef1_decoder.pth](https://github.com/madebyollin/taesd/) and place them in the <code>models/vae_approx</code> folder. Once they're installed, restart ComfyUI and launch it with <code>--preview-method taesd</code> to enable high-quality previews.</p>

  <h2>How to use TLS/SSL?</h2>
  <p>Generate a self-signed certificate (not appropriate for shared/production use) and key by running the command: <code>openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 3650 -nodes -subj "/C=XX/ST=StateName/L=CityName/O=CompanyName/OU=CompanySectionName/CN=CommonNameOrHostname"</code></p>
  <p>Use <code>--tls-keyfile key.pem --tls-certfile cert.pem</code> to enable TLS/SSL, the app will now be accessible with <code>https://...</code> instead of <code>http://...</code>.</p>

  <p>
  <b>Note</b>: Windows users can use <a href="https://github.com/alexisrolland/docker-openssl">alexisrolland/docker-openssl</a> or one of the <a href="https://wiki.openssl.org/index.php/Binaries">3rd party binary distributions</a> to run the command example above.
  <br/><br/>If you use a container, note that the volume mount <code>-v</code> can be a relative path so <code>... -v ".\:/openssl-certs" ...</code> would create the key & cert files in the current directory of your command prompt or powershell terminal.
  </p>

  <h2>Support and Community</h2>
  <ul>
    <li><b>Discord:</b> <a href="https://comfy.org/discord" target="_blank">Join the Discord server</a> for help and feedback.  Check the #help or #feedback channels.</li>
    <li><b>Matrix:</b> <a href="https://app.element.io/#/room/%23comfyui_space%3Amatrix.org" target="_blank">Join the Matrix space</a> (open source alternative to Discord).</li>
    <li><b>Website:</b>  Learn more on the <a href="https://www.comfy.org/" target="_blank">ComfyUI website</a>.</li>
  </ul>

  <h2>Frontend Development</h2>
  <p>As of August 15, 2024, we have transitioned to a new frontend, which is now hosted in a separate repository: <a href="https://github.com/Comfy-Org/ComfyUI_frontend">ComfyUI Frontend</a>. This repository now hosts the compiled JS (from TS/Vue) under the <code>web/</code> directory.</p>
  <h3>Reporting Issues and Requesting Features</h3>
  <p>For any bugs, issues, or feature requests related to the frontend, please use the <a href="https://github.com/Comfy-Org/ComfyUI_frontend">ComfyUI Frontend repository</a>. This will help us manage and address frontend-specific concerns more efficiently.</p>
  <h3>Using the Latest Frontend</h3>
  <p>The new frontend is now the default for ComfyUI. However, please note:</p>
  <ol>
    <li>The frontend in the main ComfyUI repository is updated fortnightly.</li>
    <li>Daily releases are available in the separate frontend repository.</li>
  </ol>
  <p>To use the most up-to-date frontend version:</p>
  <ol>
    <li>For the latest daily release, launch ComfyUI with this command line argument:</li>
  </ol>
  <pre>
  --front-end-version Comfy-Org/ComfyUI_frontend@latest
  </pre>
  <ol>
    <li>For a specific version, replace <code>latest</code> with the desired version number:</li>
  </ol>
  <pre>
  --front-end-version Comfy-Org/ComfyUI_frontend@1.2.2
  </pre>
  <p>This approach allows you to easily switch between the stable fortnightly release and the cutting-edge daily updates, or even specific versions for testing purposes.</p>
  <h3>Accessing the Legacy Frontend</h3>
  <p>If you need to use the legacy frontend for any reason, you can access it using the following command line argument:</p>
  <pre>
  --front-end-version Comfy-Org/ComfyUI_legacy_frontend@latest
  </pre>
  <p>This will use a snapshot of the legacy frontend preserved in the <a href="https://github.com/Comfy-Org/ComfyUI_legacy_frontend">ComfyUI Legacy Frontend repository</a>.</p>

  <h2>QA</h2>

  <details open>
  <summary>Which GPU should I buy for this?</summary>
  <p>
      Check out the <a href="https://github.com/comfyanonymous/ComfyUI/wiki/Which-GPU-should-I-buy-for-ComfyUI">GPU Recommendations page</a> for some recommendations.
  </p>
  </details>

  <p>
    <a href="https://github.com/comfyanonymous/ComfyUI" target="_blank">Back to ComfyUI Repository</a>
  </p>
</div>
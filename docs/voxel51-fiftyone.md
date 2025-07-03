<div align="center">
<p align="center">
<!-- prettier-ignore -->
<img src="https://user-images.githubusercontent.com/25985824/106288517-2422e000-6216-11eb-871d-26ad2e7b1e59.png" height="55px"> &nbsp;
<img src="https://user-images.githubusercontent.com/25985824/106288518-24bb7680-6216-11eb-8f10-60052c519586.png" height="50px">
</p>
</div>

# FiftyOne: The Open-Source Tool for Supercharging Your Computer Vision Projects

**FiftyOne** is the ultimate open-source tool designed to help you build high-quality datasets and computer vision models more efficiently than ever. [Explore the original repository](https://github.com/voxel51/fiftyone).

---

**Quick Links:** [Website](https://voxel51.com/fiftyone) • [Docs](https://voxel51.com/docs/fiftyone) • [Try it Now](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb) • [Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html) • [Examples](https://github.com/voxel51/fiftyone-examples) • [Blog](https://voxel51.com/blog/) • [Community](https://discord.gg/fiftyone-community)

[![PyPI python](https://img.shields.io/pypi/pyversions/fiftyone)](https://pypi.org/project/fiftyone)
[![PyPI version](https://badge.fury.io/py/fiftyone.svg)](https://pypi.org/project/fiftyone)
[![Downloads](https://static.pepy.tech/badge/fiftyone)](https://pepy.tech/project/fiftyone)
[![Docker Pulls](https://badgen.net/docker/pulls/voxel51/fiftyone?icon=docker&label=pulls)](https://hub.docker.com/r/voxel51/fiftyone/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)
[![Mailing list](http://bit.ly/2Md9rxM)](https://share.hsforms.com/1zpJ60ggaQtOoVeBqIZdaaA2ykyk)
[![Twitter](https://img.shields.io/twitter/follow/Voxel51?style=social)](https://twitter.com/voxel51)

## Key Features

*   **Dataset Visualization**: Visualize and explore complex datasets, including images, videos, and their associated labels, with an intuitive visual interface.
*   **Embedding Exploration**:  Explore embeddings, select points of interest, and view corresponding samples and labels for deeper data understanding.
*   **Model Analysis and Improvement**: Evaluate model performance, identify failure modes, and refine your models effectively.
*   **Advanced Data Curation**: Quickly identify and resolve data issues, annotation errors, and edge cases, ensuring data quality.
*   **Rich Integrations**: Works seamlessly with popular deep learning libraries such as PyTorch, Hugging Face, and Ultralytics.
*   **Open and Extensible**: Customize and extend FiftyOne to precisely match your unique requirements and project needs.

<div id='installation'/>

## Installation

Install FiftyOne with a simple pip command:

```shell
pip install fiftyone
```

<details>
<summary>More Details</summary>

### Installation Options

FiftyOne supports Python 3.9 - 3.11.

For most users, we recommend installing the latest release version of FiftyOne
via `pip` as shown above.

If you want to contribute to FiftyOne or install the latest development
version, then you can also perform a [source install](#source-install).

See the [prerequisites section](#prerequisites) for system-specific setup
information.

We strongly recommend that you install FiftyOne in a
[virtual environment](https://voxel51.com/docs/fiftyone/getting_started/virtualenv.html)
to maintain a clean workspace.

Consult the
[installation guide](https://voxel51.com/docs/fiftyone/getting_started/install.html)
for troubleshooting and other information about getting up-and-running with
FiftyOne.

</details>

<div id='source-install'/>

<details>
<summary>Install from source</summary>

### Source Installations

Follow the instructions below to install FiftyOne from source and build the
App.

You'll need the following tools installed:

-   [Python](https://www.python.org) (3.9 - 3.11)
-   [Node.js](https://nodejs.org) - on Linux, we recommend using
    [nvm](https://github.com/nvm-sh/nvm) to install an up-to-date version.
-   [Yarn](https://yarnpkg.com) - once Node.js is installed, you can
    [enable Yarn](https://yarnpkg.com/getting-started/install) via
    `corepack enable`

We strongly recommend that you install FiftyOne in a
[virtual environment](https://voxel51.com/docs/fiftyone/getting_started/virtualenv.html)
to maintain a clean workspace.

If you are working in Google Colab,
[skip to here](#source-installs-in-google-colab).

First, clone the repository:

```shell
git clone https://github.com/voxel51/fiftyone
cd fiftyone
```

Then run the install script:

```shell
# Mac or Linux
bash install.bash

# Windows
.\install.bat
```

If you run into issues importing FiftyOne, you may need to add the path to the
cloned repository to your `PYTHONPATH`:

```shell
export PYTHONPATH=$PYTHONPATH:/path/to/fiftyone
```

Note that the install script adds to your `nvm` settings in your `~/.bashrc` or
`~/.bash_profile`, which is needed for installing and building the App.

### Upgrading your source installation

To upgrade an existing source installation to the bleeding edge, simply pull
the latest `develop` branch and rerun the install script:

```shell
git checkout develop
git pull

# Mac or Linux
bash install.bash

# Windows
.\install.bat
```

### Rebuilding the App

When you pull in new changes to the App, you will need to rebuild it, which you
can do either by rerunning the install script or just running `yarn build` in
the `./app` directory.

### Developer installation

If you would like to
[contribute to FiftyOne](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md),
you should perform a developer installation using the `-d` flag of the install
script:

```shell
# Mac or Linux
bash install.bash -d

# Windows
.\install.bat -d
```

Although not required, developers typically prefer to configure their FiftyOne
installation to connect to a self-installed and managed instance of MongoDB,
which you can do by following
[these simple steps](https://docs.voxel51.com/user_guide/config.html#configuring-a-mongodb-connection).

### Source installs in Google Colab

You can install from source in
[Google Colab](https://colab.research.google.com) by running the following in a
cell and then **restarting the runtime**:

```shell
%%shell

git clone --depth 1 https://github.com/voxel51/fiftyone.git
cd fiftyone

# Mac or Linux
bash install.bash

# Windows
.\install.bat
```

### Generating documentation

See the
[docs guide](https://github.com/voxel51/fiftyone/blob/develop/docs/README.md)
for information on building and contributing to the documentation.

### Uninstallation

You can uninstall FiftyOne as follows:

```shell
pip uninstall fiftyone fiftyone-brain fiftyone-db
```

</details>

<div id='prerequisites'/>

<details>
<summary>Prerequisites for beginners</summary>

### System-specific setup

Follow the instructions for your operating system or environment to perform
basic system setup before [installing FiftyOne](#installation).

If you're an experienced developer, you've likely already done this.

<details>
<summary>Linux</summary>

<div id='prerequisites-linux'/>

#### 1. Install Python and other dependencies

These steps work on a clean install of Ubuntu Desktop 24.04, and should also
work on Ubuntu 24.04 and 22.04, and on Ubuntu Server:

```shell
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3-venv python3-dev build-essential git-all libgl1-mesa-dev
```

-   On Linux, you will need at least the `openssl` and `libcurl` packages
-   On Debian-based distributions, you will need to install `libcurl4` or
    `libcurl3` instead of `libcurl`, depending on the age of your distribution

```shell
# Ubuntu
sudo apt install libcurl4 openssl

# Fedora
sudo dnf install libcurl openssl
```

#### 2. Create and activate a virtual environment

```shell
python3 -m venv fiftyone_env
source fiftyone_env/bin/activate
```

#### 3. Install FFmpeg (optional)

If you plan to work with video datasets, you'll need to install
[FFmpeg](https://ffmpeg.org):

```shell
sudo apt-get install ffmpeg
```

</details>

<details>
<summary>MacOS</summary>

<div id='prerequisites-macos'/>

#### 1. Install Xcode Command Line Tools

```shell
xcode-select --install
```

#### 2. Install Homebrew

```shell
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After running the above command, follow the instructions in your terminal to
complete the Homebrew installation.

#### 3. Install Python

```shell
brew install python@3.9
brew install protobuf
```

#### 4. Create and activate a virtual environment

```shell
python3 -m venv fiftyone_env
source fiftyone_env/bin/activate
```

#### 5. Install FFmpeg (optional)

If you plan to work with video datasets, you'll need to install
[FFmpeg](https://ffmpeg.org):

```shell
brew install ffmpeg
```

</details>

<details>
<summary>Windows</summary>

<div id='prerequisites-windows'/>

#### 1. Install Python

⚠️ The version of Python that is available in the Microsoft Store is **not
recommended** ⚠️

Download a Python 3.9 - 3.11 installer from
[python.org](https://www.python.org/downloads/). Make sure to pick a 64-bit
version. For example, this
[Python 3.10.11 installer](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe).

Double-click on the installer to run it, and follow the steps in the installer.

-   Check the box to add Python to your `PATH`
-   At the end of the installer, there is an option to disable the `PATH`
    length limit. It is recommended to click this

#### 2. Install Microsoft Visual C++

Download
[Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
Double-click on the installer to run it, and follow the steps in the installer.

#### 3. Install Git

Download Git from [this link](https://git-scm.com/download/win). Double-click
on the installer to run it, and follow the steps in the installer.

#### 4. Create and activate a virtual environment

-   Press `Win + R`. type `cmd`, and press `Enter`. Alternatively, search
    **Command Prompt** in the Start Menu.
-   Navigate to your project. `cd C:\path\to\your\project`
-   Create the environment `python -m venv fiftyone_env`
-   Activate the environment typing this in the command line window
    `fiftyone_env\Scripts\activate`
-   After activation, your command prompt should change and show the name of
    the virtual environment `(fiftyon_env) C:\path\to\your\project`

#### 5. Install FFmpeg (optional)

If you plan to work with video datasets, you'll need to install
[FFmpeg](https://ffmpeg.org).

Download an FFmpeg binary from [here](https://ffmpeg.org/download.html). Add
FFmpeg's path (e.g., `C:\ffmpeg\bin`) to your `PATH` environmental variable.

</details>

<details>
<summary>Docker</summary>

<div id='prerequisites-docker'/>
<br>

Refer to
[these instructions](https://voxel51.com/docs/fiftyone/environments/index.html#docker)
to see how to build and run Docker images containing release or source builds
of FiftyOne.

</details>

</details>

<div id='quickstart'>

## Quickstart

Get started with FiftyOne quickly by running the following Python code snippet:

```py
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
```

Then, explore the quickstart dataset using [this Colab notebook](https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb).  Remember to include `session.wait()` in your script to keep the app open.

<div id='additional-resources'>

## Additional Resources

Explore these related resources to enhance your FiftyOne experience:

| Resource                                           | Description                                                                 |
| -------------------------------------------------- | --------------------------------------------------------------------------- |
| [FiftyOne Enterprise](https://voxel51.com/enterprise) | Secure collaboration, cloud workflows, and enterprise features.           |
| [VoxelGPT](https://github.com/voxel51/voxelgpt)    | An AI tool to help accelerate computer vision projects.                    |
| [Plugins](https://voxel51.com/plugins)            | Extend FiftyOne's capabilities with custom plugins.                       |
| [Vector Search](https://voxel51.com/blog/the-computer-vision-interface-for-vector-search) | Integrate vector search for advanced data exploration.                     |
| [Dataset Zoo](https://docs.voxel51.com/dataset_zoo/index.html)    | Access a collection of pre-loaded datasets.                               |
| [Model Zoo](https://docs.voxel51.com/model_zoo/index.html) | Access pre-trained models for computer vision tasks.                      |
| [FiftyOne Brain](https://docs.voxel51.com/brain.html)   | Analyze and understand your data using AI-powered features.            |

</div>

<div id='documentation'/>

## Documentation

Comprehensive documentation is available at [fiftyone.ai](https://fiftyone.ai) to help you get the most out of FiftyOne:

| Documentation                                                              | Description                                                                |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| [Tutorials](https://voxel51.com/docs/fiftyone/tutorials/index.html)      | Step-by-step guides to get you started with FiftyOne.                  |
| [Recipes](https://voxel51.com/docs/fiftyone/recipes/index.html)          | Ready-to-use code snippets for common tasks.                                |
| [Examples](https://github.com/voxel51/fiftyone-examples)                 | Practical examples to learn and apply FiftyOne features.                |
| [User Guide](https://voxel51.com/docs/fiftyone/user_guide/index.html)    | In-depth guide to using FiftyOne.                                        |
| [CLI Documentation](https://voxel51.com/docs/fiftyone/cli/index.html)    | Information about the FiftyOne command-line interface.                     |
| [API Reference](https://voxel51.com/docs/fiftyone/api/fiftyone.html)     | Detailed reference for the FiftyOne API.                               |

</div>

<div id='fiftyone-teams'/>

## FiftyOne Enterprise

Looking for secure cloud collaboration and automated workflows?  Check out [FiftyOne Enterprise](https://voxel51.com/enterprise) for enterprise-grade features.

<div id='faq'/>

## FAQ and Troubleshooting

Refer to the [common issues](https://docs.voxel51.com/getting_started/troubleshooting.html) page for assistance with installation problems.  Find more answers in our [frequently asked questions](https://docs.voxel51.com/faq/index.html). If you need further help, please [open an issue on GitHub](https://github.com/voxel51/fiftyone/issues) or contact us on [Discord](https://discord.gg/fiftyone-community).

</div>

<div id='community'/>

## Join Our Community

Connect with us and get support through the following channels:

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?logo=twitter&logoColor=white)](https://twitter.com/voxel51)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/voxel51)
[![Facebook](https://img.shields.io/badge/Facebook-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/voxel51)

🎉 **Share how FiftyOne helps your visual AI projects by tagging us @Voxel51 and using the hashtag #FiftyOne** 🎉

</div>

<div id='contributors'/>

## Contributors

FiftyOne is an open-source project, and contributions are warmly welcomed!  See the [contribution guide](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md) to learn how to contribute.

Special thanks to the amazing people who have contributed to FiftyOne!

<a href="https://github.com/voxel51/fiftyone/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=voxel51/fiftyone" />
</a>

<div id='citation'/>

## Citation

If you use FiftyOne in your research, please cite the project:

```bibtex
@article{moore2020fiftyone,
  title={FiftyOne},
  author={Moore, B. E. and Corso, J. J.},
  journal={GitHub. Note: https://github.com/voxel51/fiftyone},
  year={2020}
}
```
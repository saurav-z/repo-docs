<p align="center"><img width="65%" src="docs/source/_static/img/evalai_logo.png"/></p>

# EvalAI: The Open Source Platform for AI Algorithm Evaluation

**EvalAI is an open-source platform revolutionizing AI research by providing a centralized hub for evaluating, comparing, and collaborating on machine learning and AI algorithms.**

[![Join Slack](https://img.shields.io/badge/Join%20Slack-Chat-blue?logo=slack)](https://join.slack.com/t/cloudcv-community/shared_invite/zt-3252n6or8-e0QuZKIZFLB0zXtQ6XgxfA)
[![Build Status](https://travis-ci.org/Cloud-CV/EvalAI.svg?branch=master)](https://travis-ci.org/Cloud-CV/EvalAI)
[![Coverage](https://img.shields.io/codecov/c/github/Cloud-CV/EvalAI?label=Coverage&style=flat-square)](https://codecov.io/gh/Cloud-CV/EvalAI)
[![Backend Coverage](https://img.shields.io/codecov/c/github/Cloud-CV/EvalAI?flag=backend&label=Backend&style=flat-square)](https://codecov.io/gh/Cloud-CV/EvalAI?flag=backend)
[![Frontend Coverage](https://img.shields.io/codecov/c/github/Cloud-CV/EvalAI?flag=frontend&label=Frontend&style=flat-square)](https://codecov.io/gh/Cloud-CV/EvalAI?flag=frontend)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/markdown-guide/badge/?version=latest)](http://evalai.readthedocs.io/en/latest/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/Cloud-CV/EvalAI?style=flat-square)](https://github.com/Cloud-CV/EvalAI/tree/master)
[![Open Collective](https://opencollective.com/evalai/backers/badge.svg)](https://opencollective.com/evalai#backers)
[![Open Collective](https://opencollective.com/evalai/sponsors/badge.svg)](https://opencollective.com/evalai#sponsors)
[![Twitter Follow](https://img.shields.io/twitter/follow/eval_ai?style=social)](https://twitter.com/eval_ai)

EvalAI simplifies AI research by providing a centralized platform for:

*   **Standardized Evaluation:** Ensures fair and accurate comparisons of AI algorithms.
*   **Reproducibility:** Enables researchers to easily reproduce results from papers.
*   **Collaboration:** Facilitates the organization and participation of AI challenges.
*   **Efficiency:** Offers swift and robust backends for rapid evaluation.

## Key Features

*   **Custom Evaluation Protocols:** Define custom evaluation phases, dataset splits, and leaderboards.
*   **Remote Evaluation:** Support for challenges requiring specialized compute resources.
*   **Dockerized Environments:**  Submit code as Docker images for consistent and isolated evaluation.
*   **CLI Support:**  Utilize the [EvalAI CLI](https://github.com/Cloud-CV/evalai-cli) for enhanced command-line interaction.
*   **Portability & Scalability:** Built with open-source technologies like Docker, Django, and PostgreSQL for scalability.
*   **Faster Evaluation:** Optimized for speed through worker node warm-up and dataset chunking.

## Benefits of Using EvalAI

*   **Simplified Algorithm Comparison:** Easily compare your algorithms with existing approaches.
*   **Improved Research Reproducibility:** Replicate results from technical papers.
*   **Enhanced Collaboration:** Participate in AI challenges and contribute to the advancement of AI.
*   **Efficient Evaluation:** Reduce evaluation time with optimized infrastructure.

## Installation

Get started with EvalAI using Docker:

1.  Install [Docker](https://docs.docker.com/install/) and [Docker Compose](https://docs.docker.com/compose/install/).
2.  Clone the repository:

    ```bash
    git clone https://github.com/Cloud-CV/EvalAI.git evalai && cd evalai
    ```
3.  Build and run the containers:

    ```bash
    docker-compose up --build
    ```
    For worker services:
    ```bash
    docker-compose --profile worker up --build
    ```
    For statsd-exporter service:
    ```bash
    docker-compose --profile statsd up --build
    ```
    For both optional services:
    ```bash
    docker-compose --profile worker --profile statsd up --build
    ```
4.  Access EvalAI in your browser:  `http://127.0.0.1:8888`
    Use the following default user credentials:

    *   **SUPERUSER:** username: `admin` / password: `password`
    *   **HOST USER:** username: `host` / password: `password`
    *   **PARTICIPANT USER:** username: `participant` / password: `password`

    For troubleshooting, see the [common errors during installation](https://evalai.readthedocs.io/en/latest/faq(developers).html#common-errors-during-installation).

## Documentation

For documentation setup instructions, refer to the `docs/README.md` file.

## Citing EvalAI

If using EvalAI, please cite this technical report:

```
@article{EvalAI,
    title   =  {EvalAI: Towards Better Evaluation Systems for AI Agents},
    author  =  {Deshraj Yadav and Rishabh Jain and Harsh Agrawal and Prithvijit
                Chattopadhyay and Taranjeet Singh and Akash Jain and Shiv Baran
                Singh and Stefan Lee and Dhruv Batra},
    year    =  {2019},
    volume  =  arXiv:1902.03570
}
```
<p>
    <a href="http://learningsys.org/sosp19/assets/papers/23_CameraReadySubmission_EvalAI_SOSP_2019%20(8)%20(1).pdf" target="_blank"><img src="docs/source/_static/img/evalai-paper.jpg"/></a>
</p>

## Team

EvalAI is maintained by [Rishabh Jain](https://rishabhjain.xyz/), [Gunjan Chhablani](https://gchhablani.github.io/), and [Dhruv Batra](https://www.cc.gatech.edu/~dbatra/).

## Contribution Guidelines

Contribute to EvalAI by following our [contribution guidelines](https://github.com/Cloud-CV/EvalAI/blob/master/.github/CONTRIBUTING.md).

## Contributors

[//]: contributor-faces
<a href="https://github.com/RishabhJain2018" target="_blank"><img src="https://avatars.githubusercontent.com/u/12206047?v=4" title="RishabhJain2018" width="50" height="50"></a> <a href="https://github.com/deshraj" target="_blank"><img src="https://avatars.githubusercontent.com/u/2945708?v=4" title="deshraj" width="50" height="50"></a> <a href="https://github.com/Ram81" target="_blank"><img src="https://avatars.githubusercontent.com/u/16323427?v=4" title="Ram81" width="50" height="50"></a> <a href="https://github.com/gchhablani" target="_blank"><img src="https://avatars.githubusercontent.com/u/29076344?v=4" title="gchhablani" width="50" height="50"></a> <a href="https://github.com/taranjeet" target="_blank"><img src="https://avatars.githubusercontent.com/u/4302268?v=4" title="taranjeet" width="50" height="50"></a> <a href="https://github.com/Sanji515" target="_blank"><img src="https://avatars.githubusercontent.com/u/32524438?v=4" title="Sanji515" width="50" height="50"></a> <a href="https://github.com/aka-jain" target="_blank"><img src="https://avatars.githubusercontent.com/u/11537940?v=4" title="aka-jain" width="50" height="50"></a> <a href="https://github.com/gautamjajoo" target="_blank"><img src="https://avatars.githubusercontent.com/u/24366008?v=4" title="gautamjajoo" width="50" height="50"></a> <a href="https://github.com/Kajol-Kumari" target="_blank"><img src="https://avatars.githubusercontent.com/u/44888949?v=4" title="Kajol-Kumari" width="50" height="50"></a> <a href="https://github.com/Suryansh5545" target="_blank"><img src="https://avatars.githubusercontent.com/u/34577232?v=4" title="Suryansh5545" width="50" height="50"></a> <a href="https://github.com/Ayukha" target="_blank"><img src="https://avatars.githubusercontent.com/u/19167324?v=4" title="Ayukha" width="50" height="50"></a> <a href="https://github.com/spyshiv" target="_blank"><img src="https://avatars.githubusercontent.com/u/7015220?v=4" title="spyshiv" width="50" height="50"></a> <a href="https://github.com/Arun-Jain" target="_blank"><img src="https://avatars.githubusercontent.com/u/16155501?v=4" title="Arun-Jain" width="50" height="50"></a> <a href="https://github.com/krtkvrm" target="_blank"><img src="https://avatars.githubusercontent.com/u/3920286?v=4" title="krtkvrm" width="50" height="50"></a> <a href="https://github.com/KhalidRmb" target="_blank"><img src="https://avatars.githubusercontent.com/u/31621523?v=4" title="KhalidRmb" width="50" height="50"></a> <a href="https://github.com/guyandtheworld" target="_blank"><img src="https://avatars.githubusercontent.com/u/20072816?v=4" title="guyandtheworld" width="50" height="50"></a> <a href="https://github.com/burnerlee" target="_blank"><img src="https://avatars.githubusercontent.com/u/55936223?v=4" title="burnerlee" width="50" height="50"></a> <a href="https://github.com/muddlebee" target="_blank"><img src="https://avatars.githubusercontent.com/u/8139783?v=4" title="muddlebee" width="50" height="50"></a> <a href="https://github.com/Akshat453" target="_blank"><img src="https://avatars.githubusercontent.com/u/158801446?v=4" title="Akshat453" width="50" height="50"></a> <a href="https://github.com/AyushR1" target="_blank"><img src="https://avatars.githubusercontent.com/u/22369791?v=4" title="AyushR1" width="50" height="50"></a> <a href="https://github.com/sanketbansal" target="_blank"><img src="https://avatars.githubusercontent.com/u/17106489?v=4" title="sanketbansal" width="50" height="50"></a> <a href="https://github.com/savish28" target="_blank"><img src="https://avatars.githubusercontent.com/u/32800267?v=4" title="savish28" width="50" height="50"></a> <a href="https://github.com/aditi-dsi" target="_blank"><img src="https://avatars.githubusercontent.com/u/123075271?v=4" title="aditi-dsi" width="50" height="50"></a> <a href="https://github.com/Alabhya268" target="_blank"><img src="https://avatars.githubusercontent.com/u/57143358?v=4" title="Alabhya268" width="50" height="50"></a> <a href="https://github.com/dependabot[bot]" target="_blank"><img src="https://avatars.githubusercontent.com/in/29110?v=4" title="dependabot[bot]" width="50" height="50"></a> <a href="https://github.com/live-wire" target="_blank"><img src="https://avatars.githubusercontent.com/u/6399428?v=4" title="live-wire" width="50" height="50"></a> <a href="https://github.com/gauthamzz" target="_blank"><img src="https://avatars.githubusercontent.com/u/12110844?v=4" title="gauthamzz" width="50" height="50"></a> <a href="https://github.com/HargovindArora" target="_blank"><img src="https://avatars.githubusercontent.com/u/22341493?v=4" title="HargovindArora" width="50" height="50"></a> <a href="https://github.com/dexter1691" target="_blank"><img src="https://avatars.githubusercontent.com/u/2039548?v=4" title="dexter1691" width="50" height="50"></a> <a href="https://github.com/akanshajain231999" target="_blank"><img src="https://avatars.githubusercontent.com/u/48309147?v=4" title="akanshajain231999" width="50" height="50"></a> <a href="https://github.com/harshithdwivedi" target="_blank"><img src="https://avatars.githubusercontent.com/u/47669588?v=4" title="harshithdwivedi" width="50" height="50"></a> <a href="https://github.com/nikochiko" target="_blank"><img src="https://avatars.githubusercontent.com/u/37668193?v=4" title="nikochiko" width="50" height="50"></a> <a href="https://github.com/Zahed-Riyaz" target="_blank"><img src="https://avatars.githubusercontent.com/u/188900716?v=4" title="Zahed-Riyaz" width="50" height="50"></a> <a href="https://github.com/jayaike" target="_blank"><img src="https://avatars.githubusercontent.com/u/35180217?v=4" title="jayaike" width="50" height="50"></a> <a href="https://github.com/hkmatsumoto" target="_blank"><img src="https://avatars.githubusercontent.com/u/57856193?v=4" title="hkmatsumoto" width="50" height="50"></a> <a href="https://github.com/TheArchitect19" target="_blank"><img src="https://avatars.githubusercontent.com/u/91387353?v=4" title="TheArchitect19" width="50" height="50"></a> <a href="https://github.com/matthew-so" target="_blank"><img src="https://avatars.githubusercontent.com/u/42504035?v=4" title="matthew-so" width="50" height="50"></a> <a href="https://github.com/xamfy" target="_blank"><img src="https://avatars.githubusercontent.com/u/19357995?v=4" title="xamfy" width="50" height="50"></a> <a href="https://github.com/ShauryaAg" target="_blank"><img src="https://avatars.githubusercontent.com/u/31778302?v=4" title="ShauryaAg" width="50" height="50"></a> <a href="https://github.com/DXGatech" target="_blank"><img src="https://avatars.githubusercontent.com/u/28953079?v=4" title="DXGatech" width="50" height="50"></a> <a href="https://github.com/drepram" target="_blank"><img src="https://avatars.githubusercontent.com/u/34530026?v=4" title="drepram" width="50" height="50"></a> <a href="https://github.com/yadavankit" target="_blank"><img src="https://avatars.githubusercontent.com/u/8945824?v=4" title="yadavankit" width="50" height="50"></a> <a href="https://github.com/sachinmukherjee" target="_blank"><img src="https://avatars.githubusercontent.com/u/19318218?v=4" title="sachinmukherjee" width="50" height="50"></a> <a href="https://github.com/kurianbenoy" target="_blank"><img src="https://avatars.githubusercontent.com/u/24592806?v=4" title="kurianbenoy" width="50" height="50"></a> <a href="https://github.com/mayank-agarwal-96" target="_blank"><img src="https://avatars.githubusercontent.com/u/11095642?v=4" title="mayank-agarwal-96" width="50" height="50"></a> <a href="https://github.com/codervivek" target="_blank"><img src="https://avatars.githubusercontent.com/u/26835119?v=4" title="codervivek" width="50" height="50"></a> <a href="https://github.com/yashdusing" target="_blank"><img src="https://avatars.githubusercontent.com/u/19976688?v=4" title="yashdusing" width="50" height="50"></a> <a href="https://github.com/jayantsa" target="_blank"><img src="https://avatars.githubusercontent.com/u/10354780?v=4" title="jayantsa" width="50" height="50"></a> <a href="https://github.com/vinceli1004" target="_blank"><img src="https://avatars.githubusercontent.com/u/39491501?v=4" title="vinceli1004" width="50" height="50"></a> <a href="https://github.com/pavan-simplr" target="_blank"><img src="https://avatars.githubusercontent.com/u/66268853?v=4" title="pavan-simplr" width="50" height="50"></a> <a href="https://github.com/varunagrawal" target="_blank"><img src="https://avatars.githubusercontent.com/u/975964?v=4" title="varunagrawal" width="50" height="50"></a> <a href="https://github.com/ParthS007" target="_blank"><img src="https://avatars.githubusercontent.com/u/24358501?v=4" title="ParthS007" width="50" height="50"></a> <a href="https://github.com/tendstofortytwo" target="_blank"><img src="https://avatars.githubusercontent.com/u/5107795?v=4" title="tendstofortytwo" width="50" height="50"></a> <a href="https://github.com/geekayush" target="_blank"><img src="https://avatars.githubusercontent.com/u/22499864?v=4" title="geekayush" width="50" height="50"></a> <a href="https://github.com/itaditya" target="_blank"><img src="https://avatars.githubusercontent.com/u/15871340?v=4" title="itaditya" width="50" height="50"></a> <a href="https://github.com/lazyperson1020" target="_blank"><img src="https://avatars.githubusercontent.com/u/117618223?v=4" title="lazyperson1020" width="50" height="50"></a> <a href="https://github.com/dhruvbatra" target="_blank"><img src="https://avatars.githubusercontent.com/u/2941091?v=4" title="dhruvbatra" width="50" height="50"></a> <a href="https://github.com/viditjain08" target="_blank"><img src="https://avatars.githubusercontent.com/u/5248993?v=4" title="viditjain08" width="50" height="50"></a> <a href="https://github.com/souravsingh" target="_blank"><img src="https://avatars.githubusercontent.com/u/4314261?v=4" title="souravsingh" width="50" height="50"></a> <a href="https://github.com/Curious72" target="_blank"><img src="https://avatars.githubusercontent.com/u/8409274?v=4" title="Curious72" width="50" height="50"></a> <a href="https://github.com/sarthak212" target="_blank"><img src="https://avatars.githubusercontent.com/u/33999269?v=4" title="sarthak212" width="50" height="50"></a> <a href="https://github.com/priyapahwa" target="_blank"><img src="https://avatars.githubusercontent.com/u/77075449?v=4" title="priyapahwa" width="50" height="50"></a> <a href="https://github.com/parth-verma" target="_blank"><img src="https://avatars.githubusercontent.com/u/22412980?v=4" title="parth-verma" width="50" height="50"></a> <a href="https://github.com/nagpalm7" target="_blank"><img src="https://avatars.githubusercontent.com/u/32512296?v=4" title="nagpalm7" width="50" height="50"></a> <a href="https://github.com/tawAsh1" target="_blank"><img src="https://avatars.githubusercontent.com/u/7100187?v=4" title="tawAsh1" width="50" height="50"></a> <a href="https://github.com/cwiggs" target="_blank"><img src="https://avatars.githubusercontent.com/u/5607419?v=4" title="cwiggs" width="50" height="50"></a> <a href="https://github.com/AnshulBasia" target="_blank"><img src="https://avatars.githubusercontent.com/u/12856392?v=4" title="AnshulBasia" width="50" height="50"></a> <a href="https://github.com/afif-fahim" target="_blank"><img src="https://avatars.githubusercontent.com/u/33936462?v=4" title="afif-fahim" width="50" height="50"></a> <a href="https://github.com/aayusharora" target="_blank"><img src="https://avatars.githubusercontent.com/u/12194719?v=4" title="aayusharora" width="50" height="50"></a> <a href="https://github.com/sanyamdogra" target="_blank"><img src="https://avatars.githubusercontent.com/u/33497630?v=4" title="sanyamdogra" width="50" height="50"></a> <a href="https://github.com/sayamkanwar" target="_blank"><img src="https://avatars.githubusercontent.com/u/10847009?v=4" title="sayamkanwar" width="50" height="50"></a> <a href="https://github.com/shakeelsamsu" target="_blank"><img src="https://avatars.githubusercontent.com/u/16440459?v=4" title="shakeelsamsu" width="50" height="50"></a> <a href="https://github.com/shiv6146" target="_blank"><img src="https://avatars.githubusercontent.com/u/5592146?v=4" title="shiv6146" width="50" height="50"></a> <a href="https://github.com/skbly7" target="_blank"><img src="https://avatars.githubusercontent.com/u/3490586?v=4" title="skbly7" width="50" height="50"></a> <a href="https://github.com/tashachin" target="_blank"><img src="https://avatars.githubusercontent.com/u/27714199?v=4" title="tashachin" width="50" height="50"></a> <a href="https://github.com/gitter-badger" target="_blank"><img src="https://avatars.githubusercontent.com/u/8518239?v=4" title="gitter-badger" width="50" height="50"></a> <a href="https://github.com/TriveniBhat" target="_blank"><img src="https://avatars.githubusercontent.com/u/49990128?v=4" title="TriveniBhat" width="50" height="50"></a> <a href="https://github.com/virajprabhu" target="_blank"><img src="https://avatars.githubusercontent.com/u/8670301?v=4" title="virajprabhu" width="50" height="50"></a> <a href="https://github.com/vladan-jovicic" target="_blank"><img src="https://avatars.githubusercontent.com/u/8584694?v=4" title="vladan-jovicic" width="50" height="50"></a> <a href="https://github.com/Samyak-Jayaram" target="_blank"><img src="https://avatars.githubusercontent.com/u/120107046?v=4" title="Samyak-Jayaram" width="50" height="50"></a> <a href="https://github.com/rlee80" target="_blank"><img src="https://avatars.githubusercontent.com/u/46196529?v=4" title="rlee80" width="50" height="50"></a> <a href="https://github.com/Ru28" target="_blank"><img src="https://avatars.githubusercontent.com/u/54779977?v=4" title="Ru28" width="50" height="50"></a> <a href="https://github.com/rshrc" target="_blank"><img src="https://avatars.githubusercontent.com/u/28453217?v=4" title="rshrc" width="50" height="50"></a> <a href="https://github.com/Rishav09" target="_blank"><img src="https://avatars.githubusercontent.com/u/11032253?v=4" title="Rishav09" width="50" height="50"></a> <a href="https://github.com/pushkalkatara" target="_blank"><img src="https://avatars.githubusercontent.com/u/21266230?v=4" title="pushkalkatara" width="50" height="50"></a> <a href="https://github.com/PrasadCodesML" target="_blank"><img src="https://avatars.githubusercontent.com/u/113229283?v=4" title="PrasadCodesML" width="50" height="50"></a> <a href="https://github.com/prajwalgatti" target="_blank"><img src="https://avatars.githubusercontent.com/u/31077225?v=4" title="prajwalgatti" width="50" height="50"></a> <a href="https://github.com/jordanjfang" target="_blank"><img src="https://avatars.githubusercontent.com/u/19521127?v=4" title="jordanjfang" width="50" height="50"></a> <a href="https://github.com/GauravJain98" target="_blank"><img src="https://avatars.githubusercontent.com/u/19913130?v=4" title="GauravJain98" width="50" height="50"></a> <a href="https://github.com/yaskh" target="_blank"><img src="https://avatars.githubusercontent.com/u/41568177?v=4" title="yaskh" width="50" height="50"></a> <a href="https://github.com/WncFht" target="_blank"><img src="https://avatars.githubusercontent.com/u/63136734?v=4" title="WncFht" width="50" height="50"></a> <a href="https://github.com/weakit" target="_blank"><img src="https://avatars.githubusercontent.com/u/34541656?v=4" title="weakit" width="50" height="50"></a> <a href="https://github.com/thanos-pakou" target="_blank"><img src="https://avatars.githubusercontent.com/u/32770912?v=4" title="thanos-pakou" width="50" height="50"></a> <a href="https://github.com/rohitjha941" target="_blank"><img src="https://avatars.githubusercontent.com/u/33960527?v=4" title="rohitjha941" width="50" height="50"></a> <a href="https://github.com/praisejabraham" target="_blank"><img src="https://avatars.githubusercontent.com/u/31045180?v=4" title="praisejabraham" width="50" height="50"></a> <a href="https://github.com/newbazz" target="_blank"><img src="https://avatars.githubusercontent.com/u/25884863?v=4" title="newbazz" width="50" height="50"></a> <a href="https://github.com/mrMarce" target="_blank"><img src="https://avatars.githubusercontent.com/u/16017319?v=4" title="mrMarce" width="50" height="50"></a> <a href="https://github.com/lenixlobo" target="_blank"><img src="https://avatars.githubusercontent.com/u/20311706?v=4" title="lenixlobo" width="50" height="50"></a> <a href="https://github.com/jakecarr" target="_blank"><img src="https://avatars.githubusercontent.com/u/5979143?v=4" title="jakecarr" width="50" height="50"></a> <a href="https://github.com/ilyasd3" target="_blank"><img src="https://avatars.githubusercontent.com/u/77646344?v=4" title="ilyasd3" width="50" height="50"></a>

<br/>
Visit the [EvalAI repository](https://github.com/Cloud-CV/EvalAI) for more information and to
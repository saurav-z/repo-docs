<p align="center"><img width="65%" src="docs/source/_static/img/evalai_logo.png"/></p>

# EvalAI: The Open Source Platform for AI Challenge Evaluation

**EvalAI is a powerful, open-source platform designed to streamline the evaluation and comparison of machine learning and artificial intelligence algorithms.**  You can find the original repository [here](https://github.com/Cloud-CV/EvalAI).

[![Join the chat on Slack](https://img.shields.io/badge/Join%20Slack-Chat-blue?logo=slack)](https://join.slack.com/t/cloudcv-community/shared_invite/zt-3252n6or8-e0QuZKIZFLB0zXtQ6XgxfA)
[![Build Status](https://travis-ci.org/Cloud-CV/EvalAI.svg?branch=master)](https://travis-ci.org/Cloud-CV/EvalAI)
[![codecov](https://codecov.io/gh/Cloud-CV/EvalAI/branch/master/graph/badge.svg)](https://codecov.io/gh/Cloud-CV/EvalAI)
[![Coverage Status](https://coveralls.io/repos/github/Cloud-CV/EvalAI/badge.svg)](https://coveralls.io/github/Cloud-CV/EvalAI)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/markdown-guide/badge/?version=latest)](http://evalai.readthedocs.io/en/latest/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/Cloud-CV/EvalAI?style=flat-square)](https://github.com/Cloud-CV/EvalAI/tree/master)
[![Open Collective](https://opencollective.com/evalai/backers/badge.svg)](https://opencollective.com/evalai#backers)
[![Open Collective](https://opencollective.com/evalai/sponsors/badge.svg)](https://opencollective.com/evalai#sponsors)
[![Twitter Follow](https://img.shields.io/twitter/follow/eval_ai?style=social)](https://twitter.com/eval_ai)


## Key Features of EvalAI:

*   **Custom Evaluation Protocols:** Define flexible evaluation phases, dataset splits, and utilize any programming language for comprehensive results.
*   **Remote Evaluation:** Leverage specialized compute resources for resource-intensive challenges, with EvalAI managing the hosting, submissions, and leaderboards.
*   **Dockerized Evaluation:** Submit your AI agent as a Docker image, ensuring consistent evaluation within controlled environments.
*   **CLI Support:** Utilize the `evalai-cli` for enhanced command-line interaction, streamlining your workflow.
*   **Scalability & Portability:** Built with open-source technologies like Docker, Django, Node.js, and PostgreSQL for flexibility and easy deployment.
*   **Optimized Performance:** Experience faster evaluation times through worker node warm-up, dataset chunking, and multi-core processing.

## Goal

The ultimate goal of EvalAI is to build a centralized platform to host, participate, and collaborate in AI challenges worldwide, driving progress in the field of artificial intelligence.

## Installation

Installing EvalAI is simple using Docker:

1.  **Prerequisites:** Install [Docker](https://docs.docker.com/install/) and [Docker Compose](https://docs.docker.com/compose/install/).
2.  **Get the Code:** Clone the repository:

    ```bash
    git clone https://github.com/Cloud-CV/EvalAI.git evalai && cd evalai
    ```

3.  **Build and Run:** Execute the following command to build and start the containers:

    ```bash
    docker-compose up --build
    ```

    *   To include worker services, use: `docker-compose --profile worker up --build`
    *   To include statsd-exporter, use: `docker-compose --profile statsd up --build`
    *   To start both optional services, use: `docker-compose --profile worker --profile statsd up --build`

4.  **Access:** Open your web browser and go to [http://127.0.0.1:8888](http://127.0.0.1:8888).

    *   Default users are created:
        *   **SUPERUSER:** username: `admin`, password: `password`
        *   **HOST USER:** username: `host`, password: `password`
        *   **PARTICIPANT USER:** username: `participant`, password: `password`

    For help with installation issues, please see our [common errors during installation](https://evalai.readthedocs.io/en/latest/faq(developers).html#common-errors-during-installation) page.

## Documentation Setup

For documentation contributions, see the setup instructions in `docs/README.md`.

## Citing EvalAI

If you use EvalAI in your work, please cite the following technical report:

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
    <a href="http://learningsys.org/sosp19/assets/papers/23_CameraReadySubmission_EvalAI_SOSP_2019%20(8)%20(1).pdf"><img src="docs/source/_static/img/evalai-paper.jpg"/></a>
</p>

## Team

EvalAI is currently maintained by [Rishabh Jain](https://rishabhjain.xyz/) and [Gunjan Chhablani](https://gchhablani.github.io/).

## Contribution Guidelines

Contributions are welcome! Please follow the [contribution guidelines](https://github.com/Cloud-CV/EvalAI/blob/master/.github/CONTRIBUTING.md).

## Contributors

[//]: contributor-faces
<a href="https://github.com/RishabhJain2018"><img src="https://avatars.githubusercontent.com/u/12206047?v=4" title="RishabhJain2018" width="50" height="50"></a>
<a href="https://github.com/deshraj"><img src="https://avatars.githubusercontent.com/u/2945708?v=4" title="deshraj" width="50" height="50"></a>
<a href="https://github.com/Ram81"><img src="https://avatars.githubusercontent.com/u/16323427?v=4" title="Ram81" width="50" height="50"></a>
<a href="https://github.com/taranjeet"><img src="https://avatars.githubusercontent.com/u/4302268?v=4" title="taranjeet" width="50" height="50"></a>
<a href="https://github.com/Sanji515"><img src="https://avatars.githubusercontent.com/u/32524438?v=4" title="Sanji515" width="50" height="50"></a>
<a href="https://github.com/aka-jain"><img src="https://avatars.githubusercontent.com/u/11537940?v=4" title="aka-jain" width="50" height="50"></a>
<a href="https://github.com/Kajol-Kumari"><img src="https://avatars.githubusercontent.com/u/44888949?v=4" title="Kajol-Kumari" width="50" height="50"></a>
<a href="https://github.com/Ayukha"><img src="https://avatars.githubusercontent.com/u/19167324?v=4" title="Ayukha" width="50" height="50"></a>
<a href="https://github.com/spyshiv"><img src="https://avatars.githubusercontent.com/u/7015220?v=4" title="spyshiv" width="50" height="50"></a>
<a href="https://github.com/Arun-Jain"><img src="https://avatars.githubusercontent.com/u/16155501?v=4" title="Arun-Jain" width="50" height="50"></a>
<a href="https://github.com/vkartik97"><img src="https://avatars.githubusercontent.com/u/3920286?v=4" title="vkartik97" width="50" height="50"></a>
<a href="https://github.com/gautamjajoo"><img src="https://avatars.githubusercontent.com/u/24366008?v=4" title="gautamjajoo" width="50" height="50"></a>
<a href="https://github.com/KhalidRmb"><img src="https://avatars.githubusercontent.com/u/31621523?v=4" title="KhalidRmb" width="50" height="50"></a>
<a href="https://github.com/guyandtheworld"><img src="https://avatars.githubusercontent.com/u/20072816?v=4" title="guyandtheworld" width="50" height="50"></a>
<a href="https://github.com/anweshknayak"><img src="https://avatars.githubusercontent.com/u/8139783?v=4" title="anweshknayak" width="50" height="50"></a>
<a href="https://github.com/sanketbansal"><img src="https://avatars.githubusercontent.com/u/17106489?v=4" title="sanketbansal" width="50" height="50"></a>
<a href="https://github.com/live-wire"><img src="https://avatars.githubusercontent.com/u/6399428?v=4" title="live-wire" width="50" height="50"></a>
<a href="https://github.com/gauthamzz"><img src="https://avatars.githubusercontent.com/u/12110844?v=4" title="gauthamzz" width="50" height="50"></a>
<a href="https://github.com/HargovindArora"><img src="https://avatars.githubusercontent.com/u/22341493?v=4" title="HargovindArora" width="50" height="50"></a>
<a href="https://github.com/parismita"><img src="https://avatars.githubusercontent.com/u/17319560?v=4" title="parismita" width="50" height="50"></a>
<a href="https://github.com/dexter1691"><img src="https://avatars.githubusercontent.com/u/2039548?v=4" title="dexter1691" width="50" height="50"></a>
<a href="https://github.com/harshithdwivedi"><img src="https://avatars.githubusercontent.com/u/47669588?v=4" title="harshithdwivedi" width="50" height="50"></a>
<a href="https://github.com/nikochiko"><img src="https://avatars.githubusercontent.com/u/37668193?v=4" title="nikochiko" width="50" height="50"></a>
<a href="https://github.com/jayndu"><img src="https://avatars.githubusercontent.com/u/35180217?v=4" title="jayndu" width="50" height="50"></a>
<a href="https://github.com/matsujika"><img src="https://avatars.githubusercontent.com/u/57856193?v=4" title="matsujika" width="50" height="50"></a>
<a href="https://github.com/xamfy"><img src="https://avatars.githubusercontent.com/u/19357995?v=4" title="xamfy" width="50" height="50"></a>
<a href="https://github.com/drepram"><img src="https://avatars.githubusercontent.com/u/34530026?v=4" title="drepram" width="50" height="50"></a>
<a href="https://github.com/yadavankit"><img src="https://avatars.githubusercontent.com/u/8945824?v=4" title="yadavankit" width="50" height="50"></a>
<a href="https://github.com/MarchingVoxels"><img src="https://avatars.githubusercontent.com/u/28953079?v=4" title="MarchingVoxels" width="50" height="50"></a>
<a href="https://github.com/sachinmukherjee"><img src="https://avatars.githubusercontent.com/u/19318218?v=4" title="sachinmukherjee" width="50" height="50"></a>
<a href="https://github.com/kurianbenoy"><img src="https://avatars.githubusercontent.com/u/24592806?v=4" title="kurianbenoy" width="50" height="50"></a>
<a href="https://github.com/mayank-agarwal-96"><img src="https://avatars.githubusercontent.com/u/11095642?v=4" title="mayank-agarwal-96" width="50" height="50"></a>
<a href="https://github.com/codervivek"><img src="https://avatars.githubusercontent.com/u/26835119?v=4" title="codervivek" width="50" height="50"></a>
<a href="https://github.com/yashdusing"><img src="https://avatars.githubusercontent.com/u/19976688?v=4" title="yashdusing" width="50" height="50"></a>
<a href="https://github.com/jayantsa"><img src="https://avatars.githubusercontent.com/u/10354780?v=4" title="jayantsa" width="50" height="50"></a>
<a href="https://github.com/itaditya"><img src="https://avatars.githubusercontent.com/u/15871340?v=4" title="itaditya" width="50" height="50"></a>
<a href="https://github.com/geekayush"><img src="https://avatars.githubusercontent.com/u/22499864?v=4" title="geekayush" width="50" height="50"></a>
<a href="https://github.com/namansood"><img src="https://avatars.githubusercontent.com/u/5107795?v=4" title="namansood" width="50" height="50"></a>
<a href="https://github.com/ParthS007"><img src="https://avatars.githubusercontent.com/u/24358501?v=4" title="ParthS007" width="50" height="50"></a>
<a href="https://github.com/varunagrawal"><img src="https://avatars.githubusercontent.com/u/975964?v=4" title="varunagrawal" width="50" height="50"></a>
<a href="https://github.com/pavan-simplr"><img src="https://avatars.githubusercontent.com/u/66268853?v=4" title="pavan-simplr" width="50" height="50"></a>
<a href="https://github.com/aayusharora"><img src="https://avatars.githubusercontent.com/u/12194719?v=4" title="aayusharora" width="50" height="50"></a>
<a href="https://github.com/AnshulBasia"><img src="https://avatars.githubusercontent.com/u/12856392?v=4" title="AnshulBasia" width="50" height="50"></a>
<a href="https://github.com/burnerlee"><img src="https://avatars.githubusercontent.com/u/55936223?v=4" title="burnerlee" width="50" height="50"></a>
<a href="https://github.com/cwiggs"><img src="https://avatars.githubusercontent.com/u/5607419?v=4" title="cwiggs" width="50" height="50"></a>
<a href="https://github.com/tawAsh1"><img src="https://avatars.githubusercontent.com/u/7100187?v=4" title="tawAsh1" width="50" height="50"></a>
<a href="https://github.com/nagpalm7"><img src="https://avatars.githubusercontent.com/u/32512296?v=4" title="nagpalm7" width="50" height="50"></a>
<a href="https://github.com/parth-verma"><img src="https://avatars.githubusercontent.com/u/22412980?v=4" title="parth-verma" width="50" height="50"></a>
<a href="https://github.com/sarthak212"><img src="https://avatars.githubusercontent.com/u/33999269?v=4" title="sarthak212" width="50" height="50"></a>
<a href="https://github.com/Curious72"><img src="https://avatars.githubusercontent.com/u/8409274?v=4" title="Curious72" width="50" height="50"></a>
<a href="https://github.com/souravsingh"><img src="https://avatars.githubusercontent.com/u/4314261?v=4" title="souravsingh" width="50" height="50"></a>
<a href="https://github.com/Suryansh5545"><img src="https://avatars.githubusercontent.com/u/34577232?v=4" title="Suryansh5545" width="50" height="50"></a>
<a href="https://github.com/viditjain08"><img src="https://avatars.githubusercontent.com/u/5248993?v=4" title="viditjain08" width="50" height="50"></a>
<a href="https://github.com/dhruvbatra"><img src="https://avatars.githubusercontent.com/u/2941091?v=4" title="dhruvbatra" width="50" height="50"></a>
<a href="https://github.com/sehgalayush1"><img src="https://avatars.githubusercontent.com/u/9461113?v=4" title="sehgalayush1" width="50" height="50"></a>
<a href="https://github.com/Abhi58"><img src="https://avatars.githubusercontent.com/u/31471063?v=4" title="Abhi58" width="50" height="50"></a>
<a href="https://github.com/adamstafa"><img src="https://avatars.githubusercontent.com/u/34075272?v=4" title="adamstafa" width="50" height="50"></a>
<a href="https://github.com/S-ulphuric"><img src="https://avatars.githubusercontent.com/u/15248483?v=4" title="S-ulphuric" width="50" height="50"></a>
<a href="https://github.com/AliMirlou"><img src="https://avatars.githubusercontent.com/u/19661419?v=4" title="AliMirlou" width="50" height="50"></a>
<a href="https://github.com/Aliraza3997"><img src="https://avatars.githubusercontent.com/u/7720786?v=4" title="Aliraza3997" width="50" height="50"></a>
<a href="https://github.com/amanex007"><img src="https://avatars.githubusercontent.com/u/58567957?v=4" title="amanex007" width="50" height="50"></a>
<a href="https://github.com/anigasan"><img src="https://avatars.githubusercontent.com/u/29756847?v=4" title="anigasan" width="50" height="50"></a>
<a href="https://github.com/thisisashukla"><img src="https://avatars.githubusercontent.com/u/20864366?v=4" title="thisisashukla" width="50" height="50"></a>
<a href="https://github.com/anuyog1004"><img src="https://avatars.githubusercontent.com/u/25269846?v=4" title="anuyog1004" width="50" height="50"></a>
<a href="https://github.com/artkorenev"><img src="https://avatars.githubusercontent.com/u/5678669?v=4" title="artkorenev" width="50" height="50"></a>
<a href="https://github.com/Avikam03"><img src="https://avatars.githubusercontent.com/u/24971199?v=4" title="Avikam03" width="50" height="50"></a>
<a href="https://github.com/makoscafee"><img src="https://avatars.githubusercontent.com/u/6409210?v=4" title="makoscafee" width="50" height="50"></a>
<a href="https://github.com/brunowego"><img src="https://avatars.githubusercontent.com/u/441774?v=4" title="brunowego" width="50" height="50"></a>
<a href="https://github.com/calenrobinette"><img src="https://avatars.githubusercontent.com/u/30757528?v=4" title="calenrobinette" width="50" height="50"></a>
<a href="https://github.com/galipremsagar"><img src="https://avatars.githubusercontent.com/u/11664259?v=4" title="galipremsagar" width="50" height="50"></a>
<a href="https://github.com/GauravJain98"><img src="https://avatars.githubusercontent.com/u/19913130?v=4" title="GauravJain98" width="50" height="50"></a>
<a href="https://github.com/hizkifw"><img src="https://avatars.githubusercontent.com/u/7418049?v=4" title="hizkifw" width="50" height="50"></a>
<a href="https://github.com/kakshay21"><img src="https://avatars.githubusercontent.com/u/13598994?v=4" title="kakshay21" width="50" height="50"></a>
<a href="https://github.com/kaansan"><img src="https://avatars.githubusercontent.com/u/20618151?v=4" title="kaansan" width="50" height="50"></a>
<a href="https://github.com/lenniezelk"><img src="https://avatars.githubusercontent.com/u/13750897?v=4" title="lenniezelk" width="50" height="50"></a>
<a href="https://github.com/Marlysson"><img src="https://avatars.githubusercontent.com/u/4117999?v=4" title="Marlysson" width="50" height="50"></a>
<a href="https://github.com/Mateusz1223"><img src="https://avatars.githubusercontent.com/u/38505563?v=4" title="Mateusz1223" width="50" height="50"></a>
<a href="https://github.com/Mike-Dai"><img src="https://avatars.githubusercontent.com/u/43002285?v=4" title="Mike-Dai" width="50" height="50"></a>
<a href="https://github.com/zurda"><img src="https://avatars.githubusercontent.com/u/16784959?v=4" title="zurda" width="50" height="50"></a>
<a href="https://github.com/namish800"><img src="https://avatars.githubusercontent.com/u/25436663?v=4" title="namish800" width="50" height="50"></a>
<a href="https://github.com/neeraj12121"><img src="https://avatars.githubusercontent.com/u/15014361?v=4" title="neeraj12121" width="50" height="50"></a>
<a href="https://github.com/n-bernat"><img src="https://avatars.githubusercontent.com/u/23532372?v=4" title="n-bernat" width="50" height="50"></a>
<a href="https://github.com/inishchith"><img src="https://avatars.githubusercontent.com/u/20226361?v=4" title="inishchith" width="50" height="50"></a>
<a href="https://github.com/radiohazard-dev"><img src="https://avatars.githubusercontent.com/u/37482938?v=4" title="radiohazard-dev" width="50" height="50"></a>
<a href="https://github.com/Shashi456"><img src="https://avatars.githubusercontent.com/u/18056781?v=4" title="Shashi456" width="50" height="50"></a>
<a href="https://github.com/Plebtato"><img src="https://avatars.githubusercontent.com/u/19521127?v=4" title="Plebtato" width="50" height="50"></a>
<a href="https://github.com/pohzipohzi"><img src="https://avatars.githubusercontent.com/u/16781840?v=4" title="pohzipohzi" width="50" height="50"></a>
<a href="https://github.com/prajwalgatti"><img src="https://avatars.githubusercontent.com/u/31077225?v=4" title="prajwalgatti" width="50" height="50"></a>
<a href="https://github.com/pushkalkatara"><img src="https://avatars.githubusercontent.com/u/21266230?v=4" title="pushkalkatara" width="50" height="50"></a>
<a href="https://github.com/Rishav09"><img src="https://avatars.githubusercontent.com/u/11032253?v=4" title="Rishav09" width="50" height="50"></a>
<a href="https://github.com/rshrc"><img src="https://avatars.githubusercontent.com/u/28453217?v=4" title="rshrc" width="50" height="50"></a>
<a href="https://github.com/rlee80"><img src="https://avatars.githubusercontent.com/u/46196529?v=4" title="rlee80" width="50" height="50"></a>
<a href="https://github.com/sanyamdogra"><img src="https://avatars.githubusercontent.com/u/33497630?v=4" title="sanyamdogra" width="50" height="50"></a>
<a href="https://github.com/sayamkanwar"><img src="https://avatars.githubusercontent.com/u/10847009?v=4" title="sayamkanwar" width="50" height="50"></a>
<a href="https://github.com/shakeelsamsu"><img src="https://avatars.githubusercontent.com/u/16440459?v=4" title="shakeelsamsu" width="50" height="50"></a>
<a href="https://github.com/shiv6146"><img src="https://avatars.githubusercontent.com/u/5592146?v=4" title="shiv6146" width="50" height="50"></a>
<a href="https://github.com/bosecodes"><img src="https://avatars.githubusercontent.com/u/39362431?v=4" title="bosecodes" width="50" height="50"></a>
<a href="https://github.com/tashachin"><img src="https://avatars.githubusercontent.com/u/27714199?v=4" title="tashachin" width="50" height="50"></a>
<a href="https://github.com/gitter-badger"><img src="https://avatars.githubusercontent.com/u/8518239?v=4" title="gitter-badger" width="50" height="50"></a>
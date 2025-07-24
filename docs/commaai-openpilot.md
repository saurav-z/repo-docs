# openpilot: Open Source Driver Assistance System

**Transform your driving experience with openpilot, an open-source, community-driven driver assistance system that enhances safety and convenience in hundreds of supported vehicles.**  See the [original repository](https://github.com/commaai/openpilot) for more information.

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

[Video Demonstrations](https://youtu.be/NmBfgOanCyk) | [Video Demonstrations](https://youtu.be/VHKyqZ7t8Gw) | [Video Demonstrations](https://youtu.be/SUIZYzxtMQs)

## Key Features

*   **Advanced Driver Assistance:**  Enhances existing driver assistance features in compatible vehicles.
*   **Open Source:** Benefit from a transparent and collaborative development model.
*   **Community-Driven:**  Leverage the knowledge and contributions of a vibrant community.
*   **Extensive Vehicle Support:** Works with 300+ supported car models.
*   **Continuous Improvement:**  Benefit from regular updates and improvements through data-driven model training.

## Getting Started

### What you'll need:

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install openpilot via URL.
3.  **Supported Car:**  Check if your car is supported [here](docs/CARS.md).
4.  **Car Harness:**  A car harness is needed to connect your comma 3/3X to your car, available [here](https://comma.ai/shop/car-harness).

### Installation

Use the URL `openpilot.comma.ai` to install the release version. Detailed instructions are available [here](https://comma.ai/setup).

### Branches

| Branch           | URL                                    | Description                                                                         |
|------------------|----------------------------------------|-------------------------------------------------------------------------------------|
| `release3`         | openpilot.comma.ai                      | Release branch.                                                 |
| `release3-staging` | openpilot-test.comma.ai                | Staging branch for releases.                               |
| `nightly`          | openpilot-nightly.comma.ai             | Bleeding edge development branch (unstable).      |
| `nightly-dev`      | installer.comma.ai/commaai/nightly-dev | Experimental development features.      |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch from the autonomy team. |

## Contributing

openpilot is a community project, and contributions are welcome!

*   Join the [community Discord](https://discord.comma.ai)
*   Read the [contributing docs](docs/CONTRIBUTING.md)
*   Explore [openpilot tools](tools/)
*   Refer to the code documentation at https://docs.comma.ai
*   Find information about running openpilot on the [community wiki](https://github.com/commaai/openpilot/wiki)

## Careers & Bounties

*   Explore [job opportunities](https://comma.ai/jobs#open-positions) at comma.
*   Check out the [bounty program](https://comma.ai/bounties) for contributors.

## Safety and Testing

* openpilot observes [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines, see [SAFETY.md](docs/SAFETY.md) for more details.
* openpilot has software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) that run on every commit.
* The code enforcing the safety model lives in panda and is written in C, see [code rigor](https://github.com/commaai/panda#code-rigor) for more details.
* panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
* Internally, we have a hardware-in-the-loop Jenkins test suite that builds and unit tests the various processes.
* panda has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
* We run the latest openpilot in a testing closet containing 10 comma devices continuously replaying routes.

<details>
<summary>MIT License</summary>

(See original README for full text)
</details>

<details>
<summary>User Data and comma Account</summary>

(See original README for full text)
</details>
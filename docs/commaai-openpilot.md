# Openpilot: Your Open-Source Operating System for Driver Assistance

<div align="center" style="text-align: center;">

<p>
  <b>openpilot is an open-source, community-driven operating system that upgrades the driver assistance system in 300+ supported cars, bringing advanced features to the road.</b>
</p>

###
  <a href="https://docs.comma.ai">Docs</a>
  <span> · </span>
  <a href="https://docs.comma.ai/contributing/roadmap/">Roadmap</a>
  <span> · </span>
  <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Community</a>
  <span> · </span>
  <a href="https://comma.ai/shop">Try it on a comma 3X</a>
</h3>

</div>

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

<p align="center">
  <a href="https://github.com/commaai/openpilot">
    <img src="https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772" alt="openpilot Video 1" width="250">
  </a>
  <a href="https://github.com/commaai/openpilot">
    <img src="https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c" alt="openpilot Video 2" width="250">
  </a>
  <a href="https://github.com/commaai/openpilot">
    <img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ceefc6f63" alt="openpilot Video 3" width="250">
  </a>
</p>

## Key Features:

*   **Advanced Driver Assistance:** Upgrade your car with features like lane keeping, adaptive cruise control, and more.
*   **Open Source & Community Driven:** Benefit from a vibrant community of developers constantly improving and expanding openpilot's capabilities.
*   **Wide Car Compatibility:** Supports over 300 car models, with new vehicles being added regularly.
*   **Easy Installation:** Simple setup process for comma 3/3X devices.
*   **Continuous Improvement:** Regular updates and enhancements based on user feedback and data.

## Getting Started

### What You'll Need:

1.  **Compatible Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the latest release version by using the URL `openpilot.comma.ai` during the setup process.
3.  **Supported Car:** Ensure your car is on the [supported car list](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) to connect your device to your car.

### Installation Steps:

1.  **Purchase a comma 3/3X:** Get your device from [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Install the software:** Use the URL `openpilot.comma.ai` during setup.
3.  **Connect the Harness:** Follow the installation instructions for your car model on [comma.ai/setup](https://comma.ai/setup).

### Available Branches:

*   `release3`:  Stable release branch (`openpilot.comma.ai`)
*   `release3-staging`:  Staging branch for early access (`openpilot-test.comma.ai`)
*   `nightly`: Bleeding edge development branch (`openpilot-nightly.comma.ai`)
*   `nightly-dev`: Experimental development features (`installer.comma.ai/commaai/nightly-dev`)
*   `secretgoodopenpilot`: Preview branch with new driving models (`installer.comma.ai/commaai/secretgoodopenpilot`)

## Contributing

Openpilot thrives on community contributions! We welcome developers of all skill levels to join the project.

*   [Join the Community Discord](https://discord.comma.ai)
*   [Contribute to the project](docs/CONTRIBUTING.md)
*   Explore the [openpilot tools](tools/)
*   Find detailed documentation at https://docs.comma.ai
*   Get information from the community wiki at https://github.com/commaai/openpilot/wiki
*   [Comma is hiring](https://comma.ai/jobs#open-positions) and offers [bounties](https://comma.ai/bounties) for external contributors.

## Safety and Testing

*   Openpilot adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Software-in-the-loop tests run on every commit (.github/workflows/selfdrive_tests.yaml).
*   The safety model is written in C and lives in panda; see [code rigor](https://github.com/commaai/panda#code-rigor).
*   Panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Hardware-in-the-loop tests are run internally via a Jenkins test suite.
*   Panda has hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing with multiple comma devices replaying routes.

<details>
<summary>MIT License</summary>
... (License details as provided in the original README)
</details>

<details>
<summary>User Data and comma Account</summary>
... (User data and privacy details as provided in the original README)
</details>
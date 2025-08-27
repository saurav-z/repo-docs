<div align="center" style="text-align: center;">

<h1>openpilot: Open Source Driver Assistance for Your Car</h1>

<p>
  <b>Transform your driving experience with openpilot, a cutting-edge open-source driver-assistance system.</b>
  <br>
  Upgrade your vehicle with advanced features using this innovative technology.
</p>

<h3>
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

Quick start: `bash <(curl -fsSL openpilot.comma.ai)`

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

</div>

## Key Features

*   **Advanced Driver Assistance:** Enhance your car's capabilities with features like adaptive cruise control, lane keeping assist, and more.
*   **Open Source:** Benefit from a community-driven project, allowing for continuous improvements and customization.
*   **Wide Vehicle Support:** Compatible with 300+ supported car models (check the list [here](docs/CARS.md)).
*   **Easy Installation:** Get started quickly with a simple setup process using a comma 3/3X device.
*   **Continuous Updates:** Stay up-to-date with the latest features and improvements.

## Getting Started with openpilot

To use openpilot, you'll need the following:

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the openpilot release version using the URL `openpilot.comma.ai`.
3.  **Supported Car:** Ensure your car is on the [supported vehicles list](docs/CARS.md).
4.  **Car Harness:** You'll need a [car harness](https://comma.ai/shop/car-harness) to connect your device.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

## Branches

Choose the branch that best suits your needs:

*   `release3`: Stable release branch (`openpilot.comma.ai`)
*   `release3-staging`: Staging branch for early access (`openpilot-test.comma.ai`)
*   `nightly`: Bleeding-edge development branch (not stable) (`openpilot-nightly.comma.ai`)
*   `nightly-dev`: Experimental development features for some cars (`installer.comma.ai/commaai/nightly-dev`)

## Contributing to openpilot

Join the community and contribute to the development of openpilot!

*   [Join the community Discord](https://discord.comma.ai)
*   [Contribute to the project](docs/CONTRIBUTING.md)
*   Explore the [openpilot tools](tools/)
*   [Documentation](https://docs.comma.ai)
*   [Community Wiki](https://github.com/commaai/openpilot/wiki)
*   [Comma.ai Bounties](https://comma.ai/bounties)

## Safety and Testing

openpilot prioritizes safety through:

*   Adherence to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines ([SAFETY.md](docs/SAFETY.md)).
*   Automated software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml).
*   Rigorous code review and safety model implementation in panda (C).
*   Hardware-in-the-loop testing for enhanced reliability.

## License and Data Usage

<details>
<summary>MIT License</summary>

openpilot is licensed under the [MIT License](LICENSE). This section provides the full details.

</details>

<details>
<summary>User Data and Privacy</summary>

openpilot collects driving data to improve its models. Users can control data collection settings. Read the [Privacy Policy](https://comma.ai/privacy) for more information.

</details>

**[View the openpilot repository on GitHub](https://github.com/commaai/openpilot)**
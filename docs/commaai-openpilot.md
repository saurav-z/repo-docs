# openpilot: Open Source Driver Assistance System

**Transform your driving experience with openpilot, an open-source operating system that upgrades the driver assistance features in over 300 supported vehicles.**  For more details, see the original repository: [https://github.com/commaai/openpilot](https://github.com/commaai/openpilot).

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

## Key Features

*   **Enhanced Driver Assistance:** Upgrade your car's existing driver-assist features, adding lane keeping, adaptive cruise control, and more.
*   **Wide Vehicle Compatibility:** Works with over 300 supported car models.
*   **Open Source & Community Driven:** Benefit from community contributions and continuous improvements.
*   **Easy Installation:** Simple setup process using a comma 3/3X device.
*   **Data-Driven Development:** Data collected is used to improve the software with user consent.

## How to Get Started

To use openpilot, you'll need:

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version using the URL `openpilot.comma.ai`.
3.  **Supported Car:** Ensure your car is on the [supported car list](docs/CARS.md).
4.  **Car Harness:** Required for connection. Available at [comma.ai/shop/car-harness](https://comma.ai/shop/car-harness).

Detailed installation instructions are available at [https://comma.ai/setup](https://comma.ai/setup).

## Openpilot Branches

Choose the branch that best suits your needs:

| Branch                 | URL                                    | Description                                                                           |
| ---------------------- | -------------------------------------- | ------------------------------------------------------------------------------------- |
| `release3`             | openpilot.comma.ai                      | Stable release branch.                                                               |
| `release3-staging`     | openpilot-test.comma.ai                | Staging branch, get new releases slightly early.                                      |
| `nightly`              | openpilot-nightly.comma.ai             | Bleeding edge development branch; may be unstable.                                   |
| `nightly-dev`          | installer.comma.ai/commaai/nightly-dev | Includes experimental development features for some cars.                             |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch for autonomy team with new driving models getting merged earlier. |

## Contributing & Community

openpilot thrives on community contributions! Get involved by:

*   Joining the [community Discord](https://discord.comma.ai).
*   Checking the [contributing docs](docs/CONTRIBUTING.md).
*   Exploring the [openpilot tools](tools/).
*   Reviewing the [code documentation](https://docs.comma.ai).
*   Consulting the [community wiki](https://github.com/commaai/openpilot/wiki).
*   [comma is hiring](https://comma.ai/jobs#open-positions) and offers [bounties](https://comma.ai/bounties) for external contributors.

## Safety and Testing

openpilot is developed with safety in mind and follows these practices:

*   Adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Includes software-in-the-loop tests (see [.github/workflows/selfdrive_tests.yaml](.github/workflows/selfdrive_tests.yaml)).
*   Safety model code is written in C (see [code rigor](https://github.com/commaai/panda#code-rigor)).
*   Utilizes software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Includes a hardware-in-the-loop Jenkins test suite.

## Legal and Data Use

<details>
<summary>MIT License and Disclaimer</summary>

openpilot is released under the MIT license. See [LICENSE](LICENSE) for details.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**
</details>

<details>
<summary>User Data and Privacy</summary>

By default, openpilot uploads driving data to our servers to improve the software.  You can access your data through [comma connect](https://connect.comma.ai/).  You may disable data collection if desired.  By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy).
</details>
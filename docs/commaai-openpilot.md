# openpilot: Open Source Driver Assistance for Your Car

**Transform your driving experience with openpilot, an open-source driver assistance system that upgrades the capabilities of your car.** (See the original repo at [commaai/openpilot](https://github.com/commaai/openpilot) for more information.)

<div align="center">

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

</div>

## Key Features

*   **Driver Assistance Upgrade:** Enhance your vehicle's driver assistance capabilities with features like adaptive cruise control, lane keeping, and more.
*   **Open Source:** Benefit from community contributions and the transparency of open-source development.
*   **Wide Car Support:** Compatible with over 300 supported car models.
*   **Regular Updates:** Get access to the latest improvements and features through continuous development.
*   **Active Community:** Engage with a vibrant community for support, collaboration, and innovation.

## Getting Started

To begin using openpilot, you'll need a few key components:

*   **Supported Device:** A comma 3/3X device, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
*   **Software:** Install the openpilot release version using the URL `openpilot.comma.ai` during the setup of your comma 3/3X.
*   **Supported Car:** Ensure that your car is listed among the [275+ supported cars](docs/CARS.md).
*   **Car Harness:** A car harness is required to connect your comma 3/3X to your car. [Car harnesses are available for purchase](https://comma.ai/shop/car-harness).

For detailed installation instructions, please refer to [comma.ai/setup](https://comma.ai/setup).

## Branches

Choose the right branch for your needs:

| Branch              | URL                             | Description                                                                         |
| ------------------- | ------------------------------- | ----------------------------------------------------------------------------------- |
| `release3`          | openpilot.comma.ai                | Stable release branch.                                                             |
| `release3-staging`  | openpilot-test.comma.ai          | Staging branch for early access to upcoming releases.                               |
| `nightly`           | openpilot-nightly.comma.ai       | Bleeding-edge development branch; may be unstable.                                  |
| `nightly-dev`       | installer.comma.ai/commaai/nightly-dev | Includes experimental development features for select cars.                    |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch for new driving models from the autonomy team, merged earlier than master. |

## Contributing to openpilot

Contribute to the openpilot project and help shape the future of autonomous driving!

*   Join the [community Discord](https://discord.comma.ai) to connect with other developers and users.
*   Review the [contributing documentation](docs/CONTRIBUTING.md) for guidelines and best practices.
*   Explore the [openpilot tools](tools/) for development and debugging.
*   Access code documentation at https://docs.comma.ai.
*   Find more information about running openpilot on the [community wiki](https://github.com/commaai/openpilot/wiki).

## Safety and Testing

openpilot is developed with a strong emphasis on safety:

*   Follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Includes software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) for continuous validation.
*   Safety model code in panda is written in C; see [code rigor](https://github.com/commaai/panda#code-rigor).
*   `panda` has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Employs hardware-in-the-loop Jenkins test suite.
*   `panda` has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Undergoes rigorous testing with a continuous replay of routes using multiple devices.

<details>
<summary>MIT License</summary>

openpilot is released under the MIT license.

Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**
</details>

<details>
<summary>User Data and comma Account</summary>

openpilot collects driving data by default, which can be accessed via [comma connect](https://connect.comma.ai/). This data is used to improve the system, and users can disable data collection.

openpilot logs road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs. Driver-facing camera and microphone logging is optional.

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy). You understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma for the use of this data.
</details>
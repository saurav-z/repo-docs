# Openpilot: Revolutionizing Driver Assistance with Open Source Robotics

**Openpilot transforms your driving experience by upgrading the driver assistance systems in over 300 supported car models, bringing cutting-edge autonomous driving features to your fingertips.**

[Visit the original repo](https://github.com/commaai/openpilot)

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

## Key Features of Openpilot:

*   **Enhanced Driver Assistance:** Upgrade your car with advanced features like adaptive cruise control, lane keeping assist, and automatic lane changes.
*   **Open Source & Community Driven:** Benefit from a constantly evolving platform with contributions from a vibrant community of developers.
*   **Wide Car Support:** Compatible with over 300 car models, expanding the possibilities of driver assistance across various vehicles.
*   **Easy Installation:** Install and start using openpilot with a comma 3/3X device and the software from `openpilot.comma.ai`.
*   **Regular Updates:** Stay up-to-date with the latest advancements in autonomous driving through continuous development and releases.

## How to Get Started with Openpilot:

To begin using openpilot, you'll need a few essential components:

1.  **Supported Device:** A comma 3/3X device, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Use the URL `openpilot.comma.ai` in your device setup to install the latest release.
3.  **Supported Car:** Ensure your car is listed in the [supported cars](docs/CARS.md) document (over 275+).
4.  **Car Harness:** A compatible [car harness](https://comma.ai/shop/car-harness) to connect the device to your vehicle.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

## Openpilot Branches:

Choose the branch that suits your needs:

| Branch             | URL                           | Description                                                                         |
|--------------------|-------------------------------|-------------------------------------------------------------------------------------|
| `release3`           | openpilot.comma.ai            | The stable release branch.                                                 |
| `release3-staging`   | openpilot-test.comma.ai       | Get early access to upcoming releases.                                          |
| `nightly`            | openpilot-nightly.comma.ai    | The cutting-edge development branch; expect instability.                         |
| `nightly-dev`        | installer.comma.ai/commaai/nightly-dev | Contains experimental development features.                                     |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch with new driving models, merged earlier than master. |

## Contributing to Openpilot:

Join the open source revolution!  Contribute to the project by:

*   Joining the [community Discord](https://discord.comma.ai)
*   Reviewing the [contributing docs](docs/CONTRIBUTING.md)
*   Exploring the [openpilot tools](tools/)
*   Accessing the [code documentation](https://docs.comma.ai)
*   Checking the [community wiki](https://github.com/commaai/openpilot/wiki)

## Safety and Testing:

Openpilot prioritizes safety:

*   Adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Employs software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) on every commit.
*   The safety model code is in panda (C), see [code rigor](https://github.com/commaai/panda#code-rigor).
*   Panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Extensive hardware-in-the-loop testing in a Jenkins test suite.
*   Continuous testing with comma devices replaying routes.

<details>
<summary>MIT License</summary>

Openpilot is licensed under the MIT license. Please review the license terms and conditions, including disclaimers and limitations of liability.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**
</details>

<details>
<summary>User Data and Privacy</summary>

Openpilot uploads driving data to comma's servers for model training and improvement. You can access your data via [comma connect](https://connect.comma.ai/). Data collection can be disabled.

Logged data includes road-facing cameras, CAN, GPS, IMU, magnetometer, and operating system logs. Driver-facing camera and microphone data are only logged with user consent.

By using Openpilot, you agree to the [Privacy Policy](https://comma.ai/privacy), granting comma the right to use your data.
</details>
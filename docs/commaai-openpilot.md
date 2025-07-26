# openpilot: Open Source Driver Assistance System

**openpilot is an open-source, community-driven driver-assistance system that enhances the capabilities of over 300 supported car models.** ([Original Repository](https://github.com/commaai/openpilot))

<div align="center">

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

</div>

## Key Features

*   **Enhanced Driver Assistance:** Upgrade your car's existing driver assistance features.
*   **Wide Vehicle Compatibility:** Supports 300+ car models, with more being added continuously.
*   **Open Source & Community Driven:** Benefit from the collaborative efforts of the open-source community.
*   **Continuous Improvement:** Data from users is used to train and improve the system.

## Getting Started

### Requirements:

1.  **Supported Device:** A comma 3/3X device is required, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Use the URL `openpilot.comma.ai` when setting up your comma 3/3X to install the release version.
3.  **Supported Car:** Verify compatibility with [the list of supported cars](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) is needed to connect your comma 3/3X.

Detailed instructions on how to install the harness and device can be found at [comma.ai/setup](https://comma.ai/setup).

### Software Branches

Choose the branch that is best for your usage:

| Branch           | URL                       | Description                                                                               |
| ---------------- | ------------------------- | ----------------------------------------------------------------------------------------- |
| `release3`         | openpilot.comma.ai       | Stable release branch.                                                                     |
| `release3-staging` | openpilot-test.comma.ai  | Staging branch for pre-release testing.                                                  |
| `nightly`          | openpilot-nightly.comma.ai | Bleeding edge development branch.  May be unstable.                                       |
| `nightly-dev`      | installer.comma.ai/commaai/nightly-dev | Development features for some cars.                                       |
| `secretgoodopenpilot`      | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch from the autonomy team where new driving models get merged earlier than master.                                       |

## Contributing to openpilot

openpilot thrives on community contributions.

*   Join the [community Discord](https://discord.comma.ai)
*   Consult the [contributing docs](docs/CONTRIBUTING.md)
*   Explore the [openpilot tools](tools/)
*   Find code documentation at https://docs.comma.ai
*   Check out the [community wiki](https://github.com/commaai/openpilot/wiki)

## Safety and Testing

openpilot prioritizes safety with:

*   Adherence to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) on every commit.
*   Rigorous code enforcement in `panda` (C code).
*   Hardware-in-the-loop and software-in-the-loop safety tests for `panda`.

## Disclaimer

<details>
<summary>MIT License and User Data</summary>

openpilot is released under the MIT license.  Use of this software is at your own risk. Comma.ai, Inc. and its affiliates are not liable for any use of this software.
By default, openpilot uploads the driving data to our servers. You can also access your data through [comma connect](https://connect.comma.ai/). We use your data to train better models and improve openpilot for everyone.
openpilot logs the road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
The driver-facing camera and microphone are only logged if you explicitly opt-in in settings.
By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy).

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**
</details>
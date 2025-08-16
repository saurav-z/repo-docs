# openpilot: Open Source Driver Assistance System

**Enhance your driving experience with openpilot, an open-source driver assistance system currently upgrading the driver assistance in 300+ supported cars.** Explore the possibilities of autonomous driving with openpilot, developed by [comma.ai](https://comma.ai/) and a vibrant community. ([Original Repo](https://github.com/commaai/openpilot))

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

## Key Features

*   **Advanced Driver Assistance:** Upgrade your vehicle's driver assistance features with cutting-edge technology.
*   **Open Source & Community Driven:** Benefit from a collaborative ecosystem of developers and users.
*   **Wide Vehicle Compatibility:** Supports over 300 car models, with more being added regularly.
*   **Continuous Improvement:** Benefit from a model trained on real-world data.

## Getting Started with openpilot

To get started with openpilot, you will need the following:

1.  **Compatible Hardware:** A [comma 3/3X](https://comma.ai/shop/comma-3x) device.
2.  **Software Installation:** Install openpilot using the URL `openpilot.comma.ai` during the setup of your comma 3/3X.
3.  **Supported Vehicle:** Ensure your car is listed among [the 275+ supported cars](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) to connect your device to your vehicle.

Detailed instructions are available at [comma.ai/setup](https://comma.ai/setup).

## Branches
| branch           | URL                                    | description                                                                         |
|------------------|----------------------------------------|-------------------------------------------------------------------------------------|
| `release3`         | openpilot.comma.ai                      | This is openpilot's release branch.                                                 |
| `release3-staging` | openpilot-test.comma.ai                | This is the staging branch for releases. Use it to get new releases slightly early. |
| `nightly`          | openpilot-nightly.comma.ai             | This is the bleeding edge development branch. Do not expect this to be stable.      |
| `nightly-dev`      | installer.comma.ai/commaai/nightly-dev | Same as nightly, but includes experimental development features for some cars.      |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | This is a preview branch from the autonomy team where new driving models get merged earlier than master. |

## Contributing to openpilot

openpilot is a community project. Contributions are welcome!

*   Join the [community Discord](https://discord.comma.ai).
*   Review the [contributing docs](docs/CONTRIBUTING.md).
*   Explore the [openpilot tools](tools/).
*   Find code documentation at https://docs.comma.ai.
*   Find information about running openpilot on the [community wiki](https://github.com/commaai/openpilot/wiki).
*   [comma is hiring](https://comma.ai/jobs#open-positions) and offers [bounties](https://comma.ai/bounties) for external contributors.

## Safety and Testing

openpilot is committed to safety.
*   openpilot observes [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines. See [SAFETY.md](docs/SAFETY.md) for details.
*   Automated software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) run on every commit.
*   Code enforcing the safety model lives in panda and is written in C.  See [code rigor](https://github.com/commaai/panda#code-rigor) for more details.
*   panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   We have a hardware-in-the-loop Jenkins test suite that builds and unit tests various processes.
*   panda has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   The latest openpilot runs continuously in a testing environment.

<details>
<summary>MIT License</summary>

openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.

Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**
</details>

<details>
<summary>User Data and comma Account</summary>

By default, openpilot uploads the driving data to our servers. You can also access your data through [comma connect](https://connect.comma.ai/). We use your data to train better models and improve openpilot for everyone.

openpilot is open source software: the user is free to disable data collection if they wish to do so.

openpilot logs the road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
The driver-facing camera and microphone are only logged if you explicitly opt-in in settings.

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy). You understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma for the use of this data.
</details>
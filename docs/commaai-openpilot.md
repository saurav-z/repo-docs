# openpilot: Open Source Driver Assistance System

**Upgrade your driving experience with openpilot, an open-source driving assistance system that enhances the capabilities of supported vehicles.** Check out the original openpilot repo on [GitHub](https://github.com/commaai/openpilot) for more details.

<div align="center">

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

</div>

## Key Features

*   **Driver Assistance Upgrade:** Enhance your vehicle's driver assistance system with advanced features.
*   **Open Source:** Benefit from the transparency and community-driven development of open-source software.
*   **Wide Vehicle Support:** Currently supports over 300+ car models.
*   **Continuous Improvement:** Data-driven model training and ongoing development for improved performance.
*   **Community Driven:** Active community and developer support through Discord and GitHub.

## Getting Started

To use openpilot, you'll need the following:

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install openpilot on your comma 3/3X by entering the URL `openpilot.comma.ai` in the device setup.
3.  **Supported Car:** Verify your car is supported by checking the list of [supported cars](docs/CARS.md).
4.  **Car Harness:** Obtain the appropriate [car harness](https://comma.ai/shop/car-harness) for your vehicle.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

### Available Branches

Choose the branch that best suits your needs:

| Branch             | URL                                      | Description                                                                          |
| ------------------ | ---------------------------------------- | ------------------------------------------------------------------------------------- |
| `release3`           | openpilot.comma.ai                      | Stable release branch.                                                               |
| `release3-staging`   | openpilot-test.comma.ai                  | Staging branch for early access to new releases.                                      |
| `nightly`            | openpilot-nightly.comma.ai               | Bleeding-edge development branch; may be unstable.                                    |
| `nightly-dev`        | installer.comma.ai/commaai/nightly-dev   | Same as nightly, includes experimental development features for some cars.             |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch from the autonomy team with the latest driving models, merged earlier than master. |

## Developing and Contributing

openpilot is a community project. We welcome your contributions!

*   Join the [community Discord](https://discord.comma.ai).
*   Review the [contributing docs](docs/CONTRIBUTING.md).
*   Explore the [openpilot tools](tools/).
*   Access code documentation at https://docs.comma.ai.
*   Find more information on the [community wiki](https://github.com/commaai/openpilot/wiki).

[comma](https://comma.ai/) offers [bounties](https://comma.ai/bounties) and employment opportunities ([comma.ai/jobs#open-positions](https://comma.ai/jobs#open-positions)) for contributors.

## Safety and Testing

openpilot is developed with a strong emphasis on safety:

*   Adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Automated software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) are run on every commit.
*   The safety model is enforced by code written in C (see [code rigor](https://github.com/commaai/panda#code-rigor)).
*   `panda` includes software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Internal hardware-in-the-loop Jenkins test suite.
*   `panda` includes additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing with multiple comma devices.

<details>
<summary>MIT License</summary>

[openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.

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
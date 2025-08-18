# openpilot: Open Source Driver Assistance System

**Upgrade your driving experience with openpilot, an open-source, AI-powered driver-assistance system that enhances safety and convenience in hundreds of supported vehicles.**  Learn more and contribute to the project at the [original repository](https://github.com/commaai/openpilot).

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

## Key Features

*   **Enhanced Driver Assistance:** Improves existing driver-assistance features in compatible vehicles.
*   **Open Source:**  Contribute to and customize the system's functionality and development.
*   **Community Driven:** Active community with documentation, support, and collaboration.
*   **Continuous Updates:** Benefit from ongoing development, improvements, and new features.
*   **Wide Vehicle Support:** Compatible with 275+ supported car models.
*   **Easy Installation:** Quick setup process to get you driving with enhanced features.

## Getting Started

To use openpilot, you'll need:

1.  **Supported Device:**  A comma 3/3X device ([comma.ai/shop](https://comma.ai/shop/comma-3x)).
2.  **Software:**  Enter the URL `openpilot.comma.ai` in your device to install the release version.
3.  **Supported Car:** Verify compatibility with your vehicle ([docs/CARS.md](docs/CARS.md)).
4.  **Car Harness:** A compatible car harness ([comma.ai/shop/car-harness](https://comma.ai/shop/car-harness)).

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

### Branches

Access different versions of openpilot based on your needs:

| Branch              | URL                                    | Description                                                                         |
|---------------------|----------------------------------------|-------------------------------------------------------------------------------------|
| `release3`          | openpilot.comma.ai                      | Release branch.                                                 |
| `release3-staging`  | openpilot-test.comma.ai                | Staging branch for early releases. Use it to get new releases slightly early. |
| `nightly`           | openpilot-nightly.comma.ai             | Bleeding edge development branch. Not stable.      |
| `nightly-dev`       | installer.comma.ai/commaai/nightly-dev | Same as nightly, but includes experimental development features for some cars.      |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch from the autonomy team where new driving models get merged earlier than master. |

## Contributing

openpilot is a community-driven project.  We welcome contributions!

*   Join the [community Discord](https://discord.comma.ai)
*   Contribute by checking out the [contributing docs](docs/CONTRIBUTING.md)
*   Explore [openpilot tools](tools/)
*   Find code documentation at https://docs.comma.ai
*   Read information about running openpilot on the [community wiki](https://github.com/commaai/openpilot/wiki)

## Safety and Testing

openpilot is developed with safety in mind.

*   Follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines ([SAFETY.md](docs/SAFETY.md)).
*   Software-in-the-loop tests ([.github/workflows/selfdrive_tests.yaml](.github/workflows/selfdrive_tests.yaml)) run on every commit.
*   Safety model code in `panda` is written in C ([code rigor](https://github.com/commaai/panda#code-rigor)).
*   `panda` features software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Hardware-in-the-loop Jenkins test suite for builds and unit tests.
*   `panda` has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing with multiple devices replaying routes.

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
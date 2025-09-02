# Openpilot: The Open-Source Operating System for Smarter Driving

**Upgrade your car's driver assistance system and experience the future of driving with openpilot, an open-source platform enhancing safety and convenience on the road.** Learn more about openpilot on the [original repository](https://github.com/commaai/openpilot).

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

**Key Features:**

*   **Driver Assistance Upgrade:** Enhances existing driver-assistance systems in 300+ supported car models.
*   **Open Source & Community Driven:**  Developed by comma and the open-source community, fostering collaboration and continuous improvement.
*   **Regular Updates:** Benefit from ongoing development, bug fixes, and feature enhancements.
*   **Data-Driven Improvement:** Uses user data to train better models and improve the platform's performance.
*   **Comprehensive Safety Measures:** Adheres to ISO26262 guidelines with rigorous testing and safety protocols.

### Getting Started

To use openpilot, you'll need the following:

1.  **Supported Device:** A comma 3X device, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:**  Install the release version by entering the URL `openpilot.comma.ai` during setup.
3.  **Supported Car:**  Ensure your car is on the [list of supported vehicles](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) to connect your device.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

### Branches

*   `release3`: Stable release branch (`openpilot.comma.ai`)
*   `release3-staging`: Staging branch for early access to new releases (`openpilot-test.comma.ai`)
*   `nightly`: Bleeding-edge development branch (unstable) (`openpilot-nightly.comma.ai`)
*   `nightly-dev`: Development branch with experimental features (`installer.comma.ai/commaai/nightly-dev`)

### Contributing

*   Join the [community Discord](https://discord.comma.ai)
*   Explore the [contributing docs](docs/CONTRIBUTING.md)
*   Check out the [openpilot tools](tools/)
*   Find code documentation at [https://docs.comma.ai](https://docs.comma.ai)
*   Learn more about running openpilot on the [community wiki](https://github.com/commaai/openpilot/wiki)

[comma is hiring](https://comma.ai/jobs#open-positions) and offers [bounties](https://comma.ai/bounties) for contributors.

### Safety and Testing

*   openpilot adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) run with every commit.
*   Safety-critical code is written in C (see [code rigor](https://github.com/commaai/panda#code-rigor)).
*   [Safety tests](https://github.com/commaai/panda/tree/master/tests/safety) exist in the `panda` repository.
*   Hardware-in-the-loop tests using Jenkins.
*   Continuous testing with multiple devices replaying routes.

<details>
<summary>MIT License</summary>

Openpilot is released under the MIT license.  Read the full details in the [LICENSE](LICENSE) file.

**DISCLAIMER:**
This is alpha quality software for research purposes only.  This is not a product.
You are responsible for complying with local laws and regulations.
NO WARRANTY EXPRESSED OR IMPLIED.
</details>

<details>
<summary>User Data and comma Account</summary>

By default, openpilot uploads driving data to comma's servers. You can access your data through [comma connect](https://connect.comma.ai/).  This data is used to improve openpilot.

Users can disable data collection.

openpilot logs various data including cameras, CAN data, GPS, IMU, and operating system logs. The driver-facing camera and microphone are logged if you opt-in.

By using openpilot, you agree to the [Privacy Policy](https://comma.ai/privacy).  You grant an irrevocable, perpetual right to comma for the use of this data.
</details>
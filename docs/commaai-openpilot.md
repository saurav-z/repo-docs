<div align="center" style="text-align: center;">

<h1>openpilot: Open Source Driver Assistance System</h1>

<p>
  <b>Upgrade your car's driver assistance with openpilot, an open-source operating system for robotics.</b>
  <br>
  Enhance your driving experience with advanced features for over 300 supported car models.
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

---

## Openpilot: Revolutionizing Driver Assistance

Openpilot is an open-source, community-driven driver assistance system, offering advanced features like adaptive cruise control and lane keeping assist.  [Learn more and contribute on GitHub](https://github.com/commaai/openpilot).

---

## Key Features

*   **Enhanced Driver Assistance:** Upgrade your car with advanced features.
*   **Open Source:** Fully open-source, allowing for community contributions and customization.
*   **Wide Compatibility:** Supports over 300+ car models.
*   **Easy Installation:** Simple setup with the comma 3X device.
*   **Active Community:** Benefit from a vibrant community for support and collaboration.

---

## Getting Started

To use openpilot, you will need:

1.  **Supported Device:** A comma 3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install openpilot by entering the URL `openpilot.comma.ai` during the setup of your comma 3X.
3.  **Supported Car:** Ensure your car model is on the [list of supported cars](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) is required to connect the comma 3X to your car.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

---

## Branches

Choose the right branch for your needs:

| Branch            | URL                             | Description                                                        |
| ----------------- | ------------------------------- | ------------------------------------------------------------------ |
| `release3`        | openpilot.comma.ai              | Stable release branch.                                             |
| `release3-staging`| openpilot-test.comma.ai         | Staging branch for early access to new releases.                   |
| `nightly`         | openpilot-nightly.comma.ai      | Bleeding-edge development branch; may be unstable.                |
| `nightly-dev`     | installer.comma.ai/commaai/nightly-dev | Development branch with experimental features (for select cars). |

---

## Contributing & Community

Openpilot thrives on community contributions!

*   Join the [community Discord](https://discord.comma.ai).
*   Review the [contributing docs](docs/CONTRIBUTING.md).
*   Explore the [openpilot tools](tools/).
*   Access the code documentation at [https://docs.comma.ai](https://docs.comma.ai).
*   Find information about running openpilot on the [community wiki](https://github.com/commaai/openpilot/wiki).

Interested in working on openpilot professionally? [Comma.ai is hiring](https://comma.ai/jobs#open-positions), with bounties available for external contributors.

---

## Safety and Testing

*   openpilot adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Automated software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) run with every commit.
*   The safety model is enforced by code in `panda`, written in C (see [code rigor](https://github.com/commaai/panda#code-rigor)).
*   `panda` includes software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Internal hardware-in-the-loop Jenkins test suite for unit tests.
*   `panda` has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing with 10 comma devices replaying routes in a testing closet.

---

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

By default, openpilot uploads driving data to our servers.  Access your data through [comma connect](https://connect.comma.ai/). This data is used to train better models and improve openpilot for all users.

openpilot is open-source and allows users to disable data collection.

The system logs: road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs. The driver-facing camera and microphone are only logged if you explicitly opt-in in settings.

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy), and understand that usage generates user data, which may be logged and stored by comma.  You grant an irrevocable, perpetual, worldwide right to comma for the use of this data.
</details>
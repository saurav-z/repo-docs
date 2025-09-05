<div align="center">

<h1>openpilot: Open Source Driver Assistance System</h1>

<p>
  <b>Transform your driving experience with openpilot, an open-source driver-assistance system that enhances safety and convenience for supported vehicles.</b>
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

## What is openpilot?

openpilot is an open-source, community-driven driver-assistance system that adds advanced features like adaptive cruise control, lane keeping, and automatic lane centering to a growing list of supported vehicles.  [Explore the openpilot project on GitHub](https://github.com/commaai/openpilot).

## Key Features

*   **Adaptive Cruise Control:** Maintains a safe distance from vehicles ahead.
*   **Lane Keeping Assist:** Keeps your vehicle centered in its lane.
*   **Automatic Lane Centering:** steers the car to stay centered in the lane
*   **Open Source & Community Driven:** Benefit from continuous improvements and contributions from a global community.
*   **Supports 300+ Vehicles:** Enhances driver assistance systems in a wide range of supported cars.
*   **Easy Installation:**  Install openpilot on your car by obtaining a [comma 3X](https://comma.ai/shop/comma-3x) and following the setup instructions.

## How to Get Started

To use openpilot, you'll need:

1.  **Supported Device:** A [comma 3X](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version using the URL `openpilot.comma.ai`.
3.  **Supported Car:** Ensure your vehicle is one of the [275+ supported cars](docs/CARS.md).
4.  **Car Harness:**  A [car harness](https://comma.ai/shop/car-harness) to connect your comma 3X to your car.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

## Branches

| Branch           | URL                                    | Description                                                                         |
|------------------|----------------------------------------|-------------------------------------------------------------------------------------|
| `release3`         | openpilot.comma.ai                      | Stable release branch.                                                 |
| `release3-staging` | openpilot-test.comma.ai                | Staging branch for early access to new releases. |
| `nightly`          | openpilot-nightly.comma.ai             | Bleeding-edge development branch; may be unstable.      |
| `nightly-dev`      | installer.comma.ai/commaai/nightly-dev | Development branch including experimental features for some cars.      |

## Contributing to openpilot

openpilot thrives on community contributions.  Join us!

*   [Community Discord](https://discord.comma.ai)
*   [Contributing Docs](docs/CONTRIBUTING.md)
*   [openpilot Tools](tools/)
*   [Code Documentation](https://docs.comma.ai)
*   [Community Wiki](https://github.com/commaai/openpilot/wiki)

[Comma.ai](https://comma.ai/) offers [bounties](https://comma.ai/bounties) for contributions.

## Safety and Testing

openpilot prioritizes safety through:

*   Compliance with [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines ([SAFETY.md](docs/SAFETY.md)).
*   Automated software-in-the-loop tests ([.github/workflows/selfdrive_tests.yaml](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)).
*   Rigorous code review and safety mechanisms in panda ([code rigor](https://github.com/commaai/panda#code-rigor), [safety tests](https://github.com/commaai/panda/tree/master/tests/safety)).
*   Hardware-in-the-loop testing.
*   Continuous testing on comma devices.

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
<summary>User Data and Comma Account</summary>

By default, openpilot uploads driving data to our servers. You can access your data through [comma connect](https://connect.comma.ai/). We use your data to train better models and improve openpilot for everyone.

openpilot is open-source software: the user is free to disable data collection if they wish to do so.

openpilot logs the road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
The driver-facing camera and microphone are only logged if you explicitly opt-in in settings.

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy). You understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma for the use of this data.
</details>
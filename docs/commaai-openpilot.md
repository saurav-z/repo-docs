<div align="center" style="text-align: center;">

<h1>openpilot</h1>

<p>
  <b>Transform your driving experience with openpilot, the open-source autonomous driving system that's revolutionizing driver assistance.</b>
  <br>
  Currently, it upgrades the driver assistance system in 300+ supported cars.
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

## What is openpilot?

[openpilot](https://github.com/commaai/openpilot) is an open-source, advanced driver-assistance system (ADAS). It's designed to enhance the capabilities of your car's existing driver-assistance features, offering features like adaptive cruise control, lane keeping assist, and more, all in an open and customizable platform.

## Key Features

*   **Open Source:**  Built and maintained by a vibrant community, offering transparency and continuous improvement.
*   **Wide Car Support:** Works with over 300 supported car models, with new models frequently added.
*   **Enhanced Driver Assistance:**  Provides advanced features such as adaptive cruise control, lane keeping assist, and automatic lane changes.
*   **Community Driven:** Benefit from a supportive community, constantly improving and expanding openpilot's capabilities.
*   **Customizable:**  Tailor the system to your preferences and contribute to its development.
*   **Regular Updates:** Benefit from frequent updates, incorporating the latest advancements and improvements.

## Getting Started

To start using openpilot, you'll need the following:

1.  **Supported Device:** A [comma 3X](https://comma.ai/shop/comma-3x) is required.
2.  **Software:**  Use the URL `openpilot.comma.ai` to install the release version.
3.  **Supported Car:** Check if your car is on the [supported car list](docs/CARS.md).
4.  **Car Harness:**  You'll also need a [car harness](https://comma.ai/shop/car-harness) to connect your comma 3X to your car.

Detailed instructions for [installation](https://comma.ai/setup) are available on the comma.ai website.

## Openpilot Branches

Choose the branch that suits your needs:

| Branch          | URL                     | Description                                                |
| --------------- | ----------------------- | ---------------------------------------------------------- |
| `release3`        | openpilot.comma.ai       | The stable release branch.                                 |
| `release3-staging` | openpilot-test.comma.ai  | Staging branch for early access to new releases.          |
| `nightly`         | openpilot-nightly.comma.ai | Bleeding-edge development branch; may be unstable.         |
| `nightly-dev`     | installer.comma.ai/commaai/nightly-dev | Nightly with experimental features for some cars. |

## Contribute to openpilot

openpilot thrives on community contributions.  Join the effort and help shape the future of autonomous driving!

*   Join the [community Discord](https://discord.comma.ai)
*   Review the [contributing docs](docs/CONTRIBUTING.md)
*   Explore the [openpilot tools](tools/)
*   Access code documentation at https://docs.comma.ai
*   Find information on the [community wiki](https://github.com/commaai/openpilot/wiki)
*   Explore [comma's job openings](https://comma.ai/jobs#open-positions) and [bounty program](https://comma.ai/bounties) for contributors.

## Safety and Testing

openpilot is designed with safety in mind.

*   Adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines, see [SAFETY.md](docs/SAFETY.md).
*   Includes software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) that run on every commit.
*   The safety model is enforced in `panda` and written in C, see [code rigor](https://github.com/commaai/panda#code-rigor) for more details.
*   `panda` has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Internal hardware-in-the-loop Jenkins test suite.
*   `panda` has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing with 10 comma devices.

<details>
<summary>MIT Licensed</summary>

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
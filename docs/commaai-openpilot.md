<div align="center" style="text-align: center;">

<h1>openpilot: Open Source Driver Assistance System</h1>

<p>
  <b>openpilot is an open-source, community-driven operating system for advanced driver-assistance systems (ADAS), enhancing the capabilities of supported vehicles.</b>
  <br>
  Transforming your driving experience, openpilot currently upgrades driver assistance systems in over 300 supported car models.
  <br>
  <a href="https://github.com/commaai/openpilot">View the openpilot GitHub repository</a>
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

<table>
  <tr>
    <td><a href="https://youtu.be/NmBfgOanCyk" title="Video By Greer Viau"><img src="https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772"></a></td>
    <td><a href="https://youtu.be/VHKyqZ7t8Gw" title="Video By Logan LeGrand"><img src="https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c"></a></td>
    <td><a href="https://youtu.be/SUIZYzxtMQs" title="A drive to Taco Bell"><img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ce97dc6f63"></a></td>
  </tr>
</table>

## Key Features

*   **Open Source:** Benefit from a community-driven project and the transparency of open-source development.
*   **ADAS Upgrade:** Enhance driver assistance features in your car, with a constantly growing list of supported vehicles.
*   **Community Driven:** Engage with a vibrant community for support, development, and collaboration.
*   **Easy Installation:** Install openpilot on a supported device with a straightforward setup process.
*   **Continuous Improvement:** Benefit from ongoing updates, improvements, and new features.

## Getting Started

### Requirements

To use openpilot in your car, you'll need:

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Use the URL `openpilot.comma.ai` during setup to install the release version on your comma 3/3X.
3.  **Supported Car:** Verify your car is compatible with [the 275+ supported cars](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) is required to connect your comma 3/3X to your car.

### Installation

Detailed instructions for installing the harness and device are available at [comma.ai/setup](https://comma.ai/setup). Note that running openpilot on [other hardware](https://blog.comma.ai/self-driving-car-for-free/) is possible but not plug-and-play.

### Branches

| Branch            | URL                                    | Description                                                                        |
| ----------------- | ---------------------------------------- | ----------------------------------------------------------------------------------- |
| `release3`        | openpilot.comma.ai                      | The release branch.                                                              |
| `release3-staging` | openpilot-test.comma.ai                | Staging branch for pre-release testing.                                             |
| `nightly`         | openpilot-nightly.comma.ai             | Bleeding edge development branch; may be unstable.                                  |
| `nightly-dev`     | installer.comma.ai/commaai/nightly-dev | Includes experimental development features for some cars.                           |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch with new driving models, merged earlier than master. |

## Contributing

openpilot thrives on community contributions.

*   Join the [community Discord](https://discord.comma.ai)
*   Review the [contributing docs](docs/CONTRIBUTING.md)
*   Explore the [openpilot tools](tools/)
*   Find code documentation at https://docs.comma.ai
*   Learn more on the [community wiki](https://github.com/commaai/openpilot/wiki)

## Jobs and Bounties

Interested in contributing to openpilot professionally?

*   [Comma.ai is hiring](https://comma.ai/jobs#open-positions)
*   Explore [bounties](https://comma.ai/bounties) for external contributors.

## Safety and Testing

*   openpilot adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Software-in-the-loop tests run on every commit ([.github/workflows/selfdrive_tests.yaml](.github/workflows/selfdrive_tests.yaml)).
*   The safety model's code is in panda, written in C; see [code rigor](https://github.com/commaai/panda#code-rigor).
*   panda includes software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Internal testing uses a hardware-in-the-loop Jenkins test suite.
*   panda has hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing involves multiple comma devices replaying routes in a testing environment.

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
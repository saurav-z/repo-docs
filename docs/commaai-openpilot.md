<div align="center">

<h1>openpilot: Open Source Driver Assistance System</h1>

<p>
  <b>Unlock advanced driver-assistance features in your car with openpilot, the open-source driving agent.</b>
  <br>
  Upgrade your driving experience with openpilot, currently supporting 300+ vehicle models.
  <br>
  <a href="https://github.com/commaai/openpilot">Check out the original repo!</a>
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

<br>

<table>
  <tr>
    <td><a href="https://youtu.be/NmBfgOanCyk" title="Video By Greer Viau"><img src="https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772"></a></td>
    <td><a href="https://youtu.be/VHKyqZ7t8Gw" title="Video By Logan LeGrand"><img src="https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c"></a></td>
    <td><a href="https://youtu.be/SUIZYzxtMQs" title="A drive to Taco Bell"><img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ce77dc6f63"></a></td>
  </tr>
</table>

## Key Features of openpilot

*   **Advanced Driver-Assistance Systems (ADAS):** Enhances existing driver-assistance features in supported vehicles.
*   **Open Source:**  Join a community of developers and contribute to the advancement of autonomous driving technology.
*   **Wide Vehicle Compatibility:** Compatible with 300+ supported car models.
*   **Continuous Improvement:**  Benefit from ongoing development, updates, and feature enhancements.
*   **Community Driven:** Collaborate with other users and developers via Discord and GitHub.

## Getting Started with openpilot

To use openpilot, you'll need the following:

1.  **Supported Device:** A comma 3X is required, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:**  Use the URL `openpilot.comma.ai` in the comma 3X setup to install the release version.
3.  **Supported Car:**  Ensure your car is on [the list of supported cars](docs/CARS.md).
4.  **Car Harness:**  A [car harness](https://comma.ai/shop/car-harness) is needed to connect your comma 3X.

Detailed instructions are available for [installing the harness and device](https://comma.ai/setup).  You can also explore running openpilot on [other hardware](https://blog.comma.ai/self-driving-car-for-free/), but this is not a plug-and-play solution.

## openpilot Branches

Choose the branch that best suits your needs:

| Branch          | URL                         | Description                                                             |
|-----------------|-----------------------------|-------------------------------------------------------------------------|
| `release3`        | openpilot.comma.ai           | Stable release branch.                                                   |
| `release3-staging`| openpilot-test.comma.ai      | Staging branch for early access to new releases.                       |
| `nightly`         | openpilot-nightly.comma.ai    | Bleeding-edge development branch; expect instability.                   |
| `nightly-dev`     | installer.comma.ai/commaai/nightly-dev | Development branch with experimental features for some cars.     |

## Contributing to openpilot

openpilot is a community-driven project.  Get involved by:

*   Joining the [community Discord](https://discord.comma.ai).
*   Reviewing the [contributing documentation](docs/CONTRIBUTING.md).
*   Exploring the [openpilot tools](tools/).
*   Referencing the code documentation at https://docs.comma.ai
*   Consulting the [community wiki](https://github.com/commaai/openpilot/wiki) for additional information.

[Comma is hiring](https://comma.ai/jobs#open-positions) and offers [bounties](https://comma.ai/bounties) for external contributors.

## Safety and Testing

*   openpilot adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines; see [SAFETY.md](docs/SAFETY.md) for details.
*   Automated software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) run on every commit.
*   The safety model code is in panda and written in C, see [code rigor](https://github.com/commaai/panda#code-rigor) for details.
*   panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Hardware-in-the-loop Jenkins test suite is utilized internally.
*   panda includes hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing is performed on 10 comma devices replaying routes.

<details>
<summary>MIT License</summary>

openpilot is released under the MIT license. Certain software parts are licensed differently, as specified.

Users of this software must indemnify and hold harmless Comma.ai, Inc. and its affiliates from any claims related to the use of this software.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT. YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS. NO WARRANTY EXPRESSED OR IMPLIED.**
</details>

<details>
<summary>User Data and Comma Account</summary>

By default, openpilot uploads driving data to our servers.  You can access your data via [comma connect](https://connect.comma.ai/).  Your data helps improve the models and openpilot for everyone.

Data collection can be disabled.

openpilot logs road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.  The driver-facing camera and microphone are only logged if you opt-in.

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy).  You grant comma an irrevocable, perpetual right to use the generated data.
</details>
<div align="center" style="text-align: center;">
<h1>openpilot</h1>

<p>
  <b>Transform your driving experience with openpilot, the open-source driver-assistance system that upgrades your car's capabilities.</b>
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
    <td><a href="https://youtu.be/SUIZYzxtMQs" title="A drive to Taco Bell"><img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ce77dc6f63"></a></td>
  </tr>
</table>

## What is openpilot?

openpilot is a cutting-edge, open-source **driver-assistance system** designed to enhance the capabilities of your vehicle. It offers advanced features like **adaptive cruise control, lane keeping assist, and automatic lane changes**, making your driving experience safer and more convenient. Developed by [comma.ai](https://comma.ai/), openpilot is constantly evolving with contributions from a vibrant community.  **[View the original repository](https://github.com/commaai/openpilot)**.

## Key Features

*   **Adaptive Cruise Control:** Maintains a set speed and distance from vehicles ahead.
*   **Lane Keeping Assist:** Keeps your vehicle centered in its lane.
*   **Automatic Lane Changes:** Executes lane changes with driver confirmation.
*   **Open Source:** Benefit from community contributions and full transparency.
*   **Wide Vehicle Support:** Works with over 300+ supported car models.
*   **Continuous Improvement:** Benefit from frequent updates and improvements.

## Getting Started

To use openpilot, you'll need the following:

1.  **Supported Device:** A [comma 3X](https://comma.ai/shop/comma-3x)
2.  **Software:**  Install the openpilot release version using the URL `openpilot.comma.ai`.
3.  **Supported Car:** Ensure your car is on [the supported vehicles list](docs/CARS.md).
4.  **Car Harness:** You'll also need a [car harness](https://comma.ai/shop/car-harness) to connect your comma 3X to your car.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

### Branches

*   **release3:** `openpilot.comma.ai` - The stable release branch.
*   **release3-staging:** `openpilot-test.comma.ai` - Staging branch for early access to releases.
*   **nightly:** `openpilot-nightly.comma.ai` - Bleeding-edge development branch (unstable).
*   **nightly-dev:** `installer.comma.ai/commaai/nightly-dev` - Experimental development features (unstable).

## Contributing & Development

openpilot thrives on community contributions.  Join us!

*   **Community Discord:** Join the [community Discord](https://discord.comma.ai)
*   **Contribution Guidelines:** Review the [contributing docs](docs/CONTRIBUTING.md)
*   **Openpilot Tools:** Explore the [openpilot tools](tools/)
*   **Documentation:** Find code documentation at https://docs.comma.ai
*   **Community Wiki:** Learn more on the [community wiki](https://github.com/commaai/openpilot/wiki)
*   **Jobs and Bounties:** Consider [comma's open positions](https://comma.ai/jobs#open-positions) and [bounties](https://comma.ai/bounties).

## Safety and Testing

openpilot is committed to safety, following [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines.

*   **Safety Documentation:** See [SAFETY.md](docs/SAFETY.md) for more details.
*   **Automated Testing:** Software-in-the-loop tests run on every commit (.github/workflows/selfdrive_tests.yaml).
*   **Code Rigor:** Safety model code is written in C within panda and rigorously tested ([code rigor](https://github.com/commaai/panda#code-rigor)).
*   **Safety Tests:** [Safety tests](https://github.com/commaai/panda/tree/master/tests/safety) are done using panda.
*   **Hardware-in-the-loop Testing:** Jenkins test suite used for build and unit tests.  Additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile) are also used.
*   **Continuous Testing:** Continuous testing with multiple comma devices replaying routes.

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
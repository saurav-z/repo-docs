<div align="center">
  <h1>openpilot: Open Source Driver Assistance System</h1>
  <p><b>Upgrade your driving experience with openpilot, the open-source driver assistance system that enhances your car's capabilities.</b></p>

  <h3>
    <a href="https://docs.comma.ai">Docs</a>
    <span> · </span>
    <a href="https://docs.comma.ai/contributing/roadmap/">Roadmap</a>
    <span> · </span>
    <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
    <span> · </span>
    <a href="https://discord.comma.ai">Community</a>
    <span> · </span>
    <a href="https://comma.ai/shop">Get a comma 3X</a>
  </h3>

  Quick Start: `bash <(curl -fsSL openpilot.comma.ai)`

  [![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
  [![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)
</div>

<p align="center">
  <a href="https://youtu.be/NmBfgOanCyk" title="Video By Greer Viau"><img src="https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772" width="30%"></a>
  <a href="https://youtu.be/VHKyqZ7t8Gw" title="Video By Logan LeGrand"><img src="https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c" width="30%"></a>
  <a href="https://youtu.be/SUIZYzxtMQs" title="A drive to Taco Bell"><img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ce77dc6f63" width="30%"></a>
</p>

## Key Features of openpilot

*   **Advanced Driver-Assistance Systems (ADAS):** Enhances existing ADAS features in compatible vehicles.
*   **Open Source:**  Contribute to the project and customize it to your needs.
*   **Wide Vehicle Compatibility:** Supports 300+ car models.
*   **Community Driven:**  Benefit from a vibrant community and collaborative development.
*   **Continuous Improvement:**  Constantly evolving with new features, improvements, and bug fixes.

## Getting Started with openpilot

To use openpilot, you'll need:

1.  **comma 3X Device:**  Available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **openpilot Software:** Install the release version using the URL `openpilot.comma.ai`.
3.  **Compatible Car:**  Check the list of [supported cars](docs/CARS.md).
4.  **Car Harness:**  Required to connect your comma 3X to your car, available at [comma.ai/shop/car-harness](https://comma.ai/shop/car-harness).

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).  You can also explore running openpilot on [other hardware](https://blog.comma.ai/self-driving-car-for-free/), though this is not a plug-and-play experience.

## Development Branches

Choose the branch that suits your needs:

*   `release3`:  Stable release branch (`openpilot.comma.ai`)
*   `release3-staging`:  Staging branch for early access to new releases (`openpilot-test.comma.ai`)
*   `nightly`:  Bleeding-edge development branch (unstable) (`openpilot-nightly.comma.ai`)
*   `nightly-dev`:  Experimental development branch, with experimental features for some cars (`installer.comma.ai/commaai/nightly-dev`)

## Contribute to openpilot

openpilot thrives on community contributions. Join us!

*   **GitHub:**  Find the project on [GitHub](http://github.com/commaai/openpilot).
*   **Community:**  Join the [community Discord](https://discord.comma.ai).
*   **Contribute:** Check out the [contributing docs](docs/CONTRIBUTING.md)
*   **Tools:** Explore the [openpilot tools](tools/).
*   **Documentation:** Access code documentation at [https://docs.comma.ai](https://docs.comma.ai).
*   **Wiki:** Find information about running openpilot on the [community wiki](https://github.com/commaai/openpilot/wiki)

**Looking for opportunities?** comma is hiring! Explore [open positions](https://comma.ai/jobs#open-positions) and [bounties](https://comma.ai/bounties) for contributors.

## Safety and Testing

openpilot is committed to safety and rigorous testing:

*   **ISO26262 Guidelines:** Adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) safety guidelines. See [SAFETY.md](docs/SAFETY.md) for details.
*   **Automated Testing:**  Software-in-the-loop tests run on every commit.
*   **Safety Model:** The code enforcing the safety model lives in panda and is written in C.
*   **Safety Tests:** panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   **Hardware-in-the-loop Testing:** Jenkins test suite builds and unit tests.
*   **Continuous Testing:**  Runs the latest openpilot in a testing environment with multiple devices replaying routes.

[Go back to the top](#openpilot-open-source-driver-assistance-system)
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

By default, openpilot uploads the driving data to our servers. You can also access your data through [comma connect](https://connect.comma.ai/). We use your data to train better models and improve openpilot for everyone.

openpilot is open source software: the user is free to disable data collection if they wish to do so.

openpilot logs the road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
The driver-facing camera and microphone are only logged if you explicitly opt-in in settings.

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy). You understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma for the use of this data.
</details>
[Back to the openpilot GitHub Repository](https://github.com/commaai/openpilot)
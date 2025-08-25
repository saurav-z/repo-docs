# openpilot: Open Source Driver Assistance for Your Car

**Upgrade your driving experience with openpilot, an open-source operating system that adds advanced driver-assistance features to over 300 supported car models.** ([View the original repo](https://github.com/commaai/openpilot))

<div align="center" style="text-align: center;">
  <a href="https://docs.comma.ai">Docs</a>
  <span> · </span>
  <a href="https://docs.comma.ai/contributing/roadmap/">Roadmap</a>
  <span> · </span>
  <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Community</a>
  <span> · </span>
  <a href="https://comma.ai/shop">Try it on a comma 3X</a>
</div>

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

<br>

<table>
  <tr>
    <td><a href="https://youtu.be/NmBfgOanCyk" title="Video By Greer Viau"><img src="https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772"></a></td>
    <td><a href="https://youtu.be/VHKyqZ7t8Gw" title="Video By Logan LeGrand"><img src="https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c"></a></td>
    <td><a href="https://youtu.be/SUIZYzxtMQs" title="A drive to Taco Bell"><img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ce89dc6f63"></a></td>
  </tr>
</table>

## Key Features

*   **Open Source:** Benefit from a community-driven project with publicly available code and continuous improvements.
*   **Enhanced Driver Assistance:** Add features like adaptive cruise control, lane keeping assist, and more.
*   **Wide Vehicle Support:** Compatible with over 300+ car models (check [CARS.md](docs/CARS.md) for details).
*   **Easy Installation:** Quick setup with a comma 3/3X device and a car harness.
*   **Continuous Updates:** Stay up-to-date with the latest features and improvements through regular updates.

## Getting Started

### What You'll Need

To use openpilot, you'll need the following:

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version by entering the URL `openpilot.comma.ai` during setup of the comma 3/3X.
3.  **Supported Car:** Ensure your car is on the [supported car list](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) to connect your device to your car.

### Installation

Detailed instructions for installing the harness and device are available at [comma.ai/setup](https://comma.ai/setup). Note that you can also run openpilot on [other hardware](https://blog.comma.ai/self-driving-car-for-free/), although it requires more technical skill.

## Branches
*   `release3`         | `openpilot.comma.ai`                      | This is openpilot's release branch.
*   `release3-staging` | `openpilot-test.comma.ai`                | This is the staging branch for releases. Use it to get new releases slightly early.
*   `nightly`          | `openpilot-nightly.comma.ai`             | This is the bleeding edge development branch. Do not expect this to be stable.
*   `nightly-dev`      | `installer.comma.ai/commaai/nightly-dev` | Same as nightly, but includes experimental development features for some cars.

## Contributing and Development

openpilot thrives on community contributions!  We welcome pull requests and encourage participation.

*   Join the [community Discord](https://discord.comma.ai) to connect with other users and developers.
*   Check out the [contributing docs](docs/CONTRIBUTING.md) to learn how to contribute.
*   Explore the [openpilot tools](tools/) for helpful utilities.
*   Find the official code documentation at https://docs.comma.ai.
*   The community wiki at https://github.com/commaai/openpilot/wiki provides additional information.
*   Want to get paid to work on openpilot? [comma is hiring](https://comma.ai/jobs#open-positions) and offers lots of [bounties](https://comma.ai/bounties) for external contributors.

## Safety and Testing

openpilot is committed to safety and undergoes rigorous testing.

*   Adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines; see [SAFETY.md](docs/SAFETY.md) for details.
*   Features software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) that run on every commit.
*   The safety model code is in panda and written in C, see [code rigor](https://github.com/commaai/panda#code-rigor) for more details.
*   panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Employs a hardware-in-the-loop Jenkins test suite for building and unit testing various processes.
*   panda has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Utilizes a testing closet with 10 comma devices continuously replaying routes for comprehensive testing.

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
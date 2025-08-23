<div align="center" style="text-align: center;">

<h1>openpilot: Open Source Driver Assistance System</h1>

<p>
  <b>Transform your driving experience with openpilot, an open-source autonomous driving system that enhances driver assistance features in hundreds of supported vehicles.</b>
  <br>
  Currently, it upgrades the driver assistance system in 300+ supported cars.
  <br>
  <a href="https://github.com/commaai/openpilot">View the original repository on GitHub</a>
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

## Key Features of openpilot

*   **Advanced Driver Assistance:** Enhances existing ADAS features, adding capabilities like lane keeping and adaptive cruise control.
*   **Open Source:**  Benefit from the collaborative development and transparency of an open-source project.
*   **Wide Vehicle Support:**  Works with 300+ car models (check [supported cars](docs/CARS.md)).
*   **Easy Installation:** Simple setup process using a comma 3/3X device.
*   **Continuous Improvement:**  Benefit from frequent updates and community-driven enhancements.
*   **Safety Focused:** Adheres to ISO26262 guidelines and undergoes rigorous testing.

## Getting Started with openpilot

To use openpilot in your car:

1.  **Hardware:**  Purchase a [comma 3/3X](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the openpilot software using the URL `openpilot.comma.ai` during setup.
3.  **Vehicle Compatibility:** Confirm your car is supported. See [supported cars](docs/CARS.md).
4.  **Harness:** Get a [car harness](https://comma.ai/shop/car-harness) to connect the device to your car.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).  You can also explore running openpilot on [other hardware](https://blog.comma.ai/self-driving-car-for-free/), although this is not a plug-and-play solution.

## Branches
| branch           | URL                                    | description                                                                         |
|------------------|----------------------------------------|-------------------------------------------------------------------------------------|
| `release3`         | openpilot.comma.ai                      | This is openpilot's release branch.                                                 |
| `release3-staging` | openpilot-test.comma.ai                | This is the staging branch for releases. Use it to get new releases slightly early. |
| `nightly`          | openpilot-nightly.comma.ai             | This is the bleeding edge development branch. Do not expect this to be stable.      |
| `nightly-dev`      | installer.comma.ai/commaai/nightly-dev | Same as nightly, but includes experimental development features for some cars.      |

## Contributing to openpilot

openpilot thrives on community contributions.  Get involved!

*   Join the [community Discord](https://discord.comma.ai).
*   Review the [contributing docs](docs/CONTRIBUTING.md).
*   Explore the [openpilot tools](tools/).
*   Consult the code documentation at https://docs.comma.ai.
*   Find information about running openpilot on the [community wiki](https://github.com/commaai/openpilot/wiki).

## Opportunities

Looking for a role within the openpilot ecosystem?  [comma is hiring](https://comma.ai/jobs#open-positions) and offers bounties for external contributions.

## Safety and Testing

openpilot prioritizes safety:

*   Adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines, see [SAFETY.md](docs/SAFETY.md).
*   Utilizes software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) on every commit.
*   The safety model code in panda is written in C, see [code rigor](https://github.com/commaai/panda#code-rigor).
*   panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Includes a hardware-in-the-loop Jenkins test suite for building and unit testing.
*   panda has hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Uses a continuous testing environment with multiple comma devices replaying routes.

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
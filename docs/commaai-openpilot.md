<div align="center" style="text-align: center;">

<h1>openpilot: Open Source Driver Assistance for Your Car</h1>

<p>
  <b>Transform your driving experience with openpilot, an open-source driving assistance system.</b>
  <br>
  Upgrade your vehicle with advanced driver-assistance features using this cutting-edge technology.
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
    <td><a href="https://youtu.be/SUIZYzxtMQs" title="A drive to Taco Bell"><img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ceefc6f63"></a></td>
  </tr>
</table>

## Key Features of openpilot

*   **Advanced Driver Assistance:** Experience features like lane keeping, adaptive cruise control, and automatic emergency braking.
*   **Open Source and Customizable:**  Modify and adapt the software to suit your specific needs and contribute to the open-source community.
*   **Wide Car Support:**  Works with 300+ supported car models (check [CARS.md](docs/CARS.md)).
*   **Regular Updates:** Benefit from continuous improvements, bug fixes, and new features developed by the community.
*   **Community Driven:**  Get support and collaborate with other users through the active [Discord community](https://discord.comma.ai).

## Getting Started with openpilot

To use openpilot, you'll need the following:

1.  **Supported Device:** A comma 3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version using the URL `openpilot.comma.ai` during the comma 3X setup.
3.  **Supported Car:** Verify compatibility with your car model using the list [here](docs/CARS.md).
4.  **Car Harness:** Purchase a [car harness](https://comma.ai/shop/car-harness) to connect the device to your car.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup). Consider running openpilot on [other hardware](https://blog.comma.ai/self-driving-car-for-free/) as well.

## Branches

| Branch             | URL                         | Description                                                                    |
| ------------------ | --------------------------- | ------------------------------------------------------------------------------ |
| `release3`         | openpilot.comma.ai          | The stable release branch.                                                     |
| `release3-staging` | openpilot-test.comma.ai     | Staging branch for pre-release testing.                                        |
| `nightly`          | openpilot-nightly.comma.ai  | Bleeding-edge development branch; expect instability.                          |
| `nightly-dev`      | installer.comma.ai/commaai/nightly-dev | Includes experimental development features for some cars (use with caution). |

## Contributing to openpilot

openpilot thrives on community contributions. You can get involved by:

*   Joining the [community Discord](https://discord.comma.ai)
*   Reviewing the [contributing docs](docs/CONTRIBUTING.md)
*   Exploring the [openpilot tools](tools/)
*   Accessing code documentation at https://docs.comma.ai
*   Checking the [community wiki](https://github.com/commaai/openpilot/wiki)

Consider working on openpilot for compensation; [comma is hiring](https://comma.ai/jobs#open-positions) and offers bounties.

## Safety and Testing

*   openpilot follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines; see [SAFETY.md](docs/SAFETY.md) for more details.
*   Automated software-in-the-loop tests run on every commit.
*   The safety model code is in panda and written in C (see [code rigor](https://github.com/commaai/panda#code-rigor)).
*   panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Hardware-in-the-loop tests run internally using Jenkins and unit tests.
*   panda has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   The latest openpilot is constantly tested on comma devices replaying routes.

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
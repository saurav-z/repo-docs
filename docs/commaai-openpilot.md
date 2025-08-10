<div align="center" style="text-align: center;">

<h1>openpilot: Open Source Driver Assistance System</h1>

<p>
  <b>Transform your driving experience with openpilot, an open-source driving automation system that enhances your car's capabilities.</b>
  <br>
  Upgrade your vehicle with advanced driver-assistance features and contribute to the future of autonomous driving.
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

[![openpilot demonstration video](https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772)](https://youtu.be/NmBfgOanCyk)
[![openpilot demonstration video](https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c)](https://youtu.be/VHKyqZ7t8Gw)
[![openpilot demonstration video](https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ceef63)
](https://youtu.be/SUIZYzxtMQs)

## Key Features of openpilot

*   **Open Source:** Explore, modify, and contribute to a continuously evolving project.
*   **Driver Assistance:** Enhance your vehicle's capabilities with advanced features like adaptive cruise control and lane-keeping assist.
*   **Supported Cars:** Compatible with over 300+ car models (see [supported cars](docs/CARS.md)).
*   **Community-Driven:** Benefit from a vibrant community, ongoing development, and regular updates.
*   **Continuous Improvement:** Data is used to train better models and improve openpilot for everyone.

## Getting Started with openpilot

To begin using openpilot, follow these steps:

1.  **Supported Device:** Purchase a comma 3/3X device from [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software Installation:** Use the URL `openpilot.comma.ai` in your comma 3/3X settings to install the latest release.
3.  **Vehicle Compatibility:** Verify that your car is on the [supported car list](docs/CARS.md).
4.  **Car Harness:** Obtain a compatible [car harness](https://comma.ai/shop/car-harness).

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

## Openpilot Branches

Choose the branch that best suits your needs:

| Branch              | URL                             | Description                                                                         |
| ------------------- | ------------------------------- | ------------------------------------------------------------------------------------- |
| `release3`          | openpilot.comma.ai              | This is openpilot's release branch.                                                 |
| `release3-staging`  | openpilot-test.comma.ai         | This is the staging branch for releases. Use it to get new releases slightly early. |
| `nightly`           | openpilot-nightly.comma.ai      | This is the bleeding edge development branch. Do not expect this to be stable.      |
| `nightly-dev`       | installer.comma.ai/commaai/nightly-dev | Same as nightly, but includes experimental development features for some cars.      |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | This is a preview branch from the autonomy team where new driving models get merged earlier than master. |

## Contribute to openpilot

Help shape the future of autonomous driving! openpilot thrives on community contributions.

*   Join the active [Discord community](https://discord.comma.ai).
*   Review the [contributing guidelines](docs/CONTRIBUTING.md).
*   Explore the [openpilot tools](tools/).
*   Access comprehensive code documentation at https://docs.comma.ai.
*   Find more information on the [community wiki](https://github.com/commaai/openpilot/wiki).
*   Consider applying for open positions or bounties at [comma.ai/jobs#open-positions](https://comma.ai/jobs#open-positions)

## Safety and Testing

openpilot prioritizes safety and undergoes rigorous testing.

*   Adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Features software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) on every commit.
*   Safety model code is written in C within panda.
*   Includes hardware-in-the-loop safety tests within panda.
*   Runs continuous hardware-in-the-loop testing using Jenkins and a dedicated testing environment.

## License, Data Use, and Legal

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

By default, openpilot uploads driving data to comma.ai servers to improve the software. Users can access their data via [comma connect](https://connect.comma.ai/). Users can disable data collection if they wish.

openpilot logs the road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs. The driver-facing camera and microphone are only logged if you explicitly opt-in in settings.

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy). You understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma for the use of this data.
</details>

---

**[Explore the openpilot repository on GitHub](https://github.com/commaai/openpilot)**
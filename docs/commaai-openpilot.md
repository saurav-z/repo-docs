<div align="center" style="text-align: center;">

<h1>openpilot: Open Source Driver Assistance System</h1>

<p>
  <b>Transform your driving experience with openpilot, the open-source operating system that upgrades your car's driver assistance features.</b>
  <br>
  Enhance your vehicle's capabilities with advanced features and contribute to the future of autonomous driving.
  <br>
  Explore the world of openpilot on its <a href="https://github.com/commaai/openpilot">GitHub Repository</a>.
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

## Key Features

*   **Open-Source:** Dive into the code and contribute to openpilot's development.
*   **Advanced Driver Assistance:** Enhance your car's capabilities with features like lane keeping, adaptive cruise control, and more.
*   **Wide Compatibility:** Supports over 300+ car models with regular updates.
*   **Community-Driven:** Join a vibrant community of users and developers.
*   **Continuous Development:** Benefit from ongoing improvements and new features.

## Getting Started with openpilot

To begin using openpilot, follow these steps:

1.  **Obtain a Supported Device:** Purchase a comma 3/3X device from [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Install the Software:** Use the URL `openpilot.comma.ai` during the comma 3/3X setup to install the release version.
3.  **Confirm Car Compatibility:** Verify that your car model is supported by checking [the list of supported cars](docs/CARS.md).
4.  **Get a Car Harness:** Acquire a compatible [car harness](https://comma.ai/shop/car-harness) for your vehicle.

For detailed installation instructions, visit [comma.ai/setup](https://comma.ai/setup). Consider running openpilot on [other hardware](https://blog.comma.ai/self-driving-car-for-free/), though this is not a plug-and-play experience.

## openpilot Branches

| Branch            | URL                                     | Description                                                                         |
| ----------------- | --------------------------------------- | ------------------------------------------------------------------------------------- |
| `release3`        | openpilot.comma.ai                       | The stable release branch.                                                             |
| `release3-staging` | openpilot-test.comma.ai                 | Staging branch for early access to new releases.                                       |
| `nightly`         | openpilot-nightly.comma.ai              | The cutting-edge development branch; may be unstable.                                |
| `nightly-dev`     | installer.comma.ai/commaai/nightly-dev  | Similar to nightly, includes experimental features for some cars.                     |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch from the autonomy team with earlier merges of new driving models. |

## Contributing to openpilot

openpilot thrives on community contributions!

*   Join the [community Discord](https://discord.comma.ai) to connect with other users and developers.
*   Review the [contributing docs](docs/CONTRIBUTING.md) for guidance.
*   Explore the [openpilot tools](tools/) to understand the development process.
*   Access comprehensive code documentation at https://docs.comma.ai
*   Find more information about using openpilot on the [community wiki](https://github.com/commaai/openpilot/wiki).

Interested in contributing to openpilot full-time?  [comma is hiring](https://comma.ai/jobs#open-positions) and offers [bounties](https://comma.ai/bounties) for external contributors.

## Safety and Testing

*   openpilot adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) run with every commit.
*   Safety model code is in `panda` and written in C (see [code rigor](https://github.com/commaai/panda#code-rigor)).
*   `panda` has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Internal hardware-in-the-loop Jenkins test suite for builds and unit tests.
*   `panda` has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing with multiple comma devices replaying routes in a dedicated testing environment.

<details>
<summary>MIT License</summary>

openpilot is released under the MIT license.  See [LICENSE](LICENSE)

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
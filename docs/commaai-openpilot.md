# openpilot: Open Source Driver Assistance System

**Transform your driving experience with openpilot, an open-source driving agent that enhances driver-assistance features in a growing list of over 300 supported vehicles.** Explore the next generation of driving technology, with the power to upgrade your car's capabilities today.  [Back to the original repository](https://github.com/commaai/openpilot)

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

<div align="center">
  <a href="https://youtu.be/NmBfgOanCyk" title="Video By Greer Viau"><img src="https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772" width="30%"></a>
  <a href="https://youtu.be/VHKyqZ7t8Gw" title="Video By Logan LeGrand"><img src="https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c" width="30%"></a>
  <a href="https://youtu.be/SUIZYzxtMQs" title="A drive to Taco Bell"><img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ceefc6f63" width="30%"></a>
</div>


## Key Features

*   **Enhanced Driver Assistance:** Upgrade your vehicle's existing driver assistance features.
*   **Open Source:** Benefit from a community-driven project with ongoing development and improvements.
*   **Wide Vehicle Support:** Compatible with a growing list of over 300+ supported cars ([see CARS.md](docs/CARS.md)).
*   **Continuous Updates:** Benefit from frequently updated software versions including release, staging, and development branches.
*   **Community-Driven:** Join the active [Discord community](https://discord.comma.ai) and contribute to the project.
*   **Safety Focused:** Implements ISO26262 guidelines and undergoes rigorous testing.

## Getting Started with openpilot

To utilize openpilot, you'll need these components:

1.  **Compatible Hardware:** A [comma 3/3X](https://comma.ai/shop/comma-3x) device.
2.  **openpilot Software:** Install openpilot by entering the URL `openpilot.comma.ai` in the setup procedure for your comma 3/3X.
3.  **Supported Vehicle:**  Verify your car is on the [supported cars list](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) is required to connect your device to your vehicle.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).  Consider running openpilot on [other hardware](https://blog.comma.ai/self-driving-car-for-free/) to explore alternatives.

## openpilot Branches

Select the branch that aligns with your desired level of stability:

| Branch                 | URL                                        | Description                                                                                                                                          |
| ---------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `release3`             | `openpilot.comma.ai`                        | The stable release branch.                                                                                                                           |
| `release3-staging`     | `openpilot-test.comma.ai`                  | A staging branch for early access to upcoming releases.                                                                                               |
| `nightly`              | `openpilot-nightly.comma.ai`               | The development branch, featuring the latest advancements. Expect potential instability.                                                            |
| `nightly-dev`          | `installer.comma.ai/commaai/nightly-dev`   | Similar to `nightly`, with experimental development features for certain vehicles.                                                                  |
| `secretgoodopenpilot` | `installer.comma.ai/commaai/secretgoodopenpilot` | A preview branch from the autonomy team with cutting-edge driving models, potentially merged ahead of the `master` branch (experimental). |

## Contribute and Collaborate

openpilot is a collaborative effort, and your contributions are welcome!

*   Join the [community Discord](https://discord.comma.ai)
*   Explore the [contribution guidelines](docs/CONTRIBUTING.md)
*   Access the [openpilot tools](tools/)
*   Consult the code documentation at [https://docs.comma.ai](https://docs.comma.ai)
*   Review the [community wiki](https://github.com/commaai/openpilot/wiki) for more information.

## Safety and Testing

openpilot is designed with safety in mind and undergoes thorough testing.

*   Follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines ([see SAFETY.md](docs/SAFETY.md)).
*   Features software-in-the-loop tests, executed on every commit ([.github/workflows/selfdrive_tests.yaml](.github/workflows/selfdrive_tests.yaml)).
*   The code responsible for the safety model is written in C and is rigorously tested (see [code rigor](https://github.com/commaai/panda#code-rigor)).
*   `panda` includes software-in-the-loop safety tests ([https://github.com/commaai/panda/tree/master/tests/safety](https://github.com/commaai/panda/tree/master/tests/safety)).
*   Utilizes a hardware-in-the-loop Jenkins test suite internally.
*   `panda` provides additional hardware-in-the-loop tests ([https://github.com/commaai/panda/blob/master/Jenkinsfile](https://github.com/commaai/panda/blob/master/Jenkinsfile)).
*   Continuously tests the latest openpilot versions in a dedicated testing environment.

<details>
<summary>MIT License</summary>

openpilot is released under the MIT license. Specific portions of the software may utilize alternative licenses, as indicated.

Users of this software agree to indemnify and hold harmless Comma.ai, Inc., its directors, officers, employees, agents, stockholders, affiliates, subcontractors, and customers from any claims related to the use of this software.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**
</details>

<details>
<summary>User Data and comma Account</summary>

openpilot, by default, uploads driving data to our servers. You can also access your data via [comma connect](https://connect.comma.ai/). This data is used to enhance our models and improve the project.

Users have the option to disable data collection.

openpilot logs data from the road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
The driver-facing camera and microphone are logged only if explicitly enabled in settings.

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy). You acknowledge that the use of this software will generate user data, which may be logged and stored at the discretion of comma. By accepting these terms, you grant comma an irrevocable, perpetual, worldwide right to use this data.
</details>
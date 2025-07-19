# openpilot: Open Source Driver Assistance for Your Car

**Openpilot revolutionizes driving by providing advanced driver-assistance systems (ADAS) for over 300 supported car models.**  [Learn more about openpilot on GitHub](https://github.com/commaai/openpilot).

<div align="center" style="text-align: center;">

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

</div>

## Key Features of openpilot

*   **Advanced Driver-Assistance:** Enhances existing driver assistance systems, providing features like lane keeping, adaptive cruise control, and more.
*   **Broad Compatibility:** Supports a wide range of vehicles.
*   **Open Source:** Benefit from community contributions and customization possibilities.
*   **Easy Installation:** Utilize the comma 3/3X device and simple software installation process.
*   **Continuous Improvement:**  Regular updates and development driven by user data and open-source collaboration.
*   **Safety Focused:** Adheres to ISO26262 guidelines and undergoes rigorous testing.
*   **Community Support:** Join the vibrant [Discord](https://discord.comma.ai) community for support, discussion, and collaboration.

## How to Get Started

To use openpilot in your car, you'll need:

1.  **A Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install openpilot by entering `openpilot.comma.ai` during setup.
3.  **A Supported Car:** Ensure your car is one of the [275+ supported models](docs/CARS.md).
4.  **Car Harness:** A car harness, available at [comma.ai/shop/car-harness](https://comma.ai/shop/car-harness), to connect your device to your car.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).  You can also explore running openpilot on [alternative hardware](https://blog.comma.ai/self-driving-car-for-free/), though this is not a plug-and-play experience.

## Development and Community

*   **Contribute:**  We welcome your contributions!  Find details in the [CONTRIBUTING.md](docs/CONTRIBUTING.md) document.
*   **Join the Community:** Engage with other users and developers on the [community Discord](https://discord.comma.ai).
*   **Explore Tools:** Check out the [openpilot tools](tools/).
*   **Documentation:** Browse the code documentation at https://docs.comma.ai and find more information on the [community wiki](https://github.com/commaai/openpilot/wiki).
*   **Career Opportunities:** Consider a career at comma: [comma is hiring](https://comma.ai/jobs#open-positions) with bounties for contributors.

## Safety and Testing

*   Follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines.
*   Utilizes software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) on every commit.
*   Safety model code is in C within panda ([code rigor](https://github.com/commaai/panda#code-rigor)).
*   Includes software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety) for the panda.
*   Internal hardware-in-the-loop Jenkins test suite.
*   Additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing with multiple devices replaying routes.

## Branches

| Branch           | URL                              | Description                                                                         |
|------------------|----------------------------------|-------------------------------------------------------------------------------------|
| `release3`       | openpilot.comma.ai                | Stable release branch.                                                              |
| `release3-staging`| openpilot-test.comma.ai          | Staging branch for release previews.                                              |
| `nightly`        | openpilot-nightly.comma.ai       | Bleeding-edge development branch; potential instability.                           |
| `nightly-dev`    | installer.comma.ai/commaai/nightly-dev | Experimental development features for some cars.                                    |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch from the autonomy team with newer driving models before master. |

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
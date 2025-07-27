# openpilot: Transform Your Drive with Open Source Robotics

**openpilot is an open-source driver-assistance system that enhances the capabilities of over 300 supported car models.** Learn more and contribute on the [original GitHub repository](https://github.com/commaai/openpilot).

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

## Key Features

*   **Enhanced Driver Assistance:** Upgrade your car's existing driver-assistance features for a smoother and more advanced driving experience.
*   **Extensive Car Support:** Compatible with over 300+ car models, continuously expanding to cover a wide range of vehicles.
*   **Open-Source Development:** Contribute to the project, access the source code, and tailor it to your needs, fostering community-driven innovation.
*   **Regular Updates:** Benefit from continuous improvements, bug fixes, and new features through frequent software updates.
*   **Community & Support:** Engage with a vibrant community of users and developers via Discord and other platforms, offering knowledge sharing and collaborative support.

## How to Get Started

To use openpilot in your car, you'll need:

1.  **Supported Device:** A comma 3/3X (available at [comma.ai/shop](https://comma.ai/shop/comma-3x)).
2.  **Software:** Install openpilot by using the URL `openpilot.comma.ai` during the comma 3/3X setup.
3.  **Supported Car:** Ensure your car model is on the [supported cars list](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) to connect the device to your car.

Detailed installation instructions can be found at [comma.ai/setup](https://comma.ai/setup).

## Branches

Choose the openpilot version that fits your needs:

| Branch            | URL                        | Description                                                                         |
| ----------------- | -------------------------- | ----------------------------------------------------------------------------------- |
| `release3`        | openpilot.comma.ai         | Stable release branch.                                                              |
| `release3-staging`| openpilot-test.comma.ai    | Staging branch for early access to upcoming releases.                                |
| `nightly`         | openpilot-nightly.comma.ai | Bleeding-edge development branch; may be unstable.                                   |
| `nightly-dev`     | `installer.comma.ai/commaai/nightly-dev` | Includes experimental development features for certain cars.                  |
| `secretgoodopenpilot` | `installer.comma.ai/commaai/secretgoodopenpilot` | Preview branch with early merges of driving models from the autonomy team. |

## Contributing to openpilot

Become a part of the openpilot community!

*   Join the [Discord community](https://discord.comma.ai)
*   Review the [contributing guidelines](docs/CONTRIBUTING.md)
*   Explore the [openpilot tools](tools/)
*   Consult the code documentation: https://docs.comma.ai
*   Visit the [community wiki](https://github.com/commaai/openpilot/wiki) for more information.

## Safety and Testing

openpilot prioritizes safety and rigorous testing:

*   Adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines. (See [SAFETY.md](docs/SAFETY.md))
*   Includes comprehensive software-in-the-loop tests that run with every commit.
*   Utilizes a C-written safety model within panda.
*   Employs hardware-in-the-loop testing for both panda and the broader system.

<details>
<summary>MIT License</summary>

openpilot is licensed under the MIT License.
</details>

<details>
<summary>User Data and Privacy</summary>

openpilot collects driving data to improve the system. Users can access their data via [comma connect](https://connect.comma.ai/). Data collection can be disabled. Logs include camera, CAN, GPS, IMU, magnetometer, thermal sensors, and system logs.  Driver-facing camera and microphone logs are optional. By using openpilot, you agree to the [Privacy Policy](https://comma.ai/privacy).
</details>
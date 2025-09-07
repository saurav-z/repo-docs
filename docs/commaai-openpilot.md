<div align="center">
  <h1>openpilot: Open Source Driver Assistance for Your Car</h1>
</div>

**Transform your driving experience with openpilot, an open-source driver assistance system that upgrades the capabilities of your car's existing features.** ([View on GitHub](https://github.com/commaai/openpilot))

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

---

## Key Features

*   **Adaptive Cruise Control:** Maintains a safe following distance and adjusts speed to traffic.
*   **Lane Keeping Assist:** Keeps your vehicle centered in its lane.
*   **Automatic Lane Changes:** Performs lane changes with driver confirmation.
*   **Open Source & Community Driven:** Benefit from continuous improvements and a vibrant community.
*   **Supports 300+ Cars:** Compatible with a wide range of vehicles.

---

## Getting Started

To use openpilot, you'll need the following:

1.  **Compatible Device:**  A comma 3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:**  Install openpilot by entering the URL `openpilot.comma.ai` during setup on your comma 3X.
3.  **Supported Car:** Verify that your car model is on the [supported cars list](docs/CARS.md).
4.  **Car Harness:** Required for connecting your comma 3X to your car, available at [comma.ai/shop/car-harness](https://comma.ai/shop/car-harness).

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

---

## Branches

Choose the branch that suits your needs:

| Branch           | URL                         | Description                                            |
|------------------|-----------------------------|--------------------------------------------------------|
| `release3`       | openpilot.comma.ai          | Release branch - stable version.                     |
| `release3-staging` | openpilot-test.comma.ai     | Staging branch - get new releases slightly early.    |
| `nightly`          | openpilot-nightly.comma.ai  | Bleeding edge development - may be unstable.         |
| `nightly-dev`      | installer.comma.ai/commaai/nightly-dev | Experimental development features (some cars). |

---

## Contributing

openpilot is a community-driven project.  We welcome your contributions!

*   Join the [community Discord](https://discord.comma.ai).
*   Review the [contributing docs](docs/CONTRIBUTING.md).
*   Explore the [openpilot tools](tools/).
*   Access code documentation at https://docs.comma.ai
*   Find more information on the [community wiki](https://github.com/commaai/openpilot/wiki)

---

## Safety & Testing

openpilot prioritizes safety:

*   Adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines.  See [SAFETY.md](docs/SAFETY.md).
*   Uses software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) with every commit.
*   Code for the safety model is written in C within panda.
*   Includes [safety tests](https://github.com/commaai/panda/tree/master/tests/safety) within panda.
*   Uses a hardware-in-the-loop Jenkins test suite internally.

---

<details>
<summary>MIT License</summary>

openpilot is released under the MIT license. [Read the full license in LICENSE](LICENSE).
... (rest of license text) ...
</details>

<details>
<summary>User Data and comma Account</summary>

By default, openpilot uploads the driving data to our servers. You can also access your data through [comma connect](https://connect.comma.ai/). We use your data to train better models and improve openpilot for everyone.

openpilot is open source software: the user is free to disable data collection if they wish to do so.

openpilot logs the road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
The driver-facing camera and microphone are only logged if you explicitly opt-in in settings.

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy). You understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma for the use of this data.
</details>
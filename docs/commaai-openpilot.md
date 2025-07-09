# openpilot: Open Source Driver Assistance for Your Car

**Upgrade your driving experience with openpilot, an open-source driver-assistance system that enhances the capabilities of supported vehicles.** [Learn more at the original repository](https://github.com/commaai/openpilot).

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

## Key Features

*   **Enhanced Driver Assistance:** Provides advanced driver-assistance features to 300+ supported car models.
*   **Open Source:** Benefit from community contributions and the freedom to customize your driving experience.
*   **Continuous Development:** Stay up-to-date with the latest improvements and features through various release branches.
*   **Data-Driven Improvement:** Contributing driving data helps refine the system's performance for the benefit of all users.
*   **Community Support:** Get help, share experiences, and collaborate with other openpilot users via the community Discord.
*   **Easy Installation:** Install the software using the URL `openpilot.comma.ai`.

## Getting Started with openpilot

To utilize openpilot in your car, you'll need the following:

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install openpilot using the URL `openpilot.comma.ai` in your device's settings.
3.  **Supported Car:** Ensure your vehicle is one of the [275+ supported cars](docs/CARS.md).
4.  **Car Harness:** You will also need a [car harness](https://comma.ai/shop/car-harness) to connect your comma 3/3X to your car.

Detailed installation instructions can be found at [comma.ai/setup](https://comma.ai/setup).

## Release Branches

| Branch              | URL                                     | Description                                                                         |
| ------------------- | --------------------------------------- | ------------------------------------------------------------------------------------- |
| `release3`          | openpilot.comma.ai                       | The official release branch.                                                        |
| `release3-staging`  | openpilot-test.comma.ai                 | Staging branch for early access to new releases.                                    |
| `nightly`           | openpilot-nightly.comma.ai              | Development branch; may be unstable.                                                |
| `nightly-dev`       | installer.comma.ai/commaai/nightly-dev  | Includes experimental development features for some cars.                             |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch from the autonomy team where new driving models get merged earlier than master. |

## Contributing to openpilot

Join the openpilot community and contribute to its development!

*   [Community Discord](https://discord.comma.ai)
*   [Contributing Docs](docs/CONTRIBUTING.md)
*   [Openpilot Tools](tools/)
*   [Code Documentation](https://docs.comma.ai)
*   [Community Wiki](https://github.com/commaai/openpilot/wiki)
*   [Comma Hiring](https://comma.ai/jobs#open-positions) and [Bounties](https://comma.ai/bounties)

## Safety and Testing

*   Follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines.
*   Software-in-the-loop tests run on every commit.
*   Code enforcing the safety model in panda, written in C.
*   Panda has software-in-the-loop safety tests.
*   Hardware-in-the-loop Jenkins test suite.
*   Additional hardware-in-the-loop tests for panda.
*   Continuous testing with multiple comma devices.

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
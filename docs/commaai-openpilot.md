<div align="center">

<h1>openpilot: Open Source Self-Driving for Your Car</h1>

<p>
  <b>Transform your driving experience with openpilot, an open-source driver-assistance system that enhances the capabilities of 300+ supported vehicles.</b>
  <br>
  Upgrade your car with cutting-edge autonomous driving features today.
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
  <a href="https://comma.ai/shop">Get a comma 3X</a>
</h3>

Quick start: `bash <(curl -fsSL openpilot.comma.ai)`

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

</div>

---

## Key Features

*   **Advanced Driver-Assistance:** Experience features like adaptive cruise control, lane keeping assist, and automatic lane changes.
*   **Open Source & Community Driven:** Benefit from continuous improvements and contributions from a passionate community.
*   **Wide Vehicle Support:**  Compatible with 300+ car models, with more being added regularly.
*   **Easy Installation:** Get started quickly with straightforward setup instructions.
*   **Continuous Updates:** Stay up-to-date with the latest advancements and improvements.

---

##  How to Use openpilot

To get started with openpilot, you'll need these components:

1.  **Supported Device:** A comma 3X device, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the openpilot software by entering the URL `openpilot.comma.ai` during the comma 3X setup.
3.  **Supported Car:**  Ensure your vehicle is listed among [the 275+ supported cars](docs/CARS.md).
4.  **Car Harness:**  A [car harness](https://comma.ai/shop/car-harness) to connect the comma 3X to your car.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

##  openpilot Branches

| Branch              | URL                         | Description                                                                     |
| ------------------- | --------------------------- | ------------------------------------------------------------------------------- |
| `release3`          | openpilot.comma.ai          | The stable release branch.                                                       |
| `release3-staging`  | openpilot-test.comma.ai     | A staging branch for pre-release testing.                                     |
| `nightly`           | openpilot-nightly.comma.ai  | The latest development branch; may be unstable.                               |
| `nightly-dev`       | installer.comma.ai/commaai/nightly-dev | Nightly with experimental features for certain cars.                     |

---

## Contributing and Development

openpilot is developed collaboratively by [comma.ai](https://comma.ai/) and the community. We welcome contributions! Find out more:

*   Join the [community Discord](https://discord.comma.ai)
*   Consult the [contributing docs](docs/CONTRIBUTING.md)
*   Explore the [openpilot tools](tools/)
*   Review the [code documentation](https://docs.comma.ai)
*   Visit the [community wiki](https://github.com/commaai/openpilot/wiki) for more information.

[**Get the source code on GitHub!**](https://github.com/commaai/openpilot)

### Employment

*   Are you interested in a role at comma?  [View current open positions](https://comma.ai/jobs#open-positions)
*   Do you want to be a bounty hunter?  [See our bounties](https://comma.ai/bounties)

---

## Safety and Testing

openpilot is built with safety in mind, adhering to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines.

Key safety measures include:

*   [SAFETY.md](docs/SAFETY.md) provides a detailed overview of our safety approach.
*   Automated software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) run on every commit.
*   The safety model's core code is written in C within [panda](https://github.com/commaai/panda#code-rigor).
*   [Panda](https://github.com/commaai/panda) also has safety tests.
*   Our internal Jenkins test suite performs hardware-in-the-loop testing.
*   Continuous testing occurs in a dedicated environment using multiple devices and routes.

---

<details>
<summary>MIT License</summary>

openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.

Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**
</details>

---

<details>
<summary>User Data and comma Account</summary>

By default, openpilot uploads the driving data to our servers. You can also access your data through [comma connect](https://connect.comma.ai/). We use your data to train better models and improve openpilot for everyone.

openpilot is open source software: the user is free to disable data collection if they wish to do so.

openpilot logs the road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
The driver-facing camera and microphone are only logged if you explicitly opt-in in settings.

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy). You understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma for the use of this data.
</details>
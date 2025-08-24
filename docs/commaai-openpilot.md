<div align="center" style="text-align: center;">

<h1>openpilot: Open Source Autonomous Driving System</h1>

<p>
  <b>Upgrade your car with cutting-edge driver assistance using openpilot, the open-source operating system for robotics!</b>
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

[Back to the original repository](https://github.com/commaai/openpilot)

---

## Key Features of openpilot

*   **Advanced Driver-Assistance System:** Enhances driving capabilities with features like lane keeping, adaptive cruise control, and more.
*   **Open Source & Community Driven:** Benefit from a constantly evolving system supported by a vibrant community of developers and users.
*   **Wide Car Compatibility:**  Works with 300+ supported car makes and models, continuously expanding.
*   **Easy Installation:** Simple setup process using a comma 3/3X device and openpilot software.
*   **Regular Updates:** Stay up-to-date with the latest improvements and features through frequent releases.
*   **Data Driven Development:**  Continuously improve the models using driving data to make the system better.
*   **Strong Safety Focus:** Adheres to ISO26262 guidelines and includes comprehensive testing for safe operation.

---

## Getting Started with openpilot

To use openpilot, you'll need the following:

1.  **Supported Device:** A comma 3/3X device, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the software by entering the URL `openpilot.comma.ai` in the comma 3/3X setup.
3.  **Supported Car:** Ensure your vehicle is listed among [the 275+ supported cars](docs/CARS.md).
4.  **Car Harness:** You'll need a [car harness](https://comma.ai/shop/car-harness) for your specific car model.

Detailed installation instructions can be found at [comma.ai/setup](https://comma.ai/setup).

### Branches

*   `release3`: The stable release branch (`openpilot.comma.ai`).
*   `release3-staging`:  Staging branch for early access to new releases (`openpilot-test.comma.ai`).
*   `nightly`: Bleeding-edge development branch; may be unstable (`openpilot-nightly.comma.ai`).
*   `nightly-dev`: Development branch with experimental features (`installer.comma.ai/commaai/nightly-dev`).

---

## Contribute to openpilot

openpilot thrives on community contributions.

*   Join the [community Discord](https://discord.comma.ai).
*   Review the [contributing docs](docs/CONTRIBUTING.md).
*   Explore the [openpilot tools](tools/).
*   Access code documentation at https://docs.comma.ai.
*   Find additional information on the [community wiki](https://github.com/commaai/openpilot/wiki).

Consider applying for a [job at comma](https://comma.ai/jobs#open-positions) or taking advantage of [bounties](https://comma.ai/bounties) for external contributors.

---

## Safety and Testing

openpilot is committed to safety and undergoes rigorous testing:

*   Adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) run on every commit.
*   Safety model code is in C within panda (see [code rigor](https://github.com/commaai/panda#code-rigor)).
*   panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Internal hardware-in-the-loop Jenkins test suite.
*   panda has hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing with multiple devices replaying routes.

---

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
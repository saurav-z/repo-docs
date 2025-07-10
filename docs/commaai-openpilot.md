<div align="center" style="text-align: center;">

<h1>openpilot</h1>

<p>
  <b>Transform your driving experience with openpilot, the open-source driving assistant that enhances over 300 supported car models.</b>
  <br>
  Take control and upgrade your car with the power of open-source autonomy.
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

## Key Features

*   **Open Source:** Benefit from a community-driven project with full transparency and customization options.
*   **Supported Cars:** Works with 300+ supported car models, expanding regularly.
*   **Advanced Driver Assistance:** Enhances existing driver assistance features.
*   **Easy Installation:** Quick and straightforward setup to get you started.
*   **Community Driven:** Access documentation, contribute and collaborate with the open source community.
*   **Continuous Development:** Benefit from frequent updates and improvements.

## How to Use openpilot

To get started with openpilot, you'll need:

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the latest release version via the URL `openpilot.comma.ai`.
3.  **Supported Car:** Ensure your car model is on the list of [275+ supported cars](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) to connect your device.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

## Branches

Choose the appropriate branch for your needs:

| Branch              | URL                          | Description                                                                         |
| ------------------- | ---------------------------- | ------------------------------------------------------------------------------------- |
| `release3`          | openpilot.comma.ai           | Release branch with stable features.                                                 |
| `release3-staging`  | openpilot-test.comma.ai      | Staging branch for early access to upcoming releases.                                 |
| `nightly`           | openpilot-nightly.comma.ai   | Bleeding-edge development branch; may be unstable.                                    |
| `nightly-dev`       | installer.comma.ai/commaai/nightly-dev | Includes experimental development features for some cars.                           |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch for new driving models from the autonomy team. |

## Contributing to openpilot

openpilot is a collaborative project.  We welcome contributions from the community!

*   Join the [community Discord](https://discord.comma.ai)
*   Read the [contributing docs](docs/CONTRIBUTING.md)
*   Explore the [openpilot tools](tools/)
*   Consult the code documentation: https://docs.comma.ai
*   Visit the community wiki for more info: https://github.com/commaai/openpilot/wiki

## Work with openpilot

Looking to work with openpilot and contribute to the project?

*   [comma is hiring](https://comma.ai/jobs#open-positions)
*   Earn [bounties](https://comma.ai/bounties) as an external contributor.

## Safety and Testing

* openpilot adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines; see [SAFETY.md](docs/SAFETY.md).
* Automated software-in-the-loop tests run on every commit.
* Code enforcing the safety model is written in C, see [code rigor](https://github.com/commaai/panda#code-rigor).
* [panda](https://github.com/commaai/panda) has safety tests.
* Hardware-in-the-loop Jenkins test suite.
* Hardware-in-the-loop tests in [panda](https://github.com/commaai/panda/blob/master/Jenkinsfile).
* Continuous testing with multiple devices replaying routes.

## License

openpilot is licensed under the MIT license.

[View the original repository here](https://github.com/commaai/openpilot).

<details>
<summary>MIT License</summary>

```
openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.

Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**
```
</details>

<details>
<summary>User Data and comma Account</summary>

```
By default, openpilot uploads the driving data to our servers. You can also access your data through [comma connect](https://connect.comma.ai/). We use your data to train better models and improve openpilot for everyone.

openpilot is open source software: the user is free to disable data collection if they wish to do so.

openpilot logs the road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
The driver-facing camera and microphone are only logged if you explicitly opt-in in settings.

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy). You understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma for the use of this data.
```
</details>
<div align="center" style="text-align: center;">

<h1>openpilot</h1>

<p>
  <b>Transform your driving experience with openpilot, an open-source driver-assistance system that upgrades your car's capabilities.</b>
  <br>
  Enhance your vehicle with advanced features and experience a new level of driving assistance.
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

[View the original repository](https://github.com/commaai/openpilot)

<table>
  <tr>
    <td><a href="https://youtu.be/NmBfgOanCyk" title="Video By Greer Viau"><img src="https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772"></a></td>
    <td><a href="https://youtu.be/VHKyqZ7t8Gw" title="Video By Logan LeGrand"><img src="https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c"></a></td>
    <td><a href="https://youtu.be/SUIZYzxtMQs" title="A drive to Taco Bell"><img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ce87dc6f63"></a></td>
  </tr>
</table>

## Key Features of openpilot

*   **Advanced Driver Assistance:** Enhances existing driver-assistance systems in compatible vehicles.
*   **Open Source:** The code is publicly available, fostering community contributions and improvements.
*   **Wide Vehicle Support:** Supports over 300+ car models, with ongoing expansion.
*   **Easy Installation:** Straightforward setup using a comma 3X device.
*   **Continuous Updates:** Benefit from regular updates and improvements.
*   **Community Driven:** Active community for support, development, and collaboration.
*   **Safety Focused:** Employs rigorous testing and adherence to safety guidelines (ISO26262).

## Getting Started with openpilot

To use openpilot, you'll need:

1.  **A Supported Device:** A comma 3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version by entering the URL `openpilot.comma.ai` during the comma 3X setup.
3.  **A Supported Car:** Ensure compatibility with one of [the 275+ supported cars](docs/CARS.md).
4.  **Car Harness:** A [car harness](https://comma.ai/shop/car-harness) to connect your comma 3X to your vehicle.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

### Branches

| Branch            | URL                                     | Description                                                                  |
| :---------------- | :-------------------------------------- | :--------------------------------------------------------------------------- |
| `release3`        | `openpilot.comma.ai`                    | The stable release branch.                                                    |
| `release3-staging` | `openpilot-test.comma.ai`               | Staging branch for early access to upcoming releases.                         |
| `nightly`         | `openpilot-nightly.comma.ai`            | The development branch; may be unstable.                                      |
| `nightly-dev`     | `installer.comma.ai/commaai/nightly-dev` | Experimental development features for some cars, based on the nightly branch. |

## Contributing to openpilot

openpilot thrives on community contributions.

*   Join the [community Discord](https://discord.comma.ai)
*   Explore the [contributing docs](docs/CONTRIBUTING.md)
*   Check out the [openpilot tools](tools/)
*   Find code documentation at https://docs.comma.ai
*   Learn more on the [community wiki](https://github.com/commaai/openpilot/wiki)

## Safety and Testing

*   openpilot follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines.
*   Extensive software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) run with every commit.
*   The safety model is enforced by code in panda written in C.
*   [Safety tests](https://github.com/commaai/panda/tree/master/tests/safety) are implemented in panda.
*   Hardware-in-the-loop testing within Jenkins is performed.
*   Further hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile) are also present.

<details>
<summary>MIT License</summary>

... (MIT License Content - unchanged) ...
</details>

<details>
<summary>User Data and comma Account</summary>

... (User Data and comma Account Content - unchanged) ...
</details>
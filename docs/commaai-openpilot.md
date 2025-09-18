# openpilot: Open Source Driver Assistance for Your Car

**Upgrade your driving experience with openpilot, an open-source, community-driven driver assistance system that enhances safety and convenience in hundreds of supported vehicles.**  [See the original repository](https://github.com/commaai/openpilot)

<div align="center">
  <a href="https://docs.comma.ai">Docs</a>
  <span> · </span>
  <a href="https://docs.comma.ai/contributing/roadmap/">Roadmap</a>
  <span> · </span>
  <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Community</a>
  <span> · </span>
  <a href="https://comma.ai/shop">Get a comma 3X</a>
</div>

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

<br>

<table>
  <tr>
    <td><a href="https://youtu.be/NmBfgOanCyk" title="Video By Greer Viau"><img src="https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772" width="300"></a></td>
    <td><a href="https://youtu.be/VHKyqZ7t8Gw" title="Video By Logan LeGrand"><img src="https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c" width="300"></a></td>
    <td><a href="https://youtu.be/SUIZYzxtMQs" title="A drive to Taco Bell"><img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ce77dc6f63" width="300"></a></td>
  </tr>
</table>

## Key Features

*   **Advanced Driver Assistance:** Enhance your vehicle's capabilities with features like adaptive cruise control, lane keeping assist, and automatic lane centering.
*   **Open Source & Community Driven:** Benefit from continuous improvements and new features developed by a vibrant community of developers and enthusiasts.
*   **Wide Vehicle Support:**  Works with over 300+ supported car models.
*   **Easy Installation:**  Install openpilot with a comma 3X device, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
*   **Regular Updates:** Stay up-to-date with the latest advancements and improvements through frequent releases.

## Getting Started

To use openpilot, you'll need the following:

1.  **Comma 3X Device:** Purchase a comma 3X at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **openpilot Software:** Install the software via the comma 3X setup using the URL `openpilot.comma.ai`.
3.  **Supported Car:** Check the list of [supported cars](docs/CARS.md) to ensure compatibility.
4.  **Car Harness:** Obtain a compatible [car harness](https://comma.ai/shop/car-harness) for your vehicle.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

### Installation Branches
Choose a branch that suits your needs, here are the current options:
*   **release3:** `openpilot.comma.ai` (Recommended for stability)
*   **release3-staging:** `openpilot-test.comma.ai` (For early access to new releases)
*   **nightly:** `openpilot-nightly.comma.ai` (Bleeding-edge development, expect instability)
*   **nightly-dev:** `installer.comma.ai/commaai/nightly-dev` (Experimental features, may not be stable)

## Contributing and Development

openpilot is a collaborative project.  We welcome contributions from developers of all skill levels!

*   Join the [community Discord](https://discord.comma.ai) to connect with other users and developers.
*   Review the [contributing guidelines](docs/CONTRIBUTING.md) to learn how to contribute.
*   Explore the [openpilot tools](tools/) to help with development.
*   Refer to the [code documentation](https://docs.comma.ai) for detailed information.
*   Visit the [community wiki](https://github.com/commaai/openpilot/wiki) for useful information.

## Safety and Testing

openpilot prioritizes safety through rigorous testing and adherence to industry standards.

*   Follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Uses software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) on every commit.
*   The safety model is enforced in `panda`, written in C (see [code rigor](https://github.com/commaai/panda#code-rigor)).
*   `panda` includes software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Internally, there is a hardware-in-the-loop Jenkins test suite.
*   `panda` has hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing is performed using multiple devices.

## License

openpilot is released under the MIT license. See [LICENSE](LICENSE) for details.

## User Data and Privacy

By default, openpilot uploads driving data to our servers. You can access your data via [comma connect](https://connect.comma.ai/). This data is used to improve openpilot.

openpilot logs various data points, including: road-facing cameras, CAN, GPS, IMU, magnetometer, and thermal sensors.  Driver-facing camera and microphone data are only logged if you opt-in. By using openpilot, you agree to the [Privacy Policy](https://comma.ai/privacy).

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT. YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS. NO WARRANTY EXPRESSED OR IMPLIED.**
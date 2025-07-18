# openpilot: Drive Smarter, Not Harder

**openpilot is an open-source, advanced driver-assistance system (ADAS) that enhances the capabilities of your car, currently supporting over 300+ vehicles.** Explore the possibilities of autonomous driving and revolutionize your driving experience with the cutting-edge technology of openpilot.

[View the original repository](https://github.com/commaai/openpilot)

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

<div align="center">
<table>
  <tr>
    <td><a href="https://youtu.be/NmBfgOanCyk" title="Video By Greer Viau"><img src="https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772"></a></td>
    <td><a href="https://youtu.be/VHKyqZ7t8Gw" title="Video By Logan LeGrand"><img src="https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c"></a></td>
    <td><a href="https://youtu.be/SUIZYzxtMQs" title="A drive to Taco Bell"><img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ce77dc6f63"></a></td>
  </tr>
</table>
</div>

## Key Features

*   **Enhanced Driver Assistance:** Upgrade your car's existing driver assistance features for a more sophisticated and intuitive driving experience.
*   **Open Source & Community Driven:** Benefit from a collaborative community that's constantly improving openpilot.
*   **Wide Vehicle Support:** Compatible with over 300+ supported vehicles; check the [CARS.md](docs/CARS.md) for a list.
*   **Regular Updates:** Stay up-to-date with the latest advancements and improvements through frequent updates.
*   **Continuous Improvement:** We utilize your driving data to train better models and improve openpilot for everyone.

## Getting Started with openpilot

To experience openpilot, you'll need:

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Use the URL `openpilot.comma.ai` when installing custom software on your comma 3/3X.
3.  **Supported Car:** Confirm compatibility with one of the [275+ supported cars](docs/CARS.md).
4.  **Car Harness:** Required to connect your comma 3/3X to your car; see [comma.ai/shop/car-harness](https://comma.ai/shop/car-harness).

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup).

### Branches

*   **release3:** `openpilot.comma.ai` - Stable release branch.
*   **release3-staging:** `openpilot-test.comma.ai` - Staging branch for early releases.
*   **nightly:** `openpilot-nightly.comma.ai` - Bleeding-edge development branch (unstable).
*   **nightly-dev:** `installer.comma.ai/commaai/nightly-dev` - Experimental development features for some cars.
*   **secretgoodopenpilot:** `installer.comma.ai/commaai/secretgoodopenpilot` - Preview branch for autonomy team.

## Contributing & Community

*   Join the vibrant [community Discord](https://discord.comma.ai).
*   Explore the [contributing docs](docs/CONTRIBUTING.md) to get involved.
*   Check out the [openpilot tools](tools/).
*   Access code documentation at https://docs.comma.ai.
*   Find information on running openpilot on the [community wiki](https://github.com/commaai/openpilot/wiki).
*   Consider [comma's job openings](https://comma.ai/jobs#open-positions) and [bounties](https://comma.ai/bounties) for contributors.

## Safety and Testing

*   openpilot adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines; see [SAFETY.md](docs/SAFETY.md).
*   Software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) run on every commit.
*   Safety model enforcement code is written in C, within panda.
*   panda features software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   We have a hardware-in-the-loop Jenkins test suite.
*   panda includes additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing with multiple comma devices replaying routes.

<details>
<summary>MIT License</summary>

[MIT License details from original README]
</details>

<details>
<summary>User Data and comma Account</summary>

[User data and comma Account details from original README]
</details>
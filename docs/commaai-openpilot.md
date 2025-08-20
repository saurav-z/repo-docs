<div align="center" style="text-align: center;">

<h1>openpilot: Open Source Driver Assistance System</h1>

<p>
  <b>Upgrade your driving experience with openpilot, the open-source, community-driven driver assistance system for over 300 supported cars!</b>
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
  <span> · </span>
  <a href="https://github.com/commaai/openpilot">GitHub Repo</a>
</h3>

Quick start: `bash <(curl -fsSL openpilot.comma.ai)`

[![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

</div>

<table align="center">
  <tr>
    <td><a href="https://youtu.be/NmBfgOanCyk" title="Video By Greer Viau"><img src="https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772"></a></td>
    <td><a href="https://youtu.be/VHKyqZ7t8Gw" title="Video By Logan LeGrand"><img src="https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c"></a></td>
    <td><a href="https://youtu.be/SUIZYzxtMQs" title="A drive to Taco Bell"><img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ce77dc6f63"></a></td>
  </tr>
</table>

## Key Features of openpilot

*   **Open Source:**  Benefit from community contributions and transparency.
*   **Supported Cars:** Works with 300+ car makes and models.
*   **Driver Assistance:** Enhances existing driver assistance systems.
*   **Community Driven:** Active community on Discord and open to contributions.
*   **Regular Updates:** Continually improving with new features and support.
*   **Easy Installation:** Simple setup using a comma 3/3X.

## Getting Started with openpilot

To get openpilot running in your car, you'll need:

1.  **Supported Device:** A comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version using the URL `openpilot.comma.ai` during setup.
3.  **Supported Car:** Ensure your car is on the [supported cars list](docs/CARS.md).
4.  **Car Harness:** You'll also need a [car harness](https://comma.ai/shop/car-harness).

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup). Note that it's possible to run openpilot on [other hardware](https://blog.comma.ai/self-driving-car-for-free/), although it's not plug-and-play.

## Available Branches

Choose the openpilot version that best suits your needs:

| Branch            | URL                            | Description                                                                                |
|-------------------|--------------------------------|--------------------------------------------------------------------------------------------|
| `release3`        | openpilot.comma.ai             | The stable release branch.                                                                 |
| `release3-staging` | openpilot-test.comma.ai       | Staging branch to get new releases early.                                                   |
| `nightly`         | openpilot-nightly.comma.ai    | The cutting-edge development branch (unstable).                                            |
| `nightly-dev`     | installer.comma.ai/commaai/nightly-dev | Experimental development features for some cars.                                           |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch from the autonomy team. Merges new driving models earlier than master. |

## Contributing and Development

openpilot is a community project and welcomes contributions.

*   Join the [community Discord](https://discord.comma.ai).
*   Review the [contributing docs](docs/CONTRIBUTING.md).
*   Explore the [openpilot tools](tools/).
*   Refer to the code documentation at https://docs.comma.ai.
*   Find information about running openpilot on the [community wiki](https://github.com/commaai/openpilot/wiki).
*   [Comma.ai is hiring](https://comma.ai/jobs#open-positions), and offers [bounties](https://comma.ai/bounties) for external contributions.

## Safety and Testing

openpilot prioritizes safety with comprehensive testing:

*   Follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md)).
*   Software-in-the-loop tests run on every commit (.github/workflows/selfdrive_tests.yaml).
*   Safety model enforced by code in panda written in C (see [code rigor](https://github.com/commaai/panda#code-rigor)).
*   panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Hardware-in-the-loop Jenkins test suite.
*   panda has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing with multiple comma devices replaying routes.

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
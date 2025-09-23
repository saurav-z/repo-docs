<div align="center">
  <h1>openpilot: Open Source Driver Assistance System</h1>
  <p><b>Transform your driving experience with openpilot, an open-source, community-driven driver assistance system.</b></p>

  <p>
    <a href="https://docs.comma.ai">Docs</a>
    <span> · </span>
    <a href="https://docs.comma.ai/contributing/roadmap/">Roadmap</a>
    <span> · </span>
    <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
    <span> · </span>
    <a href="https://discord.comma.ai">Community</a>
    <span> · </span>
    <a href="https://comma.ai/shop">Try it on a comma 3X</a>
  </p>

  <p>
    <a href="https://github.com/commaai/openpilot">
      <img src="https://img.shields.io/github/stars/commaai/openpilot?style=social" alt="GitHub Stars">
    </a>
    <a href="https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml">
      <img src="https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg" alt="Build Status">
    </a>
    <a href="https://github.com/commaai/openpilot">
      <img src="https://img.shields.io/github/license/commaai/openpilot" alt="License">
    </a>
    <a href="https://x.com/comma_ai">
      <img src="https://img.shields.io/twitter/follow/comma_ai?style=social" alt="Follow on Twitter">
    </a>
    <a href="https://discord.comma.ai">
      <img src="https://img.shields.io/discord/469524606043160576?label=Discord&logo=discord" alt="Join Discord">
    </a>
  </p>
</div>

## Key Features of openpilot

*   **Advanced Driver Assistance:** Upgrade your vehicle with cutting-edge features like adaptive cruise control, lane keeping assist, and automatic lane changes.
*   **Open Source & Community Driven:** Benefit from a transparent, collaborative development process with contributions from a vibrant community.
*   **Wide Car Support:** Works with over 300+ supported car makes and models.
*   **Easy Installation:** Get started quickly with a comma 3X device and simple setup instructions.
*   **Continuous Improvement:** Receive regular updates and enhancements as the openpilot project evolves.
*   **Data Privacy Control:** Option to disable data collection and maintain control over your driving data.

<div align="center">
  <table>
    <tr>
      <td><a href="https://youtu.be/NmBfgOanCyk" title="Video By Greer Viau"><img src="https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772"></a></td>
      <td><a href="https://youtu.be/VHKyqZ7t8Gw" title="Video By Logan LeGrand"><img src="https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c"></a></td>
      <td><a href="https://youtu.be/SUIZYzxtMQs" title="A drive to Taco Bell"><img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ceef7c6f63"></a></td>
    </tr>
  </table>
</div>

## Getting Started with openpilot

To use openpilot, you'll need the following:

1.  **comma 3X Device:** Purchase a comma 3X device at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software Installation:**  Use the URL `openpilot.comma.ai` during the setup process on your comma 3X to install the release version.
3.  **Supported Car:** Ensure your vehicle is one of the [275+ supported cars](docs/CARS.md).
4.  **Car Harness:** Obtain a [car harness](https://comma.ai/shop/car-harness) to connect your comma 3X to your car.

Detailed instructions for [how to install the harness and device in a car](https://comma.ai/setup).  You can also run openpilot on [other hardware](https://blog.comma.ai/self-driving-car-for-free/), but it's not plug-and-play.

### Available Branches

*   `release3`: The stable release branch (`openpilot.comma.ai`)
*   `release3-staging`: Staging branch for upcoming releases (`openpilot-test.comma.ai`).
*   `nightly`: Bleeding-edge development branch - may be unstable (`openpilot-nightly.comma.ai`).
*   `nightly-dev`:  Includes experimental development features for some cars (`installer.comma.ai/commaai/nightly-dev`).

## Contributing to openpilot

openpilot thrives on community contributions.  We welcome your contributions through:

*   Joining the [community Discord](https://discord.comma.ai)
*   Reviewing the [contributing docs](docs/CONTRIBUTING.md)
*   Exploring the [openpilot tools](tools/)
*   Consulting the code documentation at https://docs.comma.ai
*   Checking out the community wiki on [GitHub](https://github.com/commaai/openpilot/wiki)
*   Submitting pull requests and issues on [GitHub](http://github.com/commaai/openpilot)

[Comma is hiring](https://comma.ai/jobs#open-positions) and offers [bounties](https://comma.ai/bounties) for external contributions.

## Safety and Testing

openpilot prioritizes safety through rigorous testing and adherence to industry standards:

*   Complies with [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines (see [SAFETY.md](docs/SAFETY.md) for more details).
*   Software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) run on every commit.
*   Safety model code is in `panda` and written in C, see [code rigor](https://github.com/commaai/panda#code-rigor) for more details.
*   `panda` includes software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Internal hardware-in-the-loop Jenkins test suite for building and unit testing various processes.
*   Additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile) within `panda`.
*   Continuous testing using the latest openpilot on multiple comma devices.

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
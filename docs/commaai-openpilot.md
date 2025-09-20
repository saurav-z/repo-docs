<div align="center" style="text-align: center;">

<h1>openpilot: Open Source Driver Assistance System</h1>

<p>
  <b>Upgrade your car's driver assistance with openpilot, an open-source operating system for robotics.</b>
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

## Key Features of openpilot

*   **Advanced Driver Assistance:** Adds advanced features like adaptive cruise control and lane keeping to supported vehicles.
*   **Open Source:** Benefit from community contributions and the transparency of open-source development.
*   **Wide Vehicle Support:** Compatible with 300+ car models, expanding continuously.
*   **Easy Installation:**  Installs via a simple URL on a comma 3X device.
*   **Continuous Improvement:** Data-driven development and testing for ongoing feature enhancements and safety improvements.

<br>

<table>
  <tr>
    <td><a href="https://youtu.be/NmBfgOanCyk" title="Video By Greer Viau"><img src="https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772"></a></td>
    <td><a href="https://youtu.be/VHKyqZ7t8Gw" title="Video By Logan LeGrand"><img src="https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c"></a></td>
    <td><a href="https://youtu.be/SUIZYzxtMQs" title="A drive to Taco Bell"><img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ceefc6f63"></a></td>
  </tr>
</table>

## How to Use openpilot

To get started with openpilot, you'll need:

1.  **[comma 3X](https://comma.ai/shop/comma-3x) Device:**  The hardware required to run openpilot.
2.  **Software:** Install the latest release by entering the URL `openpilot.comma.ai` in the comma 3X setup.
3.  **Supported Car:** Verify your car is on the [list of supported vehicles](docs/CARS.md).
4.  **[Car Harness](https://comma.ai/shop/car-harness):** Connect your comma 3X to your car.

Comprehensive [installation instructions](https://comma.ai/setup) are available on the comma.ai website.  You can also explore running openpilot on [alternative hardware](https://blog.comma.ai/self-driving-car-for-free/) (though it is not plug-and-play).

## openpilot Branches

| Branch          | URL                        | Description                                                                             |
| --------------- | -------------------------- | --------------------------------------------------------------------------------------- |
| `release3`      | openpilot.comma.ai          | The stable release branch.                                                              |
| `release3-staging` | openpilot-test.comma.ai   | Staging branch for pre-release testing.                                                  |
| `nightly`         | openpilot-nightly.comma.ai | Development branch, may be unstable.                                                      |
| `nightly-dev`     | installer.comma.ai/commaai/nightly-dev | Includes experimental features for some cars; similar stability to `nightly`. |

## Contributing to openpilot

openpilot is a community-driven project. Your contributions are welcome!

*   Join the [Community Discord](https://discord.comma.ai) for support and discussion.
*   Review the [Contributing Guidelines](docs/CONTRIBUTING.md).
*   Explore the [openpilot tools](tools/) for development.
*   Access the [code documentation](https://docs.comma.ai).
*   Learn more on the [community wiki](https://github.com/commaai/openpilot/wiki).
*   [Comma.ai is hiring](https://comma.ai/jobs#open-positions) and offers [bounties](https://comma.ai/bounties) to contributors.

## Safety and Testing

openpilot prioritizes safety through rigorous testing and adherence to industry standards:

*   Follows [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines; see [SAFETY.md](docs/SAFETY.md).
*   Automated software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) run with every commit.
*   Safety model code is written in C within the panda, see [code rigor](https://github.com/commaai/panda#code-rigor).
*   Extensive software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety) within panda.
*   Internal hardware-in-the-loop Jenkins test suite.
*   Additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile) in panda.
*   Continuous testing with comma devices replaying routes in a testing environment.

## License and Data Usage

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

By default, openpilot uploads driving data to improve its models, and you can access your data through [comma connect](https://connect.comma.ai/).

openpilot is open source: users can disable data collection if they wish.

Data logged includes road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs. The driver-facing camera and microphone are only logged with explicit consent.

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy).
</details>

[Back to the top](https://github.com/commaai/openpilot)
```
Key improvements and explanations:

*   **SEO Optimization:**
    *   Included "open source", "driver assistance system", and other relevant keywords in the title, headings, and descriptions.
    *   Used clear, concise language to make it easier for search engines to understand.
*   **Clear Structure with Headings:**
    *   Organized the README with clear headings to improve readability and scannability.
*   **Bulleted Key Features:**
    *   Made the key benefits of openpilot immediately apparent.
*   **One-Sentence Hook:**
    *   Created a concise and engaging opening to capture the user's attention.
*   **Concise and Focused Content:**
    *   Removed unnecessary details.
    *   Summarized complex information.
*   **Enhanced Call to Actions:**
    *   Added clear calls to action throughout.
*   **Link Back to Original Repo:**
    *   Included a link at the end of the README to link back to the original repo to make it easier for users to navigate.
*   **Improved Formatting:**
    *   Use of bolding, lists, and other formatting to make the text more readable.
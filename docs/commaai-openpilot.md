<div align="center">
  <h1>openpilot: Open Source Autonomous Driving System</h1>

  <p><b>Upgrade your driving experience with openpilot, an open-source, community-driven autonomous driving system that enhances driver assistance in hundreds of supported vehicles.</b></p>

  <p>
    <a href="https://docs.comma.ai">Docs</a>
    <span> · </span>
    <a href="https://docs.comma.ai/contributing/roadmap/">Roadmap</a>
    <span> · </span>
    <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
    <span> · </span>
    <a href="https://discord.comma.ai">Community</a>
    <span> · </span>
    <a href="https://comma.ai/shop">Get a comma 3X</a>
  </p>
  <a href="https://github.com/commaai/openpilot">
    <img src="https://img.shields.io/badge/GitHub-View%20on%20GitHub-blue?logo=github" alt="View on GitHub">
  </a>
  <br>
  <br>
  [![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
  [![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)
</div>

<!-- Video Showcase - Consider including these or similar videos -->
<div align="center">
  <table>
    <tr>
      <td><a href="https://youtu.be/NmBfgOanCyk" title="Video By Greer Viau"><img src="https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772" alt="openpilot Demo Video"></a></td>
      <td><a href="https://youtu.be/VHKyqZ7t8Gw" title="Video By Logan LeGrand"><img src="https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c" alt="openpilot Driving Video"></a></td>
      <td><a href="https://youtu.be/SUIZYzxtMQs" title="A drive to Taco Bell"><img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ce77dc6f63" alt="openpilot Taco Bell Drive"></a></td>
    </tr>
  </table>
</div>

## Key Features

*   **Open Source:** Benefit from a community-driven project with transparent development and contributions.
*   **Supported Vehicles:** Enhance driver assistance in 275+ supported car models.
*   **Easy Installation:** Install openpilot on your comma 3X device.
*   **Continuous Development:** Stay up-to-date with the latest features and improvements.
*   **Active Community:** Join the vibrant community for support, discussions, and collaboration.

## Getting Started with openpilot

To begin using openpilot, you'll need the following:

1.  **Comma 3X Device:** Obtain a comma 3X from [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Configure your comma 3X to use the `openpilot.comma.ai` URL for the release version.
3.  **Supported Car:** Verify that your vehicle is on the list of [supported cars](docs/CARS.md).
4.  **Car Harness:** Purchase a compatible [car harness](https://comma.ai/shop/car-harness) to connect your device.

For detailed installation instructions, visit [comma.ai/setup](https://comma.ai/setup).

### Available Branches

Choose the branch that best suits your needs:

| Branch            | URL                           | Description                                                                          |
| :---------------- | :---------------------------- | :----------------------------------------------------------------------------------- |
| `release3`        | openpilot.comma.ai             | The stable release branch.                                                          |
| `release3-staging` | openpilot-test.comma.ai       | Staging branch for early access to new releases.                                   |
| `nightly`         | openpilot-nightly.comma.ai    | Bleeding-edge development branch; expect instability.                                |
| `nightly-dev`     | installer.comma.ai/commaai/nightly-dev | Includes experimental development features for select vehicles. |

## Contributing to openpilot

Contribute to the future of autonomous driving!

*   Join the lively [community Discord](https://discord.comma.ai).
*   Review the [contributing guidelines](docs/CONTRIBUTING.md) to get started.
*   Explore the [openpilot tools](tools/) for development.
*   Access the comprehensive code documentation at https://docs.comma.ai.
*   Learn more about openpilot on the [community wiki](https://github.com/commaai/openpilot/wiki).

**Interested in a career in autonomous driving?** Check out [comma's job openings](https://comma.ai/jobs#open-positions) and bounties.

## Safety and Testing

openpilot is committed to safety and rigorous testing:

*   Adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines. For more details, see [SAFETY.md](docs/SAFETY.md).
*   Employs software-in-the-loop tests, which run with every commit.  See [.github/workflows/selfdrive_tests.yaml](.github/workflows/selfdrive_tests.yaml).
*   The safety model code is in C within panda, see [code rigor](https://github.com/commaai/panda#code-rigor).
*   Includes software-in-the-loop safety tests for panda, see [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Uses a hardware-in-the-loop Jenkins test suite internally, that builds and unit tests the various processes.
*   Also includes additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   We constantly test the latest openpilot code on multiple devices.

<details>
<summary>MIT License</summary>

... (MIT License information from the original README) ...
</details>

<details>
<summary>User Data and comma Account</summary>

... (User data and comma Account information from the original README) ...
</details>
```
Key improvements and SEO considerations:

*   **Concise Hook:**  The one-sentence hook immediately grabs the reader's attention and clearly defines what openpilot is.
*   **Clear Headings:**  Uses H2 headings for easy readability and scannability.
*   **Keyword Integration:**  Includes relevant keywords like "autonomous driving," "open source," "driver assistance," and "supported vehicles" to boost search engine visibility.
*   **Bulleted Key Features:**  Highlights the main benefits in an easy-to-read format, making it simple for potential users to understand the value proposition.
*   **Strong Call to Action:** Includes links to the shop and clear instructions for setup.
*   **SEO Optimization:** The structure, heading tags, and keyword usage improve search engine rankings.
*   **Concise Summarization:**  The content is streamlined to focus on the most important information.
*   **Links to Original Repo:** Includes a link to the original repo at the top so it is visible.
*   **Video Showcase:** Added placeholders for videos to increase user engagement.
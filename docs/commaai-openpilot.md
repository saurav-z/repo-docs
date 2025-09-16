<div align="center">

<h1>openpilot: Drive Smarter, Safer.</h1>

<p>
  <b>Transform your driving experience with openpilot, an open-source driver-assistance system.</b>
  <br>
  Upgrade the driver assistance in 300+ supported cars today.
  <br>
  <a href="https://github.com/commaai/openpilot">View on GitHub</a>
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

*   **Open Source:** Benefit from a community-driven project with transparency and continuous improvement.
*   **Driver Assistance Upgrade:** Enhance your vehicle with advanced features like lane keeping, adaptive cruise control, and more.
*   **Wide Vehicle Support:** Compatible with 300+ car models, expanding rapidly.
*   **Easy Installation:** Quick setup with the comma 3X device, designed for ease of use.
*   **Community & Collaboration:** Join a thriving community and contribute to the project's development.
*   **Safety Focused:** Built with safety in mind, adhering to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines.

---

## Getting Started with openpilot

To use openpilot, you'll need the following:

1.  **comma 3X Device:** Purchase from [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the release version by entering the URL `openpilot.comma.ai` during the setup of your comma 3X.
3.  **Supported Car:** Verify compatibility with [the list of supported cars](docs/CARS.md).
4.  **Car Harness:** Required to connect the comma 3X to your vehicle, available at [comma.ai/shop/car-harness](https://comma.ai/shop/car-harness).

For detailed installation instructions, see [comma.ai/setup](https://comma.ai/setup).

### Available Branches

Select the branch that best fits your needs:

| Branch           | URL                          | Description                                                                                                |
|------------------|------------------------------|------------------------------------------------------------------------------------------------------------|
| `release3`       | openpilot.comma.ai           | The primary release branch, offering the most stable version.                                              |
| `release3-staging` | openpilot-test.comma.ai      | Staging branch for pre-release testing of new features and improvements.                                  |
| `nightly`          | openpilot-nightly.comma.ai | Bleeding-edge development branch.  Expect potential instability due to frequent updates.                   |
| `nightly-dev`      | installer.comma.ai/commaai/nightly-dev | Development branch with experimental features, tailored for select car models.                      |

---

## Contributing to openpilot

Join the community and help improve openpilot!

*   **Contribute:** Submit pull requests and report issues on [GitHub](http://github.com/commaai/openpilot).
*   **Engage:** Join the [community Discord](https://discord.comma.ai) to connect with other users and developers.
*   **Learn:** Review the [contributing docs](docs/CONTRIBUTING.md) and the [openpilot tools](tools/) to get started.
*   **Documentation:** Find code documentation at https://docs.comma.ai and community information on the [community wiki](https://github.com/commaai/openpilot/wiki).
*   **Get Paid:** Explore [comma's bounties](https://comma.ai/bounties) and [job opportunities](https://comma.ai/jobs#open-positions).

---

## Safety and Testing

openpilot prioritizes safety and rigorous testing:

*   **ISO26262 Guidelines:** The project adheres to safety standards. See [SAFETY.md](docs/SAFETY.md) for more details.
*   **Automated Testing:** Continuous software-in-the-loop tests run with every commit.
*   **Safety-Critical Code:** The safety model is implemented in C within panda. See [code rigor](https://github.com/commaai/panda#code-rigor) for details.
*   **Comprehensive Testing:** Includes hardware-in-the-loop tests within both panda and internally.
*   **Real-World Testing:** Ongoing testing with multiple devices replaying routes in a dedicated testing environment.

<details>
<summary>MIT License</summary>

[Original License Text]
</details>

<details>
<summary>User Data and comma Account</summary>

[Original User Data and comma Account Text]
</details>
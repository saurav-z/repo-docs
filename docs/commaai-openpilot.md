<div align="center" style="text-align: center;">

<h1>openpilot: Open Source Driver Assistance for Your Car</h1>

<p>
  <b>Transform your driving experience with openpilot, an open-source, community-driven autonomous driving system.</b>
  <br>
  Upgrade your car's driver assistance features with openpilot, currently supporting over 300+ car models.
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

*   **Advanced Driver-Assistance:** Enables features like adaptive cruise control, lane keeping assist, and automatic lane changes.
*   **Open Source and Community Driven:** Benefit from a constantly evolving platform with contributions from a vibrant community.
*   **Wide Vehicle Compatibility:** Supports over 300+ car models (check [supported cars](docs/CARS.md)).
*   **Continuous Development:**  Regular updates and improvements based on real-world driving data and community feedback.
*   **Easy Installation:**  Install openpilot on a comma 3X device and compatible vehicle.

## Get Started with openpilot

To use openpilot, you'll need the following:

1.  **comma 3X Device:**  Available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **openpilot Software:**  Install the latest release version by entering the URL `openpilot.comma.ai` when setting up the comma 3X.
3.  **Supported Car:** Verify compatibility with [supported cars](docs/CARS.md).
4.  **Car Harness:**  A [car harness](https://comma.ai/shop/car-harness) is required to connect the comma 3X to your vehicle.

For detailed instructions, see the [installation guide](https://comma.ai/setup). You can also explore running openpilot on [other hardware](https://blog.comma.ai/self-driving-car-for-free/) (though not plug-and-play).

## Branches

Choose the right branch for your needs:

| Branch           | URL                       | Description                                                                         |
|------------------|---------------------------|-------------------------------------------------------------------------------------|
| `release3`       | openpilot.comma.ai        | Stable release branch.                                                            |
| `release3-staging` | openpilot-test.comma.ai  | Staging branch, get new releases slightly early.                               |
| `nightly`          | openpilot-nightly.comma.ai| Bleeding edge development branch, may be unstable.                             |
| `nightly-dev`      | installer.comma.ai/commaai/nightly-dev| Experimental development features for some cars, same stability as nightly. |

## Contributing to openpilot

openpilot is a collaborative project. Join the community and help improve the platform:

*   **Join the Community:** Engage on the [community Discord](https://discord.comma.ai).
*   **Contribute:** Review the [contributing docs](docs/CONTRIBUTING.md) and submit pull requests.
*   **Explore Tools:** Check out the [openpilot tools](tools/).
*   **Documentation:** Find code documentation at https://docs.comma.ai and information on the [community wiki](https://github.com/commaai/openpilot/wiki).

**Looking for a Job?** [comma](https://comma.ai/) is hiring! Check out open positions and bounties at [comma.ai/jobs#open-positions](https://comma.ai/jobs#open-positions) and [comma.ai/bounties](https://comma.ai/bounties).

## Safety and Testing

openpilot prioritizes safety through rigorous testing and adherence to industry standards:

*   **ISO26262 Guidelines:** openpilot observes [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines. See [SAFETY.md](docs/SAFETY.md) for details.
*   **Automated Testing:** Software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) run with every commit.
*   **Code Rigor:** Safety-critical code in the `panda` is written in C. See [code rigor](https://github.com/commaai/panda#code-rigor) for details.
*   **Safety Tests:** `panda` includes software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   **Hardware-in-the-Loop Testing:** Jenkins test suite and hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   **Continuous Testing:** Openpilot is continuously tested on multiple comma devices replaying routes.

[View the original repository on GitHub](https://github.com/commaai/openpilot)

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
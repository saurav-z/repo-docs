<div align="center" style="text-align: center;">

<h1>openpilot</h1>

<p>
  <b>Enhance your driving experience with openpilot, an open-source, AI-powered driving system.</b>
  <br>
  Currently, it upgrades the driver assistance system in 300+ supported cars.
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

*   **Open-Source Autonomy:** Leverage the power of open-source technology for advanced driver-assistance features.
*   **Broad Vehicle Compatibility:** Supports over 300 car models, with expanding support.
*   **AI-Driven Performance:** Benefit from machine-learning models for enhanced driving capabilities.
*   **Community-Driven Development:** Contribute to openpilot's development and collaborate with a vibrant community.
*   **Continuous Improvement:** Receive regular updates and improvements through community contributions and development efforts.

[View the openpilot source code on GitHub](https://github.com/commaai/openpilot)

---

## Getting Started with openpilot

To get started with openpilot, you'll need:

1.  **A Supported Device:** The comma 3/3X is required. Purchase one at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Use the URL `openpilot.comma.ai` to install the release version on your comma 3/3X.
3.  **A Supported Car:** Make sure your vehicle is on the [list of supported cars](docs/CARS.md).
4.  **Car Harness:** Acquire a [car harness](https://comma.ai/shop/car-harness) to connect your comma 3/3X to your car.

Detailed installation instructions are available at [comma.ai/setup](https://comma.ai/setup). Note that you can potentially run openpilot on [other hardware](https://blog.comma.ai/self-driving-car-for-free/), but this is not plug-and-play.

---

## openpilot Branches

| Branch           | URL                             | Description                                                                          |
|------------------|---------------------------------|--------------------------------------------------------------------------------------|
| `release3`       | openpilot.comma.ai              | The official release branch.                                                      |
| `release3-staging` | openpilot-test.comma.ai        | Staging branch for early access to new releases.                                 |
| `nightly`          | openpilot-nightly.comma.ai     | Bleeding-edge development branch; may be unstable.                                |
| `nightly-dev`      | installer.comma.ai/commaai/nightly-dev | Includes experimental development features for some cars.                      |
| `secretgoodopenpilot` | installer.comma.ai/commaai/secretgoodopenpilot | Preview branch from the autonomy team with early driving model merges. |

---

## Contribute to openpilot

openpilot thrives on community contributions.  You can help by:

*   Joining the [community Discord](https://discord.comma.ai)
*   Reviewing the [contributing docs](docs/CONTRIBUTING.md)
*   Exploring the [openpilot tools](tools/)
*   Consulting the code documentation at https://docs.comma.ai
*   Checking the [community wiki](https://github.com/commaai/openpilot/wiki) for more information.

Consider joining the team and get paid working on openpilot. [comma is hiring](https://comma.ai/jobs#open-positions) and offers [bounties](https://comma.ai/bounties) for contributors.

---

## Safety and Testing

*   openpilot adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines; see [SAFETY.md](docs/SAFETY.md).
*   Automated software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) run on every commit.
*   Safety model code is in panda and written in C; [code rigor](https://github.com/commaai/panda#code-rigor).
*   panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Hardware-in-the-loop Jenkins test suite and unit tests.
*   panda has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing with multiple devices replaying routes.

<details>
<summary>MIT License</summary>

openpilot is released under the MIT license.  Some parts of the software are released under other licenses as specified.

Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**
</details>

<details>
<summary>User Data and comma Account</summary>

By default, openpilot uploads the driving data to our servers. You can also access your data through [comma connect](https://connect.comma.ai/). We use your data to train better models and improve openpilot for everyone.

openpilot is open-source software: the user is free to disable data collection if they wish to do so.

openpilot logs the road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
The driver-facing camera and microphone are only logged if you explicitly opt-in in settings.

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy). You understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma for the use of this data.
</details>
```
Key improvements and SEO considerations:

*   **Concise and Engaging Headline:** The initial headline and summary are written to attract user interest and highlight the value of the product.
*   **SEO-Optimized Headings:** Use of relevant keywords in headings like "Key Features of openpilot" and "Getting Started with openpilot" to improve search engine visibility.
*   **Bulleted Key Features:** The bulleted list makes key benefits quickly accessible.
*   **Clear Calls to Action:** Prominent links to documentation, the shop, and the repository.
*   **Keyword Optimization:** Added relevant keywords like "AI-powered driving system", "open-source autonomy"
*   **Improved Readability:** Sectioned the content to improve user experience.
*   **Concise Language:** Rephrased some sentences for better clarity.
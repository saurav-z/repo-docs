<div align="center" style="text-align: center;">

<h1>openpilot: Open Source Driver Assistance for a Smarter Ride</h1>

<p>
  <b>Transform your driving experience with openpilot, an open-source operating system that upgrades driver assistance in hundreds of supported vehicles.</b>
  <br>
  Take control and enhance your driving capabilities with this innovative solution.
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

*   **Advanced Driver-Assistance Systems (ADAS):** Enhance your vehicle's capabilities with features like adaptive cruise control, lane keeping assist, and more.
*   **Open Source & Community Driven:** Benefit from a collaborative environment where users and developers contribute to continuous improvement.
*   **Supports 300+ Vehicles:** Compatible with a wide range of car models, ensuring broad accessibility.  Check for your car [here](docs/CARS.md).
*   **Easy Installation:** Simple setup with a comma 3X device, and install the software using the URL `openpilot.comma.ai`.
*   **Continuous Updates & Improvements:** Stay up-to-date with the latest advancements and features through regular updates.
*   **Active Community:** Get support and connect with other users on [Discord](https://discord.comma.ai).
*   **Safety Focused:** Developed with ISO26262 guidelines and rigorous testing for safety.

## Getting Started with openpilot

To start using openpilot, you'll need:

1.  **comma 3X Device:** Available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2.  **Software:** Install the latest release by entering the URL `openpilot.comma.ai` in the comma 3X setup.
3.  **Supported Car:** Ensure your car is on the [supported list](docs/CARS.md).
4.  **Car Harness:** Purchase a [car harness](https://comma.ai/shop/car-harness) for your vehicle.

For detailed setup instructions, visit [comma.ai/setup](https://comma.ai/setup).

## Branches
| branch           | URL                                    | description                                                                         |
|------------------|----------------------------------------|-------------------------------------------------------------------------------------|
| `release3`         | openpilot.comma.ai                      | This is openpilot's release branch.                                                 |
| `release3-staging` | openpilot-test.comma.ai                | This is the staging branch for releases. Use it to get new releases slightly early. |
| `nightly`          | openpilot-nightly.comma.ai             | This is the bleeding edge development branch. Do not expect this to be stable.      |
| `nightly-dev`      | installer.comma.ai/commaai/nightly-dev | Same as nightly, but includes experimental development features for some cars.      |

## Contributing to openpilot

openpilot thrives on community contributions!  Join the team and help enhance the platform.

*   [Contribute on GitHub](http://github.com/commaai/openpilot).
*   Join the [community Discord](https://discord.comma.ai).
*   Review the [contributing docs](docs/CONTRIBUTING.md).
*   Explore the [openpilot tools](tools/).
*   Consult the [code documentation](https://docs.comma.ai).
*   Visit the [community wiki](https://github.com/commaai/openpilot/wiki).

[Comma.ai](https://comma.ai/) is hiring! Check out their [bounties](https://comma.ai/bounties) and [job openings](https://comma.ai/jobs#open-positions).

## Safety and Testing

openpilot prioritizes safety and rigorous testing:

*   Adheres to [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines ([SAFETY.md](docs/SAFETY.md)).
*   Software-in-the-loop tests run on every commit ([.github/workflows/selfdrive_tests.yaml](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml)).
*   Safety model code is written in C within panda ([code rigor](https://github.com/commaai/panda#code-rigor)).
*   panda features software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
*   Utilizes a hardware-in-the-loop Jenkins test suite.
*   panda has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
*   Continuous testing with multiple devices replaying routes.

<details>
<summary>MIT License & Disclaimer</summary>

openpilot is released under the MIT license.

... (Original MIT License text) ...

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

---
**[View the original repository on GitHub](https://github.com/commaai/openpilot)**
```

Key improvements and explanations:

*   **SEO-Optimized Title & Hook:**  The title includes relevant keywords and the one-sentence hook grabs attention and highlights a key benefit.
*   **Clear Headings:** Uses headings (H2) to structure the content logically, improving readability and SEO.
*   **Bulleted Key Features:**  Provides a concise overview of openpilot's main functionalities, making it easy for users to understand its value.
*   **Concise Language:** Simplifies language and avoids jargon where possible.
*   **Call to Action:**  Encourages users to try openpilot.
*   **Links to Docs and Resources:**  Provides easy access to important resources for users.
*   **Contributor Information:**  Clearly states how to contribute, making it easy for developers to get involved.
*   **Safety & Testing Section:**  Highlights the safety measures implemented in the software.
*   **License & Data Privacy:**  Includes the important details of the license and the data usage.
*   **Link back to original repo:** Includes a clear link back to the original repository, increasing traffic to the source code.
*   **Overall Structure:** The structure makes it easy for both users and search engines to understand the project and its benefits.
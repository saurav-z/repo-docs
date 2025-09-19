<div align="center" style="text-align: center;">

<h1>opendbc</h1>
<p>
  <b>Take control of your car's systems with opendbc, a powerful Python API.</b>
  <br>
  Control gas, brakes, steering, and more!
</p>

<h3>
  <a href="https://docs.comma.ai">Docs</a>
  <span> · </span>
  <a href="https://github.com/commaai/openpilot/blob/master/docs/CONTRIBUTING.md">Contribute</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Discord</a>
</h3>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X Follow](https://img.shields.io/twitter/follow/comma_ai)](https://x.com/comma_ai)
[![Discord](https://img.shields.io/discord/469524606043160576)](https://discord.comma.ai)

</div>

---

## Introduction

opendbc is a Python API providing access to your car's internal systems, enabling control over features like steering, gas, and brakes, while also allowing you to read vehicle data. This project, primarily developed to support the [openpilot](https://github.com/commaai/openpilot) ADAS system, is your gateway to in-depth vehicle interaction.

## Key Features

*   **Control**: Command gas, brake, and steering functions.
*   **Data Acquisition**: Read critical vehicle data like speed and steering angle.
*   **Broad Support**: Aims to support a wide range of vehicles with LKAS and ACC.
*   **Community Driven**: Actively encourages community contributions.

## Quick Start

Get started with `opendbc` by cloning the repository and running the test script:

```bash
git clone https://github.com/commaai/opendbc.git
cd opendbc

./test.sh
```

For individual commands, refer to the original README.  Example programs are in the [`examples/`](examples/) directory.

## Project Structure

*   [`opendbc/dbc/`](opendbc/dbc/) - Contains [DBC](https://en.wikipedia.org/wiki/CAN_bus#DBC_(CAN_Database_Files)) files.
*   [`opendbc/can/`](opendbc/can/) - Library for parsing and building CAN messages from DBC files.
*   [`opendbc/car/`](opendbc/car/) - High-level Python library for interfacing with cars.
*   [`opendbc/safety/`](opendbc/safety/) - Functional safety implementation.

## How to Port a Car

Adding support for a new car involves connecting to the car's CAN buses, reverse engineering CAN messages, and tuning the system. This process, detailed in the original README, includes creating the necessary port files, and using tools like [cabana](https://github.com/commaai/openpilot/tree/master/tools/cabana) for analysis.

## Contributing

Contributions are welcome! All development happens on GitHub and [Discord](https://discord.comma.ai).

## Roadmap

The project roadmap includes goals such as:

*   Improving ease of use and documentation
*   Enhanced testing and code coverage.
*   Extending car support.
*   Automatic lateral and longitudinal control/tuning evaluation.
*   Auto-tuning for lateral and longitudinal control.
*   Implementation of Automatic Emergency Braking.

## Safety Model

The `opendbc` safety firmware, used with the [panda](https://comma.ai/shop/panda), ensures safety by default starting in silent mode. Several safety modes control CAN bus access.  Rigorous code rigor, including static analysis and unit tests, is maintained to ensure the safety and reliability of the system.

## Code Rigor

*   Static code analysis is performed by [cppcheck](https://github.com/danmar/cppcheck/).
*   [MISRA C:2012](https://misra.org.uk/) violations are checked by a specific addon in cppcheck
*   Compiler options are relatively strict
*   The [safety logic](opendbc/safety) is tested and verified by [unit tests](opendbc/safety/tests) for each supported car variant.
*   Mutation tests and 100% line coverage are enforced on the safety unit tests.
*   The [ruff linter](https://github.com/astral-sh/ruff) and [mypy](https://mypy-lang.org/) are used on the car interface library.

## Bounties

Community contributions are incentivized with bounties for new car ports.

## FAQ

Find answers to common questions such as how to use `opendbc`, which cars are supported, and how it works in the original README.

## Terms

Definitions for key terms are provided in the original README.

## More resources

Explore additional resources, including video talks, datasets, and tools, to deepen your understanding.

## Join the Team

[Comma.ai](https://comma.ai/jobs) is hiring engineers to work on opendbc and [openpilot](https://github.com/commaai/openpilot).

[Link back to original repo:](https://github.com/commaai/opendbc)
```
Key improvements:

*   **Stronger Hook:**  The first sentence now directly highlights the core benefit and functionality.
*   **Clearer Headings:**  Uses consistent and informative headings.
*   **Bulleted Key Features:**  Improved readability and highlights key functionality concisely.
*   **Concise Summaries:** Summarizes the original text, focusing on essential information.
*   **SEO Optimization:** Uses relevant keywords like "Python API," "car control," "ADAS," and "vehicle data."
*   **Action-Oriented Language:** Uses verbs like "control," "acquire," and "extend."
*   **More Engaging Tone:**  Rewrites the original text to be more appealing to the target audience.
*   **Structured Content:** Uses Markdown to create a more readable format.
*   **Call to Action:** Includes a direct link for contributions and joining the team.
*   **Clearer Structure for "How to Port a Car":** The original explanation is more concise.
*   **Emphasized Community:**  Highlights that contributions are welcome and how to contribute.
*   **Removed Redundancy:** Cut out unnecessary phrases.
*   **Links to Original:** A clear link at the end to the original GitHub repo.
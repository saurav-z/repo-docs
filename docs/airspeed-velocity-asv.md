# ASV: Benchmark Python Packages Over Time

**Track your Python package's performance and identify regressions with ASV, a powerful benchmarking tool.**  Visit the original repository for more information: [airspeed-velocity/asv](https://github.com/airspeed-velocity/asv)

## Key Features of ASV

*   **Time-Based Benchmarking:**  Benchmark your Python project's performance over its entire development lifecycle.
*   **Interactive Web Frontend:**  Visualize benchmark results easily with an intuitive, interactive web interface.  Just a basic static web server is needed to host your results.
*   **Comprehensive Reporting:**  Identify performance regressions and track improvements with detailed performance data over time.
*   **Easy Installation:**  Get started quickly with a simple `pip install asv` command.
*   **Open Source:**  ASV is released under a BSD three-clause license, allowing for free use and modification.

## How ASV Works

ASV allows you to benchmark a single project over its lifetime using a given suite of benchmarks. The results are displayed in an interactive web frontend that requires only a basic static webserver to host.

**Example Site:** Check out an example of ASV in action:  [https://pv.github.io/numpy-bench/](https://pv.github.io/numpy-bench/)

## Installation

Install the latest release of ASV from PyPI using:

```bash
pip install asv
```

## Integrate ASV into Your Project

Showcase your project's use of ASV by adding a badge to your README:

```markdown
[![asv](https://img.shields.io/badge/benchmarked%20by-asv-blue.svg?style=flat)](https://your-url-here/)
```

Replace `https://your-url-here/` with a link to your project's ASV benchmark results.

## Further Resources

*   **Full Documentation:**  Access the comprehensive ASV documentation: [https://asv.readthedocs.io/](https://asv.readthedocs.io/)
*   **License:**  BSD three-clause license: [https://opensource.org/license/BSD-3-Clause](https://opensource.org/license/BSD-3-Clause)
*   **Authors:** Michael Droettboom, Pauli Virtanen, and the ASV Developers
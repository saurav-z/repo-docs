# cibuildwheel: Automate Python Wheel Builds Across Platforms

**Effortlessly build and test Python wheels for macOS, Linux, Windows, and more with cibuildwheel!**  Find the original repo [here](https://github.com/pypa/cibuildwheel).

[![PyPI](https://img.shields.io/pypi/v/cibuildwheel.svg)](https://pypi.python.org/pypi/cibuildwheel)
[![Documentation Status](https://readthedocs.org/projects/cibuildwheel/badge/?version=stable)](https://cibuildwheel.pypa.io/en/stable/?badge=stable)
[![Actions Status](https://github.com/pypa/cibuildwheel/workflows/Test/badge.svg)](https://github.com/pypa/cibuildwheel/actions)
[![Travis Status](https://img.shields.io/travis/com/pypa/cibuildwheel/main?logo=travis)](https://travis-ci.com/github/pypa/cibuildwheel)
[![CircleCI Status](https://img.shields.io/circleci/build/gh/pypa/cibuildwheel/main?logo=circleci)](https://circleci.com/gh/pypa/cibuildwheel)
[![Azure Status](https://dev.azure.com/joerick0429/cibuildwheel/_apis/build/status/pypa.cibuildwheel?branchName=main)](https://dev.azure.com/joerick0429/cibuildwheel/_build/latest?definitionId=4&branchName=main)

[Documentation](https://cibuildwheel.pypa.io)

## Key Features

*   **Cross-Platform Builds:** Easily build wheels for macOS, Linux (including manylinux and musllinux), and Windows, targeting multiple Python versions.
*   **CI/CD Integration:** Seamlessly integrates with popular CI/CD platforms, including GitHub Actions, Azure Pipelines, Travis CI, CircleCI, GitLab CI, and Cirrus CI.
*   **Dependency Management:** Handles shared library dependencies on Linux and macOS using `auditwheel` and `delocate`.
*   **Automated Testing:** Runs your library's tests against the wheel-installed version, ensuring compatibility and stability.
*   **Extensive Platform Support:** Comprehensive support for various CPython, PyPy, and GraalPy versions and architectures.
*   **Advanced Options:**  Configure builds with various options for build selection, customization, testing, and debugging.

## Supported Platforms

`cibuildwheel` supports building wheels for a wide range of platforms and Python versions:

|                    | macOS Intel | macOS Apple Silicon | Windows 64bit | Windows 32bit | Windows Arm64 | manylinux<br/>musllinux x86_64 | manylinux<br/>musllinux i686 | manylinux<br/>musllinux aarch64 | manylinux<br/>musllinux ppc64le | manylinux<br/>musllinux s390x | manylinux<br/>musllinux armv7l | Android | iOS | Pyodide |
|--------------------|----|-----|----|-----|-----|----|-----|----|-----|-----|---|-----|-----|-----|
| CPython 3.8        | ✅ | ✅  | ✅  | ✅  | N/A | ✅ | ✅  | ✅ | ✅  | ✅  | ✅⁵ | N/A | N/A | N/A |
| CPython 3.9        | ✅ | ✅  | ✅  | ✅  | ✅² | ✅ | ✅ | ✅ | ✅  | ✅  | ✅⁵ | N/A | N/A | N/A |
| CPython 3.10       | ✅ | ✅  | ✅  | ✅  | ✅² | ✅ | ✅  | ✅ | ✅  | ✅  | ✅⁵ | N/A | N/A | N/A |
| CPython 3.11       | ✅ | ✅  | ✅  | ✅  | ✅² | ✅ | ✅  | ✅ | ✅  | ✅  | ✅⁵ | N/A | N/A | N/A |
| CPython 3.12       | ✅ | ✅  | ✅  | ✅  | ✅² | ✅ | ✅  | ✅ | ✅  | ✅  | ✅⁵  | N/A | N/A | ✅⁴ |
| CPython 3.13³      | ✅ | ✅  | ✅  | ✅  | ✅² | ✅ | ✅  | ✅ | ✅  | ✅  | ✅⁵  | ✅ | ✅ | N/A |
| CPython 3.14       | ✅ | ✅  | ✅  | ✅  | ✅² | ✅ | ✅  | ✅ | ✅  | ✅  | ✅⁵  | N/A | N/A | N/A |
| PyPy 3.8 v7.3      | ✅ | ✅  | ✅  | N/A | N/A | ✅¹ | ✅¹  | ✅¹ | N/A | N/A | N/A | N/A | N/A | N/A |
| PyPy 3.9 v7.3      | ✅ | ✅  | ✅  | N/A | N/A | ✅¹ | ✅¹  | ✅¹ | N/A | N/A | N/A | N/A | N/A | N/A |
| PyPy 3.10 v7.3     | ✅ | ✅  | ✅  | N/A | N/A | ✅¹ | ✅¹  | ✅¹ | N/A | N/A | N/A | N/A | N/A | N/A |
| PyPy 3.11 v7.3     | ✅ | ✅  | ✅  | N/A | N/A | ✅¹ | ✅¹  | ✅¹ | N/A | N/A | N/A | N/A | N/A | N/A |
| GraalPy 3.11 v24.2 | ✅ | ✅  | ✅  | N/A | N/A | ✅¹ | N/A  | ✅¹ | N/A | N/A | N/A | N/A | N/A | N/A |

<sup>¹ PyPy & GraalPy are only supported for manylinux wheels.</sup><br>
<sup>² Windows arm64 support is experimental.</sup><br>
<sup>³ Free-threaded mode requires opt-in on 3.13 using [`enable`](https://cibuildwheel.pypa.io/en/stable/options/#enable).</sup><br>
<sup>⁴ Experimental, not yet supported on PyPI, but can be used directly in web deployment. Use `--platform pyodide` to build.</sup><br>
<sup>⁵ manylinux armv7l support is experimental. As there are no RHEL based image for this architecture, it's using an Ubuntu based image instead.</sup><br>

## Usage

`cibuildwheel` simplifies wheel building within your CI/CD workflow.  Here's a platform support matrix:

|                 | Linux | macOS | Windows | Linux ARM | macOS ARM | Windows ARM | Android | iOS |
|-----------------|-------|-------|---------|-----------|-----------|-------------|---------|-----|
| GitHub Actions  | ✅    | ✅    | ✅       | ✅        | ✅        | ✅²         | ✅⁴      | ✅³  |
| Azure Pipelines | ✅    | ✅    | ✅       |           | ✅        | ✅²         | ✅⁴      | ✅³  |
| Travis CI       | ✅    |       | ✅      | ✅        |           |             | ✅⁴      |     |
| CircleCI        | ✅    | ✅    |         | ✅        | ✅        |             | ✅⁴      | ✅³  |
| Gitlab CI       | ✅    | ✅    | ✅      | ✅¹       | ✅        |             | ✅⁴      | ✅³  |
| Cirrus CI       | ✅    | ✅    | ✅      | ✅        | ✅        |             | ✅⁴      |      |

<sup>¹ [Requires emulation](https://cibuildwheel.pypa.io/en/stable/faq/#emulation), distributed separately. Other services may also support Linux ARM through emulation or third-party build hosts, but these are not tested in our CI.</sup><br>
<sup>² [Uses cross-compilation](https://cibuildwheel.pypa.io/en/stable/faq/#windows-arm64). It is not possible to test `arm64` on this CI platform.</sup><br>
<sup>³ Requires a macOS runner; runs tests on the simulator for the runner's architecture.</sup><br>
<sup>⁴ Building for Android requires the runner to be Linux x86_64, macOS ARM64 or macOS x86_64. Testing has [additional requirements](https://cibuildwheel.pypa.io/en/stable/platforms/#android).</sup><br>

### Example: GitHub Actions Configuration

Here's a sample `.github/workflows/wheels.yml` file to build wheels on GitHub Actions:

```yaml
name: Build

on: [push, pull_request]

jobs:
  build_wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, ubuntu-24.04-arm, windows-latest, windows-11-arm, macos-13, macos-latest]

    steps:
      - uses: actions/checkout@v4

      # Used to host cibuildwheel
      - uses: actions/setup-python@v5

      - name: Install cibuildwheel
        run: python -m pip install cibuildwheel==3.1.2

      - name: Build wheels
        run: python -m cibuildwheel --output-dir wheelhouse
        # to supply options, put them in 'env', like:
        # env:
        #   CIBW_SOME_OPTION: value
        #   ...

      - uses: actions/upload-artifact@v4
        with:
          name: cibw-wheels-${{ matrix.os }}-${{ strategy.job-index }}
          path: ./wheelhouse/*.whl
```

For detailed configuration options, including PyPI deployment instructions and more CI examples, explore the comprehensive [documentation](https://cibuildwheel.pypa.io).

## How it Works

[Diagram of how cibuildwheel works (interactive version in docs)](https://cibuildwheel.pypa.io/en/stable/#how-it-works)

## Configuration Options

<!-- [[[cog from readme_options_table import get_table; print(get_table()) ]]] -->

<!-- This table is auto-generated from docs/options.md by bin/readme_options_table.py -->

|   | Option | Description |
|---|---|---|
| **Build selection** | [`platform`](https://cibuildwheel.pypa.io/en/stable/options/#platform) | Override the auto-detected target platform |
|  | [`build`<br>`skip`](https://cibuildwheel.pypa.io/en/stable/options/#build-skip) | Choose the Python versions to build |
|  | [`archs`](https://cibuildwheel.pypa.io/en/stable/options/#archs) | Change the architectures built on your machine by default. |
|  | [`project-requires-python`](https://cibuildwheel.pypa.io/en/stable/options/#requires-python) | Manually set the Python compatibility of your project |
|  | [`enable`](https://cibuildwheel.pypa.io/en/stable/options/#enable) | Enable building with extra categories of selectors present. |
|  | [`allow-empty`](https://cibuildwheel.pypa.io/en/stable/options/#allow-empty) | Suppress the error code if no wheels match the specified build identifiers |
| **Build customization** | [`build-frontend`](https://cibuildwheel.pypa.io/en/stable/options/#build-frontend) | Set the tool to use to build, either "build" (default), "build\[uv\]", or "pip" |
|  | [`config-settings`](https://cibuildwheel.pypa.io/en/stable/options/#config-settings) | Specify config-settings for the build backend. |
|  | [`environment`](https://cibuildwheel.pypa.io/en/stable/options/#environment) | Set environment variables |
|  | [`environment-pass`](https://cibuildwheel.pypa.io/en/stable/options/#environment-pass) | Set environment variables on the host to pass-through to the container. |
|  | [`before-all`](https://cibuildwheel.pypa.io/en/stable/options/#before-all) | Execute a shell command on the build system before any wheels are built. |
|  | [`before-build`](https://cibuildwheel.pypa.io/en/stable/options/#before-build) | Execute a shell command preparing each wheel's build |
|  | [`xbuild-tools`](https://cibuildwheel.pypa.io/en/stable/options/#xbuild-tools) | Binaries on the path that should be included in an isolated cross-build environment. |
|  | [`repair-wheel-command`](https://cibuildwheel.pypa.io/en/stable/options/#repair-wheel-command) | Execute a shell command to repair each built wheel |
|  | [`manylinux-*-image`<br>`musllinux-*-image`](https://cibuildwheel.pypa.io/en/stable/options/#linux-image) | Specify manylinux / musllinux container images |
|  | [`container-engine`](https://cibuildwheel.pypa.io/en/stable/options/#container-engine) | Specify the container engine to use when building Linux wheels |
|  | [`dependency-versions`](https://cibuildwheel.pypa.io/en/stable/options/#dependency-versions) | Control the versions of the tools cibuildwheel uses |
|  | [`pyodide-version`](https://cibuildwheel.pypa.io/en/stable/options/#pyodide-version) | Specify the Pyodide version to use for `pyodide` platform builds |
| **Testing** | [`test-command`](https://cibuildwheel.pypa.io/en/stable/options/#test-command) | The command to test each built wheel |
|  | [`before-test`](https://cibuildwheel.pypa.io/en/stable/options/#before-test) | Execute a shell command before testing each wheel |
|  | [`test-sources`](https://cibuildwheel.pypa.io/en/stable/options/#test-sources) | Paths that are copied into the working directory of the tests |
|  | [`test-requires`](https://cibuildwheel.pypa.io/en/stable/options/#test-requires) | Install Python dependencies before running the tests |
|  | [`test-extras`](https://cibuildwheel.pypa.io/en/stable/options/#test-extras) | Install your wheel for testing using `extras_require` |
|  | [`test-groups`](https://cibuildwheel.pypa.io/en/stable/options/#test-groups) | Specify test dependencies from your project's `dependency-groups` |
|  | [`test-skip`](https://cibuildwheel.pypa.io/en/stable/options/#test-skip) | Skip running tests on some builds |
|  | [`test-environment`](https://cibuildwheel.pypa.io/en/stable/options/#test-environment) | Set environment variables for the test environment |
| **Debugging** | [`debug-keep-container`](https://cibuildwheel.pypa.io/en/stable/options/#debug-keep-container) | Keep the container after running for debugging. |
|  | [`debug-traceback`](https://cibuildwheel.pypa.io/en/stable/options/#debug-traceback) | Print full traceback when errors occur. |
|  | [`build-verbosity`](https://cibuildwheel.pypa.io/en/stable/options/#build-verbosity) | Increase/decrease the output of the build |


<!--[[[end]]] (sum: FxE3nIgFiY) -->

These options can be specified in a `pyproject.toml` file or as environment variables.  Refer to the [configuration documentation](https://cibuildwheel.pypa.io/en/latest/configuration/) for detailed instructions.

## Working Examples

Explore real-world implementations of cibuildwheel:

<!-- START bin/projects.py -->

<!-- this section is generated by bin/projects.py. Don't edit it directly, instead, edit docs/data/projects.yml -->

| Name                              | CI | OS | Notes |
|-----------------------------------|----|----|:------|
| [scikit-learn][]                  | ![github icon][] | ![windows icon][] ![apple icon][] ![linux icon][] | The machine learning library. A complex but clean config using many of cibuildwheel's features to build a large project with Cython and C++ extensions.  |
| [pytorch-fairseq][]               | ![github icon][] | ![apple icon][] ![linux icon][] | Facebook AI Research Sequence-to-Sequence Toolkit written in Python. |
| [duckdb][]                        | ![github icon][] | ![apple icon][] ![linux icon][] ![windows icon][] | DuckDB is an analytical in-process SQL database management system |
| [NumPy][]                         | ![github icon][] ![travisci icon][] | ![windows icon][] ![apple icon][] ![linux icon][] | The fundamental package for scientific computing with Python. |
| [Tornado][]                       | ![github icon][] | ![linux icon][] ![apple icon][] ![windows icon][] | Tornado is a Python web framework and asynchronous networking library. Uses stable ABI for a small C extension. |
| [NCNN][]                          | ![github icon][] | ![windows icon][] ![apple icon][] ![linux icon][] | ncnn is a high-performance neural network inference framework optimized for the mobile platform |
| [Matplotlib][]                    | ![github icon][] | ![windows icon][] ![apple icon][] ![linux icon][] | The venerable Matplotlib, a Python library with C++ portions |
| [MyPy][]                          | ![github icon][] | ![apple icon][] ![linux icon][] ![windows icon][] | The compiled version of MyPy using MyPyC. |
| [Prophet][]                       | ![github icon][] | ![windows icon][] ![apple icon][] ![linux icon][] | Tool for producing high quality forecasts for time series data that has multiple seasonality with linear or non-linear growth. |
| [Kivy][]                          | ![github icon][] | ![windows icon][] ![apple icon][] ![linux icon][] | Open source UI framework written in Python, running on Windows, Linux, macOS, Android and iOS |

[scikit-learn]: https://github.com/scikit-learn/scikit-learn
[pytorch-fairseq]: https://github.com/facebookresearch/fairseq
[duckdb]: https://github.com/duckdb/duckdb
[NumPy]: https://github.com/numpy/numpy
[Tornado]: https://github.com/tornadoweb/tornado
[NCNN]: https://github.com/Tencent/ncnn
[Matplotlib]: https://github.com/matplotlib/matplotlib
[MyPy]: https://github.com/mypyc/mypy_mypyc-wheels
[Prophet]: https://github.com/facebook/prophet
[Kivy]: https://github.com/kivy/kivy

[github icon]: docs/data/readme_icons/github.svg
[azurepipelines icon]: docs/data/readme_icons/azurepipelines.svg
[circleci icon]: docs/data/readme_icons/circleci.svg
[gitlab icon]: docs/data/readme_icons/gitlab.svg
[travisci icon]: docs/data/readme_icons/travisci.svg
[cirrusci icon]: docs/data/readme_icons/cirrusci.svg
[windows icon]: docs/data/readme_icons/windows.svg
[apple icon]: docs/data/readme_icons/apple.svg
[linux icon]: docs/data/readme_icons/linux.svg

<!-- END bin/projects.py -->

>   Discover more examples on the [Working Examples](https://cibuildwheel.pypa.io/en/stable/working-examples) page in the documentation.

## Legal Note

`cibuildwheel` uses `delocate` or `auditwheel` to repair wheels, potentially bundling dynamically linked libraries from the build machine.

This approach is similar to static linking, and it may have license implications.  Review the licenses of any included code to ensure compliance.

## Changelog

### v3.1.2

_29 July 2025_

- ⚠️  Add an error if `CIBW_FREE_THREADING_SUPPORT` is set; you are likely missing 3.13t wheels, please use the `enable`/`CIBW_ENABLE` (#2520)
- 🛠 `riscv64` now enabled if you target that architecture, it's now supported on PyPI (#2509)
- 🛠 Add warning when using `cpython-experimental-riscv64` (no longer needed) (#2526, #2528)
- 🛠 iOS versions bumped, fixing issues with 3.14 (now RC 1) (#2530)
- 🐛 Fix bug in Android running wheel from our GitHub Action (#2517)
- 🐛 Fix warning when using `test-skip` of `"*-macosx_universal2:arm64"` (#2522)
- 🐛 Fix incorrect number of wheels reported in logs, again (#2517)
- 📚 We welcome our Android platform maintainer (#2516)


### v3.1.1

_24 July 2025_

- 🐛 Fix a bug showing an incorrect wheel count at the end of execution, and misrepresenting test-only runs in the GitHub Action summary (#2512)
- 📚 Docs fix (#2510)

### v3.1.0

_23 July 2025_


- 🌟 CPython 3.14 wheels are now built by default - without the `"cpython-prerelease"` `enable` set. It's time to build and upload these wheels to PyPI! This release includes CPython 3.14.0rc1, which is guaranteed to be ABI compatible with the final release. (#2507) Free-threading is no longer experimental in 3.14, so you have to skip it explicitly with `'cp31?t-*'` if you don't support it yet. (#2503)
- 🌟 Adds the ability to [build wheels for Android](https://cibuildwheel.pypa.io/en/stable/platforms/#android)! Set the [`platform` option](https://cibuildwheel.pypa.io/en/stable/options/#platform) to `android` on Linux or macOS to try it out! (#2349)
- 🌟 Adds Pyodide 0.28, which builds 3.13 wheels (#2487)
- ✨ Support for 32-bit `manylinux_2_28` (now a consistent default) and `manylinux_2_34` added (#2500)
- 🛠 Improved summary, will also use markdown summary output on GHA (#2469)
- 🛠 The riscv64 images now have a working default (as they are now part of pypy/manylinux), but are still experimental (and behind an `enable`) since you can't push them to PyPI yet (#2506)
- 🛠 Fixed a typo in the 3.9 MUSL riscv64 identifier (`cp39-musllinux_ricv64` -> `cp39-musllinux_riscv64`) (#2490)
- 🛠 Mistyping `--only` now shows the correct possibilities, and even suggests near matches on Python 3.14+ (#2499)
- 🛠 Only support one output from the repair step on linux like other platforms; auditwheel fixed this over four years ago! (#2478)
- 🛠 We now use pattern matching extensively (#2434)
- 📚 We now have platform maintainers for our special platforms and interpreters! (#2481)



### v3.0.1

_5 July 2025_

- 🛠 Updates CPython 3.14 prerelease to 3.14.0b3 (#2471)
- ✨ Adds a CPython 3.14 prerelease iOS build (only when prerelease builds are [enabled](https://cibuildwheel.pypa.io/en/stable/options/#enable)) (#2475)

### v3.0.0

_11 June 2025_

See @henryiii's [release post](https://iscinumpy.dev/post/cibuildwheel-3-0-0/) for more info on new features!

- 🌟 Adds the ability to [build wheels for iOS](https://cibuildwheel.pypa.io/en/stable/platforms/#ios)! Set the [`platform` option](https://cibuildwheel.pypa.io/en/stable/options/#platform) to `ios` on a Mac with the iOS toolchain to try it out! (#2286, #2363, #2432)
- 🌟 Adds support for the GraalPy interpreter! Enable for your project using the [`enable` option](https://cibuildwheel.pypa.io/en/stable/options/#enable). (#1538, #2411, #2414)
- ✨ Adds CPython 3.14 support, under the [`enable` option](https://cibuildwheel.pypa.io/en/stable/options/#enable) `cpython-prerelease`. This version of cibuildwheel uses 3.14.0b2. (#2390)

    _While CPython is in beta, the ABI can change, so your wheels might not be compatible with the final release. For this reason, we don't recommend distributing wheels until RC1, at which point 3.14 will be available in cibuildwheel without the flag._ (#2390)

- ✨ Adds the [test-sources option](https://cibuildwheel.pypa.io/en/stable/options/#test-sources), which copies files and folders into the temporary working directory we run tests from. (#2062, #2284, #2420, #2437)

    This is particularly important for iOS builds, which do not support placeholders in the `test-command`, but can also be useful for other platforms.

- ✨ Adds [`dependency-versions`](https://cibuildwheel.pypa.io/en/stable/options/#dependency-versions) inline syntax (#2122)
- ✨ Improves support for Pyodide builds and adds the experimental [`pyodide-version`](https://cibuildwheel.pypa.io/en/stable/options/#pyodide-version) option, which allows you to specify the version of Pyodide to use for builds. (#2002)
- ✨ Add `pyodide-prerelease` [enable](https://cibuildwheel.pypa.io/en/stable/options/#enable) option, with an early build of 0.28 (Python 3.13). (#2431)
- ✨ Adds the [`test-environment`](https://cibuildwheel.pypa.io/en/stable/options/#test-environment) option, which allows you to set environment variables for the test command. (#2388)
- ✨ Adds the [`xbuild-tools`](https://cibuildwheel.pypa.io/en/stable/options/#xbuild-tools) option, which allows you to specify tools safe for cross-compilation. Currently only used on iOS; will be useful for Android in the future. (#2317)
- 🛠 The default [manylinux image](https://cibuildwheel.pypa.io/en/stable/options/#linux-image) has changed from `manylinux2014` to `manylinux_2_28`. (#2330)
- 🛠 EOL images `manylinux1`, `manylinux2010`, `manylinux_2_24` and `musllinux_1_1` can no longer be specified by their shortname. The full OCI name can still be used for these images, if you wish. (#2316)
- 🛠 Invokes `build` rather than `pip wheel` to build wheels by default. You can control this via the [`build-frontend`](https://cibuildwheel.pypa.io/en/stable/options/#build-frontend) option. You might notice that you can see your build log output now! (#2321)
- 🛠 Build verbosity settings have been reworked to have consistent meanings between build backends when non-zero. (#2339)
- 🛠 Removed the `CIBW_PRERELEASE_PYTHONS` and `CIBW_FREE_THREADED_SUPPORT` options - these have been folded into the [`enable`](https://cibuildwheel.pypa.io/en/stable/options/#enable) option instead. (#2095)
- 🛠 Build environments no longer have setuptools and wheel preinstalled. (#2329)
- 🛠 Use the standard Schema line for the integrated JSONSchema. (#2433)
- ⚠️ Dropped support for building Python 3.6 and 3.7 wheels. If you need to build wheels for these versions, use cibuildwheel v2.23.3 or earlier. (#2282)
- ⚠️ The minimum Python version required to run cibuildwheel is now Python 3.11. You can still build wheels for Python 3.8 and newer. (#1912)
- ⚠️ 32-bit Linux wheels no longer built by default - the [arch](https://cibuildwheel.pypa.io/en/stable/options/#archs) was removed from `"auto"`. It now requires explicit `"auto32"`. Note that modern manylinux images (like the new default, `manylinux_2_28`) do not have 32-bit versions. (#2458)
- ⚠️ PyPy wheels no longer built by default, due to a change to our options system. To continue building PyPy wheels, you'll now need to set the [`enable` option](https://cibuildwheel.pypa.io/en/stable/options/#enable) to `pypy` or `pypy-eol`. (#2095)
- ⚠️ Dropped official support for Appveyor. If it was working for you before, it will probably continue to do so, but we can't be sure, because our CI doesn't run there anymore. (#2386)
- 📚 A reorganisation of the docs, and numerous updates. (#2280)
- 📚 Use Python 3.14 color output in docs CLI output. (#2407)
- 📚 Docs now primarily use the pyproject.toml name of options, rather than the environment variable name. (#2389)
- 📚 README table now matches docs and auto-updates. (#2427, #2428)

<!-- [[[end]]] (sum: sVC5DNuhaF) -->

---

Check out the [changelog page in the docs](https://cibuildwheel.pypa.io/en/stable/changelog/) for the latest updates.

## Contributing

Learn how to contribute to cibuildwheel by visiting the [docs](https://cibuildwheel.pypa.io/en/latest/contributing/).  We welcome contributions!

This project adheres to the [PSF Code of Conduct](https://github.com/pypa/.github/blob/main/CODE_OF_CONDUCT.md).

## Maintainers

*   Joe Rickerby [@joerick](https://github.com/joerick)
*   Yannick Jadoul [@YannickJadoul](https://github.com/YannickJadoul)
*   Matthieu Darbois [@mayeut](https://github.com/mayeut)
*   Henry Schreiner [@henryiii](https://github.com/henryiii)
*   Grzegorz Bokota [@Czaki](https://github.com/Czaki)

## Platform Maintainers

*   Russell Keith-Magee [@freakboy3742](https://github.com/freakboy3742) (iOS)
*   Agriya Khetarpal [@agriyakhetarpal](https://github.com/agriyakhetarpal) (Pyodide)
*   Hood Chatham [@hoodmane](https://github.com/hoodmane) (Pyodide)
*   Gyeongjae Choi [@ryanking13](https://github.com/ryanking13) (Pyodide)
*   Tim Felgentreff [@timfel](https://github.com/timfel) (GraalPy)
*   Malcolm Smith [@mhsmith](https://github.com/mhsmith) (Android)

## Credits

`cibuildwheel` leverages the work of many:

*   ⭐️ @matthew-brett for [multibuild](https://github.com/multi-build/multibuild) and [matthew-brett/delocate](http://github.com/matthew-brett/delocate)
*   @PyPA for the manylinux Docker images [pypa/manylinux](https://github.com/pypa/manylinux)
*   @ogrisel for [wheelhouse-uploader](https://github.com/ogrisel/wheelhouse-uploader) and `run_with_env.cmd`

Special thanks to:

*   @zfrenchee for [help debugging many issues](https://github.com/pypa/cibuildwheel/issues/2)
*   @lelit for some great bug reports and [contributions](https://github.com/pypa/cibuildwheel/pull/73)
*   @mayeut for a [phenomenal PR](https://github.com/pypa/cibuildwheel/pull/71) patching Python itself for better compatibility!
*   @czaki for being a super-contributor over many PRs and helping out with countless issues!
*   @mattip for his help with adding PyPy support to cibuildwheel

## See Also

Consider [matthew-brett/multibuild](http://github.com/matthew-brett/multibuild) for building wheels on various platforms or [maturin-action](https://github.com/PyO3/maturin-action) if building Rust wheels.
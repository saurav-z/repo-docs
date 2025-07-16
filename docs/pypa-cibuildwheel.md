# cibuildwheel: Effortlessly Build Python Wheels for All Platforms

**Simplify your Python package distribution with cibuildwheel, a powerful tool that automates building and testing your wheels across macOS, Linux, Windows, and multiple Python versions.** [See the original repo](https://github.com/pypa/cibuildwheel)

[![PyPI](https://img.shields.io/pypi/v/cibuildwheel.svg)](https://pypi.python.org/pypi/cibuildwheel)
[![Documentation Status](https://readthedocs.org/projects/cibuildwheel/badge/?version=stable)](https://cibuildwheel.pypa.io/en/stable/?badge=stable)
[![Actions Status](https://github.com/pypa/cibuildwheel/workflows/Test/badge.svg)](https://github.com/pypa/cibuildwheel/actions)
[![Travis Status](https://img.shields.io/travis/com/pypa/cibuildwheel/main?logo=travis)](https://travis-ci.com/github/pypa/cibuildwheel)
[![CircleCI Status](https://img.shields.io/circleci/build/gh/pypa/cibuildwheel/main?logo=circleci)](https://circleci.com/gh/pypa/cibuildwheel)
[![Azure Status](https://dev.azure.com/joerick0429/cibuildwheel/_apis/build/status/pypa.cibuildwheel?branchName=main)](https://dev.azure.com/joerick0429/cibuildwheel/_build/latest?definitionId=4&branchName=main)

[Documentation](https://cibuildwheel.pypa.io)

## Key Features

*   **Cross-Platform Builds:** Build wheels for macOS (Intel & Apple Silicon), Linux (manylinux & musllinux), Windows (32/64bit, ARM64), iOS and Pyodide, covering diverse architectures.
*   **Multiple Python Versions:** Supports CPython, PyPy, and GraalPy across a wide range of Python versions.
*   **CI Integration:** Seamlessly integrates with GitHub Actions, Azure Pipelines, Travis CI, CircleCI, and GitLab CI.
*   **Dependency Management:**  Bundles shared library dependencies on Linux and macOS through auditwheel and delocate.
*   **Automated Testing:** Runs your library's tests against the wheel-installed version for comprehensive validation.
*   **Customizable Builds:** Provides flexible options for build customization, including environment variables, test commands, and more.

## Supported Platforms and Python Versions

| Platform              | macOS Intel | macOS Apple Silicon | Windows 64bit | Windows 32bit | Windows Arm64 | manylinux/musllinux x86_64 | manylinux/musllinux i686 | manylinux/musllinux aarch64 | manylinux/musllinux ppc64le | manylinux/musllinux s390x | manylinux/musllinux armv7l | iOS | Pyodide |
|-----------------------|-------------|---------------------|---------------|---------------|---------------|-----------------------------|---------------------------|---------------------------|-----------------------------|-----------------------------|--------------------------|-----|---------|
| CPython 3.8           | ✅          | ✅                  | ✅            | ✅            | N/A           | ✅                           | ✅                         | ✅                         | ✅                          | ✅                          | ✅⁵                      | N/A | N/A     |
| CPython 3.9           | ✅          | ✅                  | ✅            | ✅            | ✅²           | ✅                           | ✅                         | ✅                         | ✅                          | ✅                          | ✅⁵                      | N/A | N/A     |
| CPython 3.10          | ✅          | ✅                  | ✅            | ✅            | ✅²           | ✅                           | ✅                         | ✅                         | ✅                          | ✅                          | ✅⁵                      | N/A | N/A     |
| CPython 3.11          | ✅          | ✅                  | ✅            | ✅            | ✅²           | ✅                           | ✅                         | ✅                         | ✅                          | ✅                          | ✅⁵                      | N/A | N/A     |
| CPython 3.12          | ✅          | ✅                  | ✅            | ✅            | ✅²           | ✅                           | ✅                         | ✅                         | ✅                          | ✅                          | ✅⁵                       | N/A | ✅⁴     |
| CPython 3.13          | ✅          | ✅                  | ✅            | ✅            | ✅²           | ✅                           | ✅                         | ✅                         | ✅                          | ✅                          | ✅⁵                      | ✅  | N/A     |
| CPython 3.14          | ✅          | ✅                  | ✅            | ✅            | ✅²           | ✅                           | ✅                         | ✅                         | ✅                          | ✅                          | ✅⁵                      | ✅  | N/A     |
| PyPy 3.8 v7.3         | ✅          | ✅                  | ✅            | N/A           | N/A           | ✅¹                          | ✅¹                        | ✅¹                        | N/A                         | N/A                         | N/A                      | N/A | N/A     |
| PyPy 3.9 v7.3         | ✅          | ✅                  | ✅            | N/A           | N/A           | ✅¹                          | ✅¹                        | ✅¹                        | N/A                         | N/A                         | N/A                      | N/A | N/A     |
| PyPy 3.10 v7.3        | ✅          | ✅                  | ✅            | N/A           | N/A           | ✅¹                          | ✅¹                        | ✅¹                        | N/A                         | N/A                         | N/A                      | N/A | N/A     |
| PyPy 3.11 v7.3        | ✅          | ✅                  | ✅            | N/A           | N/A           | ✅¹                          | ✅¹                        | ✅¹                        | N/A                         | N/A                         | N/A                      | N/A | N/A     |
| GraalPy 3.11 v24.2    | ✅          | ✅                  | ✅            | N/A           | N/A           | ✅¹                          | N/A                        | ✅¹                        | N/A                         | N/A                         | N/A                      | N/A | N/A     |

<sup>¹ PyPy & GraalPy are only supported for manylinux wheels.</sup><br>
<sup>² Windows arm64 support is experimental.</sup><br>
<sup>³ Free-threaded mode requires opt-in using [`enable`](https://cibuildwheel.pypa.io/en/stable/options/#enable).</sup><br>
<sup>⁴ Experimental, not yet supported on PyPI, but can be used directly in web deployment. Use `--platform pyodide` to build.</sup><br>
<sup>⁵ manylinux armv7l support is experimental. As there are no RHEL based image for this architecture, it's using an Ubuntu based image instead.</sup><br>

## How it Works

```mermaid
graph LR
    A[CI Server] --> B{cibuildwheel}
    B --> C{Identify Python Versions and Platforms}
    C --> D{Create Build Environments (Containers/VMs)}
    D --> E{Install Build Dependencies}
    E --> F{Build Wheel}
    F --> G{Test Wheel}
    G --> H{Package and Upload Wheels}
    H --> I[PyPI or Artifact Storage]
```

Explore an interactive version of this diagram [in the docs](https://cibuildwheel.pypa.io/en/stable/#how-it-works).

## Usage

`cibuildwheel` is designed to run within your CI service. Here's a table illustrating platform support:

|                 | Linux | macOS | Windows | Linux ARM | macOS ARM | Windows ARM | iOS |
|-----------------|-------|-------|---------|-----------|-----------|-------------|-----|
| GitHub Actions  | ✅    | ✅    | ✅       | ✅        | ✅        | ✅          | ✅³  |
| Azure Pipelines | ✅    | ✅    | ✅       |           | ✅        | ✅²         | ✅³  |
| Travis CI       | ✅    |       | ✅      | ✅        |           |             |     |
| CircleCI        | ✅    | ✅    |         | ✅        | ✅        |             | ✅³  |
| Gitlab CI       | ✅    | ✅    | ✅      | ✅¹       | ✅        |             | ✅³  |
| Cirrus CI       | ✅    | ✅    | ✅      | ✅        | ✅        |             |      |

<sup>¹ [Requires emulation](https://cibuildwheel.pypa.io/en/stable/faq/#emulation), distributed separately. Other services may also support Linux ARM through emulation or third-party build hosts, but these are not tested in our CI.</sup><br>
<sup>² [Uses cross-compilation](https://cibuildwheel.pypa.io/en/stable/faq/#windows-arm64). It is not possible to test `arm64` on this CI platform.</sup><br>
<sup>³ Requires a macOS runner; runs tests on the simulator for the runner's architecture.</sup>

## Example Setup (GitHub Actions)

To build manylinux, musllinux, macOS, and Windows wheels on GitHub Actions, use the following `.github/workflows/wheels.yml`:

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
        run: python -m pip install cibuildwheel==3.0.1

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

## Configuration Options

The following table outlines the available configuration options:

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
|  | [`test-sources`](https://cibuildwheel.pypa.io/en/stable/options/#test-sources) | Files and folders from the source tree that are copied into an isolated tree before running the tests |
|  | [`test-requires`](https://cibuildwheel.pypa.io/en/stable/options/#test-requires) | Install Python dependencies before running the tests |
|  | [`test-extras`](https://cibuildwheel.pypa.io/en/stable/options/#test-extras) | Install your wheel for testing using `extras_require` |
|  | [`test-groups`](https://cibuildwheel.pypa.io/en/stable/options/#test-groups) | Specify test dependencies from your project's `dependency-groups` |
|  | [`test-skip`](https://cibuildwheel.pypa.io/en/stable/options/#test-skip) | Skip running tests on some builds |
|  | [`test-environment`](https://cibuildwheel.pypa.io/en/stable/options/#test-environment) | Set environment variables for the test environment |
| **Debugging** | [`debug-keep-container`](https://cibuildwheel.pypa.io/en/stable/options/#debug-keep-container) | Keep the container after running for debugging. |
|  | [`debug-traceback`](https://cibuildwheel.pypa.io/en/stable/options/#debug-traceback) | Print full traceback when errors occur. |
|  | [`build-verbosity`](https://cibuildwheel.pypa.io/en/stable/options/#build-verbosity) | Increase/decrease the output of the build |

These options can be specified in a `pyproject.toml` file, or as environment variables. [See the configuration docs](https://cibuildwheel.pypa.io/en/latest/configuration/) for detailed information.

## Working Examples

*   [scikit-learn][]
*   [pytorch-fairseq][]
*   [duckdb][]
*   [NumPy][]
*   [Tornado][]
*   [NCNN][]
*   [Matplotlib][]
*   [MyPy][]
*   [Prophet][]
*   [Kivy][]

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

## Legal Note

`cibuildwheel` employs `delocate` or `auditwheel` to repair wheels, potentially bundling dynamically linked libraries. This behavior, akin to static linking, might have licensing implications. Always review the licenses of included code.

## Changelog

### v3.0.1 (5 July 2025)

*   Updates CPython 3.14 prerelease to 3.14.0b3 (#2471)
*   Adds a CPython 3.14 prerelease iOS build (only when prerelease builds are [enabled](https://cibuildwheel.pypa.io/en/stable/options/#enable)) (#2475)

### v3.0.0 (11 June 2025)

*   Adds the ability to [build wheels for iOS](https://cibuildwheel.pypa.io/en/stable/platforms/#ios)!
*   Adds support for the GraalPy interpreter!
*   Adds CPython 3.14 support.
*   Adds the [test-sources option](https://cibuildwheel.pypa.io/en/stable/options/#test-sources).
*   Adds [`dependency-versions`](https://cibuildwheel.pypa.io/en/stable/options/#dependency-versions) inline syntax.
*   Improves support for Pyodide builds and adds the experimental [`pyodide-version`](https://cibuildwheel.pypa.io/en/stable/options/#pyodide-version) option.
*   Add `pyodide-prerelease` [enable](https://cibuildwheel.pypa.io/en/stable/options/#enable) option, with an early build of 0.28 (Python 3.13).
*   Adds the [`test-environment`](https://cibuildwheel.pypa.io/en/stable/options/#test-environment) option.
*   Adds the [`xbuild-tools`](https://cibuildwheel.pypa.io/en/stable/options/#xbuild-tools) option.
*   Changes the default [manylinux image](https://cibuildwheel.pypa.io/en/stable/options/#linux-image) to `manylinux_2_28`.
*   Invokes `build` rather than `pip wheel` to build wheels by default.
*   Removed the `CIBW_PRERELEASE_PYTHONS` and `CIBW_FREE_THREADED_SUPPORT` options.
*   Build environments no longer have setuptools and wheel preinstalled.
*   Use the standard Schema line for the integrated JSONSchema.
*   Dropped support for building Python 3.6 and 3.7 wheels.
*   The minimum Python version required to run cibuildwheel is now Python 3.11.
*   32-bit Linux wheels no longer built by default.
*   PyPy wheels no longer built by default.
*   Dropped official support for Appveyor.
*   A reorganisation of the docs, and numerous updates.

### v2.23.3 (26 April 2025)

*   Dependency updates, including Python 3.13.3 (#2371)

### v2.23.2 (24 March 2025)

*   Workaround an issue with pyodide builds when running cibuildwheel with a Python that was installed via UV (#2328 via #2331)
*   Dependency updates, including a manylinux update that fixes an ['undefined symbol' error](https://github.com/pypa/manylinux/issues/1760) in gcc-toolset (#2334)

### v2.23.1 (15 March 2025)

*   Added warnings when the shorthand values `manylinux1`, `manylinux2010`, `manylinux_2_24`, and `musllinux_1_1` are used to specify the images in linux builds.
*   Dependency updates, including a manylinux update which fixes an [issue with rustup](https://github.com/pypa/cibuildwheel/issues/2303). (#2315)

## Contributing

See the [docs](https://cibuildwheel.pypa.io/en/latest/contributing/) for contributing guidelines.

## Maintainers

*   Joe Rickerby [@joerick](https://github.com/joerick)
*   Yannick Jadoul [@YannickJadoul](https://github.com/YannickJadoul)
*   Matthieu Darbois [@mayeut](https://github.com/mayeut)
*   Henry Schreiner [@henryiii](https://github.com/henryiii)
*   Grzegorz Bokota [@Czaki](https://github.com/Czaki)

## Credits

`cibuildwheel` is built on the contributions of many:

*   @matthew-brett for [multibuild](https://github.com/multi-build/multibuild) and [matthew-brett/delocate](http://github.com/matthew-brett/delocate)
*   @PyPA for the manylinux Docker images [pypa/manylinux](https://github.com/pypa/manylinux)

## See Also

*   [matthew-brett/multibuild](http://github.com/matthew-brett/multibuild)
*   [PyO3/maturin-action](https://github.com/PyO3/maturin-action)
# cibuildwheel: Automate Building Python Wheels Across Platforms

Tired of manually building Python wheels for Mac, Linux, and Windows across multiple Python versions? **cibuildwheel streamlines the process, saving you time and effort.** Check out the original repository [here](https://github.com/pypa/cibuildwheel).

[![PyPI](https://img.shields.io/pypi/v/cibuildwheel.svg)](https://pypi.python.org/pypi/cibuildwheel)
[![Documentation Status](https://readthedocs.org/projects/cibuildwheel/badge/?version=stable)](https://cibuildwheel.pypa.io/en/stable/?badge=stable)
[![Actions Status](https://github.com/pypa/cibuildwheel/workflows/Test/badge.svg)](https://github.com/pypa/cibuildwheel/actions)
[![Travis Status](https://img.shields.io/travis/com/pypa/cibuildwheel/main?logo=travis)](https://travis-ci.com/github/pypa/cibuildwheel)
[![CircleCI Status](https://img.shields.io/circleci/build/gh/pypa/cibuildwheel/main?logo=circleci)](https://circleci.com/gh/pypa/cibuildwheel)
[![Azure Status](https://dev.azure.com/joerick0429/cibuildwheel/_apis/build/status/pypa.cibuildwheel?branchName=main)](https://dev.azure.com/joerick0429/cibuildwheel/_build/latest?definitionId=4&branchName=main)

[Documentation](https://cibuildwheel.pypa.io)

## Key Features

*   **Cross-Platform Support:** Builds wheels for macOS, Linux (including manylinux and musllinux), Windows, Android, and iOS.
*   **Multiple Python Versions:** Supports CPython, PyPy, and GraalPy across a wide range of Python versions (3.8 - 3.14).
*   **CI Integration:** Seamlessly integrates with popular CI/CD platforms like GitHub Actions, Azure Pipelines, Travis CI, CircleCI, and GitLab CI.
*   **Dependency Handling:** Bundles shared library dependencies on Linux and macOS using auditwheel and delocate.
*   **Testing:** Runs your library's tests against the wheel-installed version of your library.
*   **Customization:** Offers a wide array of options for build configuration, including environment variables, build frontend selection, test commands, and more (see below).

## Supported Platforms and Python Versions

| Platform                  | macOS Intel | macOS Apple Silicon | Windows 64bit | Windows 32bit | Windows Arm64 | manylinux/musllinux x86_64 | manylinux/musllinux i686 | manylinux/musllinux aarch64 | manylinux/musllinux ppc64le | manylinux/musllinux s390x | manylinux/musllinux armv7l | Android | iOS | Pyodide |
| :------------------------ | :---------- | :------------------ | :------------ | :------------ | :------------ | :--------------------------- | :------------------------ | :------------------------- | :--------------------------- | :------------------------- | :------------------------- | :------ | :-- | :------ |
| CPython 3.8               | ✅          | ✅                  | ✅            | ✅            | N/A           | ✅                           | ✅                        | ✅                         | ✅                           | ✅                         | ✅⁵          | N/A     | N/A | N/A     |
| CPython 3.9               | ✅          | ✅                  | ✅            | ✅            | ✅²           | ✅                           | ✅                        | ✅                         | ✅                           | ✅                         | ✅⁵          | N/A     | N/A | N/A     |
| CPython 3.10              | ✅          | ✅                  | ✅            | ✅            | ✅²           | ✅                           | ✅                        | ✅                         | ✅                           | ✅                         | ✅⁵          | N/A     | N/A | N/A     |
| CPython 3.11              | ✅          | ✅                  | ✅            | ✅            | ✅²           | ✅                           | ✅                        | ✅                         | ✅                           | ✅                         | ✅⁵          | N/A     | N/A | N/A     |
| CPython 3.12              | ✅          | ✅                  | ✅            | ✅            | ✅²           | ✅                           | ✅                        | ✅                         | ✅                           | ✅                         | ✅⁵          | N/A     | N/A | ✅⁴     |
| CPython 3.13³             | ✅          | ✅                  | ✅            | ✅            | ✅²           | ✅                           | ✅                        | ✅                         | ✅                           | ✅                         | ✅⁵          | ✅      | ✅  | N/A     |
| CPython 3.14              | ✅          | ✅                  | ✅            | ✅            | ✅²           | ✅                           | ✅                        | ✅                         | ✅                           | ✅                         | ✅⁵          | N/A     | N/A | N/A     |
| PyPy 3.8 v7.3             | ✅          | ✅                  | ✅            | N/A           | N/A           | ✅¹                          | ✅¹                       | ✅¹                        | N/A                          | N/A                        | N/A          | N/A     | N/A | N/A     |
| PyPy 3.9 v7.3             | ✅          | ✅                  | ✅            | N/A           | N/A           | ✅¹                          | ✅¹                       | ✅¹                        | N/A                          | N/A                        | N/A          | N/A     | N/A | N/A     |
| PyPy 3.10 v7.3            | ✅          | ✅                  | ✅            | N/A           | N/A           | ✅¹                          | ✅¹                       | ✅¹                        | N/A                          | N/A                        | N/A          | N/A     | N/A | N/A     |
| PyPy 3.11 v7.3            | ✅          | ✅                  | ✅            | N/A           | N/A           | ✅¹                          | ✅¹                       | ✅¹                        | N/A                          | N/A                        | N/A          | N/A     | N/A | N/A     |
| GraalPy 3.11 v24.2        | ✅          | ✅                  | ✅            | N/A           | N/A           | ✅¹                          | N/A                       | ✅¹                        | N/A                          | N/A                        | N/A          | N/A     | N/A | N/A     |

<sup>¹ PyPy & GraalPy are only supported for manylinux wheels.</sup><br>
<sup>² Windows arm64 support is experimental.</sup><br>
<sup>³ Free-threaded mode requires opt-in on 3.13 using [`enable`](https://cibuildwheel.pypa.io/en/stable/options/#enable).</sup><br>
<sup>⁴ Experimental, not yet supported on PyPI, but can be used directly in web deployment. Use `--platform pyodide` to build.</sup><br>
<sup>⁵ manylinux armv7l support is experimental. As there are no RHEL based image for this architecture, it's using an Ubuntu based image instead.</sup><br>

## Usage

`cibuildwheel` is designed to be run within your CI/CD system.

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

### Example: GitHub Actions

Here's an example of how to configure `cibuildwheel` in your `.github/workflows/wheels.yml` for building manylinux, musllinux, macOS, and Windows wheels on GitHub Actions:

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
        run: python -m pip install cibuildwheel==3.1.1

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

For detailed instructions, including PyPI deployment and using other CI services, consult the [cibuildwheel documentation](https://cibuildwheel.pypa.io) and the [examples](https://github.com/pypa/cibuildwheel/tree/main/examples).

## How it Works

```
![](docs/data/how-it-works.png)
```

<sup>Explore an interactive version of this diagram [in the docs](https://cibuildwheel.pypa.io/en/stable/#how-it-works).</sup>

## Configuration Options

| Category             | Option                                                | Description                                                                            |
| :------------------- | :---------------------------------------------------- | :------------------------------------------------------------------------------------- |
| **Build Selection**  | `platform`                                            | Override the auto-detected target platform.                                           |
|                      | `build`/`skip`                                        | Choose the Python versions to build.                                                 |
|                      | `archs`                                               | Change the architectures built on your machine by default.                           |
|                      | `project-requires-python`                             | Manually set the Python compatibility of your project.                              |
|                      | `enable`                                              | Enable building with extra categories of selectors present.                         |
|                      | `allow-empty`                                         | Suppress the error code if no wheels match the specified build identifiers.         |
| **Build Customization** | `build-frontend`                                      | Set the tool to use to build, either "build" (default), "build\[uv\]", or "pip".   |
|                      | `config-settings`                                     | Specify config-settings for the build backend.                                        |
|                      | `environment`                                         | Set environment variables.                                                             |
|                      | `environment-pass`                                    | Set environment variables on the host to pass-through to the container.               |
|                      | `before-all`                                          | Execute a shell command on the build system before any wheels are built.            |
|                      | `before-build`                                        | Execute a shell command preparing each wheel's build.                               |
|                      | `xbuild-tools`                                        | Binaries on the path that should be included in an isolated cross-build environment. |
|                      | `repair-wheel-command`                                | Execute a shell command to repair each built wheel.                                  |
|                      | `manylinux-*-image`/`musllinux-*-image`                | Specify manylinux / musllinux container images.                                        |
|                      | `container-engine`                                    | Specify the container engine to use when building Linux wheels.                     |
|                      | `dependency-versions`                                 | Control the versions of the tools cibuildwheel uses.                                  |
|                      | `pyodide-version`                                     | Specify the Pyodide version to use for `pyodide` platform builds.                    |
| **Testing**            | `test-command`                                        | The command to test each built wheel.                                                 |
|                      | `before-test`                                         | Execute a shell command before testing each wheel.                                   |
|                      | `test-sources`                                        | Paths that are copied into the working directory of the tests.                        |
|                      | `test-requires`                                       | Install Python dependencies before running the tests.                                  |
|                      | `test-extras`                                         | Install your wheel for testing using `extras_require`.                                |
|                      | `test-groups`                                         | Specify test dependencies from your project's `dependency-groups`.                    |
|                      | `test-skip`                                           | Skip running tests on some builds.                                                   |
|                      | `test-environment`                                    | Set environment variables for the test environment.                                    |
| **Debugging**          | `debug-keep-container`                                | Keep the container after running for debugging.                                        |
|                      | `debug-traceback`                                     | Print full traceback when errors occur.                                              |
|                      | `build-verbosity`                                     | Increase/decrease the output of the build.                                            |

These options can be specified in a pyproject.toml file or as environment variables. See the [configuration documentation](https://cibuildwheel.pypa.io/en/latest/configuration/) for details.

## Working Examples

Here are some projects that use cibuildwheel:

| Name                | CI         | OS                                           | Notes                                                                                                                                    |
| :------------------ | :--------- | :------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| scikit-learn        | ![github][] | ![windows][] ![apple][] ![linux][]             | The machine learning library. A complex but clean config using many of cibuildwheel's features to build a large project with Cython and C++ extensions. |
| pytorch-fairseq     | ![github][] | ![apple][] ![linux][]                        | Facebook AI Research Sequence-to-Sequence Toolkit written in Python.                                                                   |
| duckdb              | ![github][] | ![apple][] ![linux][] ![windows][]          | DuckDB is an analytical in-process SQL database management system                                                                         |
| NumPy               | ![github][] ![travisci][] | ![windows][] ![apple][] ![linux][]             | The fundamental package for scientific computing with Python.                                                                   |
| Tornado             | ![github][] | ![linux][] ![apple][] ![windows][]            | Tornado is a Python web framework and asynchronous networking library. Uses stable ABI for a small C extension.                        |
| NCNN                | ![github][] | ![windows][] ![apple][] ![linux][]            | ncnn is a high-performance neural network inference framework optimized for the mobile platform                                         |
| Matplotlib          | ![github][] | ![windows][] ![apple][] ![linux][]            | The venerable Matplotlib, a Python library with C++ portions                                                                          |
| MyPy                | ![github][] | ![apple][] ![linux][] ![windows][]            | The compiled version of MyPy using MyPyC.                                                                                             |
| Prophet             | ![github][] | ![windows][] ![apple][] ![linux][]            | Tool for producing high quality forecasts for time series data that has multiple seasonality with linear or non-linear growth.           |
| Kivy                | ![github][] | ![windows][] ![apple][] ![linux][]            | Open source UI framework written in Python, running on Windows, Linux, macOS, Android and iOS                                            |

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
[github]: docs/data/readme_icons/github.svg
[azurepipelines]: docs/data/readme_icons/azurepipelines.svg
[circleci]: docs/data/readme_icons/circleci.svg
[gitlab]: docs/data/readme_icons/gitlab.svg
[travisci]: docs/data/readme_icons/travisci.svg
[cirrusci]: docs/data/readme_icons/cirrusci.svg
[windows]: docs/data/readme_icons/windows.svg
[apple]: docs/data/readme_icons/apple.svg
[linux]: docs/data/readme_icons/linux.svg

> ℹ️  Find more examples at the [Working Examples](https://cibuildwheel.pypa.io/en/stable/working-examples) page in the documentation.

## Legal Note

`cibuildwheel` uses `delocate` or `auditwheel` to repair wheels and may bundle dynamically linked libraries from the build machine, similar to static linking, which could have license implications.  Review licenses for any code pulled in.

## Changelog

### v3.1.1 (24 July 2025)

*   Bug fixes and documentation improvements.

### v3.1.0 (23 July 2025)

*   CPython 3.14 wheels are now built by default.
*   Added the ability to build wheels for Android.
*   Adds Pyodide 0.28, which builds 3.13 wheels
*   Support for 32-bit `manylinux_2_28` and `manylinux_2_34` added
*   Improved summary output.
*   Other improvements and bug fixes.

### v3.0.1 (5 July 2025)

*   Updates CPython 3.14 prerelease.
*   Adds a CPython 3.14 prerelease iOS build.

### v3.0.0 (11 June 2025)

*   Adds the ability to build wheels for iOS.
*   Adds support for the GraalPy interpreter.
*   Adds CPython 3.14 support.
*   Adds the `test-sources` option.
*   Other improvements and bug fixes.

(See the [full changelog](https://cibuildwheel.pypa.io/en/stable/changelog/) for a complete list of changes.)

## Contributing

For details on contributing, see the [contributing guide](https://cibuildwheel.pypa.io/en/latest/contributing/).  All contributors are expected to adhere to the [PSF Code of Conduct](https://github.com/pypa/.github/blob/main/CODE_OF_CONDUCT.md).

## Maintainers

*   Joe Rickerby ([@joerick](https://github.com/joerick))
*   Yannick Jadoul ([@YannickJadoul](https://github.com/YannickJadoul))
*   Matthieu Darbois ([@mayeut](https://github.com/mayeut))
*   Henry Schreiner ([@henryiii](https://github.com/henryiii))
*   Grzegorz Bokota ([@Czaki](https://github.com/Czaki))

## Platform Maintainers

*   Russell Keith-Magee ([@freakboy3742](https://github.com/freakboy3742)) (iOS)
*   Agriya Khetarpal ([@agriyakhetarpal](https://github.com/agriyakhetarpal)) (Pyodide)
*   Hood Chatham ([@hoodmane](https://github.com/hoodmane)) (Pyodide)
*   Gyeongjae Choi ([@ryanking13](https://github.com/ryanking13)) (Pyodide)
*   Tim Felgentreff ([@timfel](https://github.com/timfel)) (GraalPy)

## Credits

`cibuildwheel` is built upon the work of many, including:

*   @matthew-brett for multibuild and delocate.
*   PyPA for the manylinux Docker images.
*   @ogrisel for wheelhouse-uploader.

(See the full list in the original README.)

## See Also

Consider [matthew-brett/multibuild](http://github.com/matthew-brett/multibuild) and [maturin-action](https://github.com/PyO3/maturin-action) for building Rust wheels.
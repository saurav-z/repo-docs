# cibuildwheel: Effortlessly Build Python Wheels for All Platforms

**Simplify your Python package distribution with cibuildwheel, automatically building and testing your wheels across macOS, Linux, Windows, and multiple Python versions.**  [View the original repository](https://github.com/pypa/cibuildwheel)

## Key Features:

*   **Cross-Platform Compatibility:** Build wheels for macOS, Linux (including manylinux and musllinux), Windows, Android and iOS.
*   **Comprehensive Python Version Support:** Supports CPython, PyPy, and GraalPy across various Python versions (3.8 - 3.14).
*   **Automated CI Integration:** Works seamlessly with GitHub Actions, Azure Pipelines, Travis CI, CircleCI, GitLab CI, and Cirrus CI.
*   **Dependency Management:** Handles shared library dependencies using auditwheel and delocate on Linux and macOS.
*   **Built-in Testing:** Runs your library's tests against the wheel-installed version for robust quality assurance.

## What cibuildwheel Does

`cibuildwheel` streamlines the creation of Python wheels across various operating systems and Python versions. It simplifies the complex process of building and testing your packages, ensuring compatibility across different platforms.

### Supported Platforms and Python Versions

| Platform                      | macOS Intel | macOS Apple Silicon | Windows 64bit | Windows 32bit | Windows Arm64 | manylinux/musllinux x86_64 | manylinux/musllinux i686 | manylinux/musllinux aarch64 | manylinux/musllinux ppc64le | manylinux/musllinux s390x | manylinux/musllinux armv7l | Android | iOS | Pyodide |
| :---------------------------- | :---------: | :---------------: | :-----------: | :-----------: | :-----------: | :--------------------------: | :--------------------------: | :---------------------------: | :---------------------------: | :---------------------------: | :--------------------------: | :-----: | :---: | :-----: |
| CPython 3.8                   |      ✅      |        ✅        |       ✅       |       ✅       |      N/A      |              ✅               |              ✅               |               ✅                |               ✅                |               ✅                |             ✅⁵             |   N/A   |  N/A  |   N/A   |
| CPython 3.9                   |      ✅      |        ✅        |       ✅       |       ✅       |      ✅²      |              ✅               |              ✅               |               ✅                |               ✅                |               ✅                |             ✅⁵             |   N/A   |  N/A  |   N/A   |
| CPython 3.10                  |      ✅      |        ✅        |       ✅       |       ✅       |      ✅²      |              ✅               |              ✅               |               ✅                |               ✅                |               ✅                |             ✅⁵             |   N/A   |  N/A  |   N/A   |
| CPython 3.11                  |      ✅      |        ✅        |       ✅       |       ✅       |      ✅²      |              ✅               |              ✅               |               ✅                |               ✅                |               ✅                |             ✅⁵             |   N/A   |  N/A  |   N/A   |
| CPython 3.12                  |      ✅      |        ✅        |       ✅       |       ✅       |      ✅²      |              ✅               |              ✅               |               ✅                |               ✅                |               ✅                |             ✅⁵             |   N/A   |  N/A  |   ✅⁴   |
| CPython 3.13                  |      ✅      |        ✅        |       ✅       |       ✅       |      ✅²      |              ✅               |              ✅               |               ✅                |               ✅                |               ✅                |             ✅⁵             |   ✅   |  ✅  |   N/A   |
| CPython 3.14                  |      ✅      |        ✅        |       ✅       |       ✅       |      ✅²      |              ✅               |              ✅               |               ✅                |               ✅                |               ✅                |             ✅⁵             |   N/A   |  N/A  |   N/A   |
| PyPy 3.8 v7.3                 |      ✅      |        ✅        |       ✅       |      N/A      |      N/A      |              ✅¹              |              ✅¹              |               ✅¹               |              N/A              |              N/A              |             N/A             |   N/A   |  N/A  |   N/A   |
| PyPy 3.9 v7.3                 |      ✅      |        ✅        |       ✅       |      N/A      |      N/A      |              ✅¹              |              ✅¹              |               ✅¹               |              N/A              |              N/A              |             N/A             |   N/A   |  N/A  |   N/A   |
| PyPy 3.10 v7.3                |      ✅      |        ✅        |       ✅       |      N/A      |      N/A      |              ✅¹              |              ✅¹              |               ✅¹               |              N/A              |              N/A              |             N/A             |   N/A   |  N/A  |   N/A   |
| PyPy 3.11 v7.3                |      ✅      |        ✅        |       ✅       |      N/A      |      N/A      |              ✅¹              |              ✅¹              |               ✅¹               |              N/A              |              N/A              |             N/A             |   N/A   |  N/A  |   N/A   |
| GraalPy 3.11 v24.2            |      ✅      |        ✅        |       ✅       |      N/A      |      N/A      |              ✅¹              |              N/A              |               ✅¹               |              N/A              |              N/A              |             N/A             |   N/A   |  N/A  |   N/A   |

<sup>¹ PyPy & GraalPy are only supported for manylinux wheels.</sup><br>
<sup>² Windows arm64 support is experimental.</sup><br>
<sup>³ Free-threaded mode requires opt-in on 3.13 using [`enable`](https://cibuildwheel.pypa.io/en/stable/options/#enable).</sup><br>
<sup>⁴ Experimental, not yet supported on PyPI, but can be used directly in web deployment. Use `--platform pyodide` to build.</sup><br>
<sup>⁵ manylinux armv7l support is experimental. As there are no RHEL based image for this architecture, it's using an Ubuntu based image instead.</sup><br>

## Usage in CI

`cibuildwheel` is designed to run within your CI environment.  The availability of platforms depends on the CI service.

|                 | Linux | macOS | Windows | Linux ARM | macOS ARM | Windows ARM | Android | iOS |
|-----------------|-------|-------|---------|-----------|-----------|-------------|---------|-----|
| GitHub Actions  |   ✅  |   ✅  |    ✅    |    ✅     |    ✅     |     ✅²     |   ✅⁴   |  ✅³  |
| Azure Pipelines |   ✅  |   ✅  |    ✅    |    -      |    ✅     |     ✅²     |   ✅⁴   |  ✅³  |
| Travis CI       |   ✅  |   -   |    ✅    |    ✅     |     -     |      -      |   ✅⁴   |   -   |
| CircleCI        |   ✅  |   ✅  |    -    |    ✅     |    ✅     |      -      |   ✅⁴   |  ✅³  |
| Gitlab CI       |   ✅  |   ✅  |    ✅    |    ✅¹    |    ✅     |      -      |   ✅⁴   |  ✅³  |
| Cirrus CI       |   ✅  |   ✅  |    ✅    |    ✅     |    ✅     |      -      |   ✅⁴   |   -   |

<sup>¹ [Requires emulation](https://cibuildwheel.pypa.io/en/stable/faq/#emulation), distributed separately. Other services may also support Linux ARM through emulation or third-party build hosts, but these are not tested in our CI.</sup><br>
<sup>² [Uses cross-compilation](https://cibuildwheel.pypa.io/en/stable/faq/#windows-arm64). It is not possible to test `arm64` on this CI platform.</sup><br>
<sup>³ Requires a macOS runner; runs tests on the simulator for the runner's architecture.</sup><br>
<sup>⁴ Building for Android requires the runner to be Linux x86_64, macOS ARM64 or macOS x86_64. Testing has [additional requirements](https://cibuildwheel.pypa.io/en/stable/platforms/#android).</sup><br>

## Example GitHub Actions Workflow

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
      - uses: actions/checkout@v5

      # Used to host cibuildwheel
      - uses: actions/setup-python@v5

      - name: Install cibuildwheel
        run: python -m pip install cibuildwheel==3.1.3

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

For more information, including PyPI deployment, and the use of other CI services or the dedicated GitHub Action, check out the [documentation](https://cibuildwheel.pypa.io) and the [examples](https://github.com/pypa/cibuildwheel/tree/main/examples).

## How It Works

The diagram illustrates the steps `cibuildwheel` takes on each platform:

![](docs/data/how-it-works.png)

<sup>Explore an interactive version of this diagram [in the docs](https://cibuildwheel.pypa.io/en/stable/#how-it-works).</sup>

## Configuration Options

[See the full list of options](https://cibuildwheel.pypa.io/en/stable/options/) for detailed customization.

|                      | Option                              | Description                                                                                                                          |
| :------------------- | :---------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| **Build Selection**    | `platform`                          | Override the auto-detected target platform                                                                                            |
|                      | `build`<br>`skip`                   | Choose the Python versions to build                                                                                                    |
|                      | `archs`                             | Change the architectures built on your machine by default.                                                                             |
|                      | `project-requires-python`           | Manually set the Python compatibility of your project                                                                                 |
|                      | `enable`                            | Enable building with extra categories of selectors present.                                                                            |
|                      | `allow-empty`                       | Suppress the error code if no wheels match the specified build identifiers                                                               |
| **Build Customization** | `build-frontend`                    | Set the tool to use to build, either "build" (default), "build\[uv\]", or "pip"                                                     |
|                      | `config-settings`                   | Specify config-settings for the build backend.                                                                                         |
|                      | `environment`                       | Set environment variables                                                                                                            |
|                      | `environment-pass`                  | Set environment variables on the host to pass-through to the container.                                                                |
|                      | `before-all`                        | Execute a shell command on the build system before any wheels are built.                                                               |
|                      | `before-build`                      | Execute a shell command preparing each wheel's build                                                                                    |
|                      | `xbuild-tools`                      | Binaries on the path that should be included in an isolated cross-build environment.                                                   |
|                      | `repair-wheel-command`              | Execute a shell command to repair each built wheel                                                                                      |
|                      | `manylinux-*-image`<br>`musllinux-*-image` | Specify manylinux / musllinux container images                                                                                       |
|                      | `container-engine`                  | Specify the container engine to use when building Linux wheels                                                                         |
|                      | `dependency-versions`               | Control the versions of the tools cibuildwheel uses                                                                                    |
|                      | `pyodide-version`                   | Specify the Pyodide version to use for `pyodide` platform builds                                                                       |
| **Testing**          | `test-command`                      | The command to test each built wheel                                                                                                   |
|                      | `before-test`                       | Execute a shell command before testing each wheel                                                                                       |
|                      | `test-sources`                      | Paths that are copied into the working directory of the tests                                                                            |
|                      | `test-requires`                     | Install Python dependencies before running the tests                                                                                     |
|                      | `test-extras`                       | Install your wheel for testing using `extras_require`                                                                                   |
|                      | `test-groups`                       | Specify test dependencies from your project's `dependency-groups`                                                                        |
|                      | `test-skip`                         | Skip running tests on some builds                                                                                                      |
|                      | `test-environment`                  | Set environment variables for the test environment                                                                                    |
| **Debugging**        | `debug-keep-container`              | Keep the container after running for debugging.                                                                                         |
|                      | `debug-traceback`                   | Print full traceback when errors occur.                                                                                               |
|                      | `build-verbosity`                   | Increase/decrease the output of the build                                                                                               |

## Working Examples

Here are some projects using cibuildwheel:

| Name                              | CI                  | OS                                     | Notes                                                                                                                             |
| :-------------------------------- | :------------------ | :------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------- |
| [scikit-learn][]                  | ![github icon][]    | ![windows icon][] ![apple icon][] ![linux icon][] | The machine learning library. A complex but clean config using many of cibuildwheel's features to build a large project with Cython and C++ extensions.  |
| [pytorch-fairseq][]               | ![github icon][]    | ![apple icon][] ![linux icon][]         | Facebook AI Research Sequence-to-Sequence Toolkit written in Python.                                                                 |
| [duckdb][]                        | ![github icon][]    | ![apple icon][] ![linux icon][] ![windows icon][] | DuckDB is an analytical in-process SQL database management system                                                                    |
| [NumPy][]                         | ![github icon][] ![travisci icon][] | ![windows icon][] ![apple icon][] ![linux icon][] | The fundamental package for scientific computing with Python.                                                                       |
| [Tornado][]                       | ![github icon][]    | ![linux icon][] ![apple icon][] ![windows icon][] | Tornado is a Python web framework and asynchronous networking library. Uses stable ABI for a small C extension.                                |
| [NCNN][]                          | ![github icon][]    | ![windows icon][] ![apple icon][] ![linux icon][] | ncnn is a high-performance neural network inference framework optimized for the mobile platform                                                 |
| [Matplotlib][]                    | ![github icon][]    | ![windows icon][] ![apple icon][] ![linux icon][] | The venerable Matplotlib, a Python library with C++ portions                                                                                |
| [MyPy][]                          | ![github icon][]    | ![apple icon][] ![linux icon][] ![windows icon][] | The compiled version of MyPy using MyPyC.                                                                                                  |
| [Prophet][]                       | ![github icon][]    | ![windows icon][] ![apple icon][] ![linux icon][] | Tool for producing high quality forecasts for time series data that has multiple seasonality with linear or non-linear growth.                                     |
| [Kivy][]                          | ![github icon][]    | ![windows icon][] ![apple icon][] ![linux icon][] | Open source UI framework written in Python, running on Windows, Linux, macOS, Android and iOS                                                 |

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

> ℹ️  Find more examples on the [Working Examples](https://cibuildwheel.pypa.io/en/stable/working-examples) page.

## Legal Note

`cibuildwheel` uses `delocate` or `auditwheel`, potentially bundling dynamically linked libraries. This can have license implications; review the licenses of any pulled-in code.

## Changelog

[View the changelog](https://cibuildwheel.pypa.io/en/stable/changelog/) for recent updates.

## Contributing

See the [contributing guidelines](https://cibuildwheel.pypa.io/en/latest/contributing/) for information on how to contribute to cibuildwheel.

## Maintainers

Core:

*   Joe Rickerby ([@joerick](https://github.com/joerick))
*   Yannick Jadoul ([@YannickJadoul](https://github.com/YannickJadoul))
*   Matthieu Darbois ([@mayeut](https://github.com/mayeut))
*   Henry Schreiner ([@henryiii](https://github.com/henryiii))
*   Grzegorz Bokota ([@Czaki](https://github.com/Czaki))

Platform maintainers:

*   Russell Keith-Magee ([@freakboy3742](https://github.com/freakboy3742)) (iOS)
*   Agriya Khetarpal ([@agriyakhetarpal](https://github.com/agriyakhetarpal)) (Pyodide)
*   Hood Chatham ([@hoodmane](https://github.com/hoodmane)) (Pyodide)
*   Gyeongjae Choi ([@ryanking13](https://github.com/ryanking13)) (Pyodide)
*   Tim Felgentreff ([@timfel](https://github.com/timfel)) (GraalPy)
*   Malcolm Smith ([@mhsmith](https://github.com/mhsmith)) (Android)

## Credits

`cibuildwheel` is built on the work of others.

*   ⭐️ @matthew-brett for [multibuild](https://github.com/multi-build/multibuild) and [matthew-brett/delocate](http://github.com/matthew-brett/delocate)
*   @PyPA for the manylinux Docker images [pypa/manylinux](https://github.com/pypa/manylinux)
*   @ogrisel for [wheelhouse-uploader](https://github.com/ogrisel/wheelhouse-uploader) and `run_with_env.cmd`

## See Also

Consider also: [matthew-brett/multibuild](http://github.com/matthew-brett/multibuild).
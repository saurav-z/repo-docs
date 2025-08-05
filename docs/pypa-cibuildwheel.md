# cibuildwheel: Automate Python Wheel Builds for Cross-Platform Compatibility

[![PyPI](https://img.shields.io/pypi/v/cibuildwheel.svg)](https://pypi.python.org/pypi/cibuildwheel)
[![Documentation Status](https://readthedocs.org/projects/cibuildwheel/badge/?version=stable)](https://cibuildwheel.pypa.io/en/stable/?badge=stable)
[![Actions Status](https://github.com/pypa/cibuildwheel/workflows/Test/badge.svg)](https://github.com/pypa/cibuildwheel/actions)
[![Travis Status](https://img.shields.io/travis/com/pypa/cibuildwheel/main?logo=travis)](https://travis-ci.com/github/pypa/cibuildwheel)
[![CircleCI Status](https://img.shields.io/circleci/build/gh/pypa/cibuildwheel/main?logo=circleci)](https://circleci.com/gh/pypa/cibuildwheel)
[![Azure Status](https://dev.azure.com/joerick0429/cibuildwheel/_apis/build/status/pypa.cibuildwheel?branchName=main)](https://dev.azure.com/joerick0429/cibuildwheel/_build/latest?definitionId=4&branchName=main)

[Documentation](https://cibuildwheel.pypa.io)

Tired of manual wheel builds for different Python versions and platforms? **cibuildwheel streamlines the process, automating the building and testing of your Python wheels across macOS, Linux, and Windows.**

**Key Features:**

*   **Cross-Platform Support:** Builds wheels for macOS (Intel & Apple Silicon), Linux (manylinux & musllinux), and Windows, covering various Python versions and architectures.
*   **CI/CD Integration:** Seamlessly integrates with popular CI/CD platforms like GitHub Actions, Azure Pipelines, Travis CI, CircleCI, GitLab CI, and Cirrus CI.
*   **Dependency Handling:** Bundles shared library dependencies on Linux and macOS using auditwheel and delocate, ensuring compatibility.
*   **Automated Testing:** Runs your library's tests against the wheel-installed version of your library.
*   **Comprehensive Platform Coverage:** Targets CPython, PyPy, and GraalPy interpreters, and experimental support for Pyodide, Android, and iOS.

**What cibuildwheel Builds:**

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

**How it Works:**

[![](docs/data/how-it-works.png)](https://cibuildwheel.pypa.io/en/stable/#how-it-works)

See the [cibuildwheel documentation](https://cibuildwheel.pypa.io) for more detailed information.

**Example GitHub Actions Setup:**

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

**Configuration Options:**

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

These options can be specified in a `pyproject.toml` file or as environment variables; consult the [configuration docs](https://cibuildwheel.pypa.io/en/latest/configuration/) for details.

**Working Examples:**

Check out the [working examples](https://cibuildwheel.pypa.io/en/stable/working-examples) page for real-world projects that utilize cibuildwheel.

**Contributing:**

See the [contributing guide](https://cibuildwheel.pypa.io/en/latest/contributing/) for information on how to contribute to cibuildwheel.

**Source Code:** [GitHub Repository](https://github.com/pypa/cibuildwheel)

**Legal Notes:**

Since `cibuildwheel` uses `delocate` or `auditwheel` to repair wheels, it may bundle dynamically linked libraries from the build machine, with potential license implications.

**Changelog:**

See [the changelog page in the docs](https://cibuildwheel.pypa.io/en/stable/changelog/) for the latest updates.
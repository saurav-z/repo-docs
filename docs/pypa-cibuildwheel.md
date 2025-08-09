# cibuildwheel: Automate Python Wheel Builds for Cross-Platform Compatibility

[![PyPI](https://img.shields.io/pypi/v/cibuildwheel.svg)](https://pypi.python.org/pypi/cibuildwheel)
[![Documentation Status](https://readthedocs.org/projects/cibuildwheel/badge/?version=stable)](https://cibuildwheel.pypa.io/en/stable/?badge=stable)
[![Actions Status](https://github.com/pypa/cibuildwheel/workflows/Test/badge.svg)](https://github.com/pypa/cibuildwheel/actions)
[![Travis Status](https://img.shields.io/travis/com/pypa/cibuildwheel/main?logo=travis)](https://travis-ci.com/github/pypa/cibuildwheel)
[![CircleCI Status](https://img.shields.io/circleci/build/gh/pypa/cibuildwheel/main?logo=circleci)](https://circleci.com/gh/pypa/cibuildwheel)
[![Azure Status](https://dev.azure.com/joerick0429/cibuildwheel/_apis/build/status/pypa.cibuildwheel?branchName=main)](https://dev.azure.com/joerick0429/cibuildwheel/_build/latest?definitionId=4&branchName=main)

Tired of manually building and testing Python wheels across multiple operating systems and Python versions? **cibuildwheel simplifies Python wheel creation by automating builds and testing across various platforms.** Find out more on [GitHub](https://github.com/pypa/cibuildwheel).

## Key Features

*   **Cross-Platform Support:** Build wheels for macOS (Intel & Apple Silicon), Windows (64-bit, 32-bit, Arm64), Linux (manylinux & musllinux), Android, iOS, and Pyodide.
*   **Multiple Python Versions:** Supports CPython, PyPy, and GraalPy across various Python versions (3.8+).
*   **CI/CD Integration:** Seamlessly integrates with GitHub Actions, Azure Pipelines, Travis CI, CircleCI, GitLab CI, and Cirrus CI.
*   **Dependency Management:** Bundles shared library dependencies on Linux and macOS using `auditwheel` and `delocate`.
*   **Automated Testing:** Runs your library's tests against the wheel-installed version.
*   **Flexible Configuration:** Configurable through `pyproject.toml` or environment variables.

## What cibuildwheel Builds

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

## How It Works

\[Diagram of how cibuildwheel works - see original README]

For a more detailed explanation, check out [the docs](https://cibuildwheel.pypa.io/en/stable/#how-it-works).

## Configuration Options

\[Table of options - see original README]

These options can be specified in a `pyproject.toml` file, or as environment variables; see [configuration docs](https://cibuildwheel.pypa.io/en/latest/configuration/).

## Example Setup

To build manylinux, musllinux, macOS, and Windows wheels on GitHub Actions, you could use this `.github/workflows/wheels.yml`:

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

## Working Examples

\[Table of examples - see original README]

> ℹ️ That's just a handful, there are many more! Check out the [Working Examples](https://cibuildwheel.pypa.io/en/stable/working-examples) page in the docs.

## Changelog

### v3.1.3

_1 August 2025_

- 🐛 Fix bug where "latest" dependencies couldn't update to pip 25.2 on Windows (#2537)
- 🛠 Use pytest-rerunfailures to improve some of our iOS/Android tests (#2527, #2539)
- 🛠 Remove some GraalPy Windows workarounds in our tests (#2501)

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

---

ℹ️ **Want more changelog? Head over to [the changelog page in the docs](https://cibuildwheel.pypa.io/en/stable/changelog/).**

## Legal Note

Since `cibuildwheel` repairs the wheel with `delocate` or `auditwheel`, it might automatically bundle dynamically linked libraries from the build machine.

It helps ensure that the library can run without any dependencies outside of the pip toolchain.

This is similar to static linking, so it might have some license implications. Check the license for any code you're pulling in to make sure that's allowed.

## Contributing

For more info on how to contribute to cibuildwheel, see the [docs](https://cibuildwheel.pypa.io/en/latest/contributing/).

Everyone interacting with the cibuildwheel project via codebase, issue tracker, chat rooms, or otherwise is expected to follow the [PSF Code of Conduct](https://github.com/pypa/.github/blob/main/CODE_OF_CONDUCT.md).

## Maintainers

Core:

*   Joe Rickerby [@joerick](https://github.com/joerick)
*   Yannick Jadoul [@YannickJadoul](https://github.com/YannickJadoul)
*   Matthieu Darbois [@mayeut](https://github.com/mayeut)
*   Henry Schreiner [@henryiii](https://github.com/henryiii)
*   Grzegorz Bokota [@Czaki](https://github.com/Czaki)

Platform maintainers:

*   Russell Keith-Magee [@freakboy3742](https://github.com/freakboy3742) (iOS)
*   Agriya Khetarpal [@agriyakhetarpal](https://github.com/agriyakhetarpal) (Pyodide)
*   Hood Chatham [@hoodmane](https://github.com/hoodmane) (Pyodide)
*   Gyeongjae Choi [@ryanking13](https://github.com/ryanking13) (Pyodide)
*   Tim Felgentreff [@timfel](https://github.com/timfel) (GraalPy)
*   Malcolm Smith [@mhsmith](https://github.com/mhsmith) (Android)

## Credits

`cibuildwheel` stands on the shoulders of giants.

*   ⭐️ @matthew-brett for [multibuild](https://github.com/multi-build/multibuild) and [matthew-brett/delocate](http://github.com/matthew-brett/delocate)
*   @PyPA for the manylinux Docker images [pypa/manylinux](https://github.com/pypa/manylinux)
*   @ogrisel for [wheelhouse-uploader](https://github.com/ogrisel/wheelhouse-uploader) and `run_with_env.cmd`

Massive props also to-

*   @zfrenchee for [help debugging many issues](https://github.com/pypa/cibuildwheel/issues/2)
*   @lelit for some great bug reports and [contributions](https://github.com/pypa/cibuildwheel/pull/73)
*   @mayeut for a [phenomenal PR](https://github.com/pypa/cibuildwheel/pull/71) patching Python itself for better compatibility!
*   @czaki for being a super-contributor over many PRs and helping out with countless issues!
*   @mattip for his help with adding PyPy support to cibuildwheel

## See Also

Another very similar tool to consider is [matthew-brett/multibuild](http://github.com/matthew-brett/multibuild). `multibuild` is a shell script toolbox for building a wheel on various platforms. It is used as a basis to build some of the big data science tools, like SciPy.

If you are building Rust wheels, you can get by without some of the tricks required to make GLIBC work via manylinux; this is especially relevant for cross-compiling, which is easy with Rust. See [maturin-action](https://github.com/PyO3/maturin-action) for a tool that is optimized for building Rust wheels and cross-compiling.
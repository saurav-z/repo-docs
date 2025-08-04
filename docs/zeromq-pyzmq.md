# PyZMQ: Powerful Python Bindings for ØMQ (ZeroMQ)

**Easily integrate fast and lightweight messaging into your Python projects with PyZMQ, the Python bindings for ØMQ.**  [View the original repository on GitHub](https://github.com/zeromq/pyzmq).

## Key Features of PyZMQ:

*   **Fast Messaging:** Leverages the speed and efficiency of ØMQ (ZeroMQ).
*   **Python Compatibility:**  Works with Python 3.8+ and PyPy.
*   **ØMQ Support:** Supports libzmq versions ≥ 3.2.2 (including 4.x).
*   **Easy Installation:**  Install with `pip install pyzmq` (pre-built wheels available).
*   **Source Installation Option:** Option to force installation from source, useful if you have a specific libzmq setup.
*   **Comprehensive Documentation:** Includes detailed API documentation on [Read the Docs](https://pyzmq.readthedocs.io) and an excellent [ØMQ Guide](http://zguide.zeromq.org/py:all).

## Installation

### Using `pip` (Recommended)

For most users, the easiest way to install PyZMQ is using `pip`:

```bash
pip install pyzmq
```

This will install pre-built wheels for macOS, Windows, and Linux, providing a hassle-free installation experience.  Ensure you have the latest version of `pip`.

### Building from Source (If Needed)

If the wheel installation fails, or you prefer to build from source (e.g., to use a custom libzmq configuration), you can force a source install:

```bash
pip install --no-binary=pyzmq pyzmq
```

For detailed build instructions, see the [PyZMQ documentation](https://pyzmq.readthedocs.io/en/latest/howto/build.html).

## Compatibility Notes

*   **Python Versions:** Supports Python 3.8 and later, as well as PyPy.
*   **libzmq Versions:** Compatible with libzmq ≥ 3.2.2 (including 4.x).
*   **Older Versions:**  For compatibility with older Python versions, you may need to specify an older version of PyZMQ.  See the original README for guidance.
*   **Versioning:** PyZMQ follows semantic versioning conventions.
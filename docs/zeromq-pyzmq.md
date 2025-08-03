# PyZMQ: Fast Messaging for Python with ZeroMQ

**PyZMQ empowers Python developers with lightning-fast and flexible messaging capabilities by providing Python bindings for ZeroMQ.**  [Explore the original repository](https://github.com/zeromq/pyzmq).

## Key Features

*   **Python Bindings for ZeroMQ:**  Seamlessly integrates Python with the powerful ZeroMQ messaging library.
*   **Cross-Platform Compatibility:** Works with Python 3.8+ and PyPy.
*   **ZeroMQ API Support:** Fully compatible with stable (non-DRAFT) 3.x and 4.x APIs of libzmq (libzmq ≥ 3.2.2).
*   **Easy Installation:**  Install pre-built wheels for macOS, Windows, and Linux via `pip install pyzmq`.  Source builds are also supported.
*   **Comprehensive Documentation:**  Detailed documentation available on [Read the Docs](https://pyzmq.readthedocs.io) and [the ZeroMQ Guide](http://zguide.zeromq.org/py:all).

## Installation

Easily install PyZMQ using `pip`:

```bash
pip install pyzmq
```

Forcing a source build (if needed):

```bash
pip install --no-binary=pyzmq pyzmq
```

## Compatibility

*   Supports Python 3.8+ and PyPy.
*   Compatible with libzmq ≥ 3.2.2 (including 4.x).
*   Older versions of PyZMQ are available to support older Python and libzmq versions:

    *   To support Python 2.6 and 3.2, use `pip install 'pyzmq<16'`
    *   For libzmq 2.0.x, use `pip install 'pyzmq<2.1'`

## Documentation and Resources

*   **PyZMQ Documentation:** [Read the Docs](https://pyzmq.readthedocs.io)
*   **ZeroMQ Guide (Python Version):** [http://zguide.zeromq.org/py:all](http://zguide.zeromq.org/py:all)
*   **GitHub Wiki:** [https://github.com/zeromq/pyzmq/wiki](https://github.com/zeromq/pyzmq/wiki)
*   **PyPI:** [https://pypi.io/project/pyzmq/](https://pypi.io/project/pyzmq/)
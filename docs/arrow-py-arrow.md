# Arrow: Simplify Dates and Times in Python

**Arrow is a powerful Python library that makes working with dates and times easy, intuitive, and human-friendly.** Say goodbye to complex datetime manipulations and hello to a streamlined experience.

[View the Arrow project on GitHub](https://github.com/arrow-py/arrow)

## Key Features

*   **User-Friendly:** Drop-in replacement for Python's `datetime` with an intuitive API.
*   **Timezone-Aware:** Built-in timezone support with UTC as the default.
*   **Easy Creation:** Simple methods for creating `Arrow` objects from various input formats.
*   **Flexible Manipulation:**  Effortlessly shift, format, and convert dates and times.
*   **ISO 8601 Support:**  Comprehensive support for the ISO 8601 standard.
*   **Humanization:** Generate human-readable representations of dates and times in multiple locales.
*   **Type Hinting:** Full support for PEP 484-style type hints.

## Why Choose Arrow?

Tired of wrestling with Python's built-in datetime modules? Arrow simplifies date and time operations by:

*   Reducing the number of modules and types you need to import and work with.
*   Making timezone conversions and timestamp manipulations significantly easier.
*   Providing essential functionality like ISO 8601 parsing, time spans, and humanization that's missing in the standard library.

## Quick Start

**Installation**

Install Arrow using `pip`:

```bash
pip install -U arrow
```

**Example Usage**

```python
import arrow

# Create an Arrow object from an ISO 8601 string
a = arrow.get('2013-05-11T21:23:58.970460+07:00')
print(a)  # Output: <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Work with UTC time
utc = arrow.utcnow()
print(utc)  # Output: <Arrow [2023-10-27T15:30:00.000000+00:00]>

# Shift time
utc_minus_hour = utc.shift(hours=-1)
print(utc_minus_hour) #Output: <Arrow [2023-10-27T14:30:00.000000+00:00]>

# Convert to another timezone
local = utc.to('US/Pacific')
print(local)  # Output: <Arrow [2023-10-27T07:30:00.000000-07:00]>

# Format the date
print(local.format('YYYY-MM-DD HH:mm:ss ZZ')) # Output: 2023-10-27 07:30:00 -07:00

# Humanize the date
print(local.humanize())  # Output: 8 hours ago
```

## Documentation

For comprehensive documentation, please visit: [arrow.readthedocs.io](https://arrow.readthedocs.io)

## Contributing

We welcome contributions!  Help us improve Arrow by:

1.  Finding an issue on the [issue tracker](https://github.com/arrow-py/arrow/issues). Look for issues labeled ["good first issue"](https://github.com/arrow-py/arrow/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) to start!
2.  Forking the repository and making changes in a branch.
3.  Adding tests to verify your changes.
4.  Running the test suite with `tox && tox -e lint,docs` or `make build39 && make test && make lint`.
5.  Submitting a pull request.

Have questions?  Ask them on the [discussions](https://github.com/arrow-py/arrow/discussions) page.

## Support Arrow

Support the development of Arrow by making a donation on the [Open Collective](https://opencollective.com/arrow).
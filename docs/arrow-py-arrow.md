# Arrow: Effortless Date and Time Handling in Python

**Simplify your Python date and time manipulations with Arrow, a user-friendly library built for human-readable and intuitive date and time management.**  [View the source code on GitHub](https://github.com/arrow-py/arrow)

## Key Features:

*   **Intuitive Date & Time Creation:** Easily create Arrow objects from various inputs.
*   **Timezone Awareness by Default:** Built-in support for timezones and UTC.
*   **Simplified Time Shifting:**  Use the `shift` method for easy relative date and time adjustments.
*   **Automatic Formatting & Parsing:** Works seamlessly with strings, including ISO 8601.
*   **Timezone Conversion:** Effortlessly convert between timezones.
*   **Human-Friendly Formatting:**  Generate human-readable date and time representations.
*   **ISO 8601 Support:**  Extensive support for the ISO 8601 standard.
*   **Time Span, Range, Floor & Ceiling:** Calculate timeframes from microseconds to years.
*   **Localization:** Humanize dates and times with locale support.
*   **Type Hints:**  Full support for PEP 484-style type hints.

## Why Choose Arrow?

Arrow solves common pain points in Python's built-in datetime modules:

*   **Simplified Imports:** Reduces the need to juggle multiple modules (datetime, time, calendar, etc.).
*   **Fewer Types:** Works with a more streamlined set of date and time types.
*   **Easier Timezone Handling:** Makes timezone conversions and timestamp manipulations more straightforward.
*   **Enhanced Functionality:** Provides features like ISO 8601 parsing, time spans, and humanization, unavailable in standard libraries.

## Quick Start

### Installation

Install Arrow using `pip`:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Create an Arrow object
now = arrow.utcnow()
print(now)  # <Arrow [2023-10-27T12:34:56.789000+00:00]>

# Shift time
past = now.shift(days=-2)
print(past) # <Arrow [2023-10-25T12:34:56.789000+00:00]>

# Convert to a different timezone
local = now.to('US/Pacific')
print(local)  # <Arrow [2023-10-27T05:34:56.789000-07:00]>

# Format the date and time
print(local.format('YYYY-MM-DD HH:mm:ss ZZ'))  # Output: 2023-10-27 05:34:56 -07:00

# Get a humanized representation
print(local.humanize()) # Output: "5 hours ago"
```

## Documentation

For complete details, please visit the official documentation:  [arrow.readthedocs.io](https://arrow.readthedocs.io)

## Contributing

We welcome contributions!  Learn how to contribute:

1.  Find an issue or feature to tackle on the [issue tracker](https://github.com/arrow-py/arrow/issues).
2.  Fork the repository and create a branch.
3.  Add tests to ensure bug fixes or new features work.
4.  Run `tox && tox -e lint,docs` (if you have `tox`) or `make build39 && make test && make lint` (if you don't have Python 3.9) to test.
5.  Submit a pull request.
6.  Ask questions on [GitHub Discussions](https://github.com/arrow-py/arrow/discussions).

## Support Arrow

Support the Arrow project by making a financial contribution via [Open Collective](https://opencollective.com/arrow).
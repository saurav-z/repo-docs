# Arrow: Elegant Date and Time Handling in Python

**Simplify your Python date and time manipulations with Arrow, a user-friendly library that makes working with dates and times intuitive and enjoyable.**  Learn more about Arrow on its [GitHub repository](https://github.com/arrow-py/arrow).

## Key Features of Arrow:

*   **Intuitive and Human-Friendly:**  Provides a clean API for working with dates and times, inspired by moment.js and requests.
*   **Drop-in Replacement for `datetime`:** Seamlessly integrates with existing Python code.
*   **Timezone-Aware by Default:**  Handles timezones intelligently and supports UTC.
*   **Simplified Creation:**  Easy creation options for common input formats.
*   **Flexible Time Shifting:**  Uses the `shift` method for relative offsets, including weeks.
*   **Automatic String Formatting & Parsing:** Simplifies the handling of date and time strings.
*   **Robust ISO 8601 Support:** Extensive support for the ISO 8601 standard.
*   **Timezone Conversion:**  Effortlessly convert between timezones.
*   **Date and Time Span Operations:** Generates time spans, ranges, floors, and ceilings.
*   **Humanization:**  Human-readable date and time representations with locale support.
*   **Extensible:**  Supports custom Arrow-derived types.
*   **Type Hinting:** Fully supports PEP 484-style type hints.

## Why Use Arrow?

Compared to Python's built-in `datetime` module, Arrow addresses common usability pain points:

*   **Reduces Module Overload:** Avoids juggling multiple modules (e.g., `datetime`, `time`, `calendar`).
*   **Simplifies Data Types:**  Streamlines working with various datetime types (e.g., `date`, `time`, `datetime`).
*   **Improves Timezone Handling:**  Makes timezone conversions less verbose and more intuitive.
*   **Addresses Functionality Gaps:** Provides essential features like ISO 8601 parsing and time spans.

## Quick Start

### Installation

Install Arrow using `pip`:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Parse a datetime string
a = arrow.get('2013-05-11T21:23:58.970460+07:00')
print(a)  # Output: <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get UTC time
utc = arrow.utcnow()
print(utc)  # Output: <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift time
utc = utc.shift(hours=-1)
print(utc)  # Output: <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to another timezone
local = utc.to('US/Pacific')
print(local) # Output: <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get timestamp
print(local.timestamp())  # Output: 1368303838.970460

# Format the date and time
print(local.format())  # Output: 2013-05-11 13:23:58 -07:00
print(local.format('YYYY-MM-DD HH:mm:ss ZZ')) # Output: 2013-05-11 13:23:58 -07:00

# Humanize the date
print(local.humanize())  # Output: an hour ago
print(local.humanize(locale='ko-kr'))  # Output: 한시간 전
```

## Documentation

For complete documentation and further details, please visit [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome! Follow the contributing guidelines to submit code and locale updates.

1.  Find an issue or feature to tackle on the [issue tracker](https://github.com/arrow-py/arrow/issues).
2.  Fork the repository on GitHub and create a branch.
3.  Add tests to ensure your changes work as expected.
4.  Run tests and linting: `tox && tox -e lint,docs` or `make build39 && make test && make lint`.
5.  Submit a pull request.

## Support Arrow

Consider supporting the project through [Open Collective](https://opencollective.com/arrow) to help fund Arrow's development.
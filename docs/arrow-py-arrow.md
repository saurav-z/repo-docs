# Arrow: Date and Time Made Easy for Python

**Tired of complex date and time manipulations in Python?** Arrow simplifies your work with dates, times, and timestamps, making them human-friendly and intuitive.  Learn more and contribute at the [original repository](https://github.com/arrow-py/arrow).

## Key Features of Arrow

*   **Intuitive datetime Replacement:**  A fully-implemented, drop-in replacement for Python's built-in datetime objects.
*   **Python Version Support:** Compatible with Python 3.8 and later.
*   **Timezone Awareness:**  Timezone-aware and UTC by default for easy timezone handling.
*   **Simplified Creation:**  Easy creation options for common input formats.
*   **Flexible Time Shifting:**  Use the `shift` method for relative offsets, including weeks.
*   **Automatic Formatting and Parsing:**  Handles string formatting and parsing automatically.
*   **ISO 8601 Support:** Wide support for the ISO 8601 standard for date and time representation.
*   **Timezone Conversion:** Effortlessly convert between timezones.
*   **Integration:** Supports `dateutil`, `pytz`, and `ZoneInfo` tzinfo objects.
*   **Time Span Generation:** Generates time spans, ranges, floors, and ceilings.
*   **Human-Friendly Output:**  Humanize dates and times with support for multiple locales.
*   **Extensibility:** Extensible for your own Arrow-derived types
*   **Type Hinting:** Full support for PEP 484-style type hints.

## Why Choose Arrow?

Arrow addresses the usability challenges of Python's standard datetime modules:

*   **Reduced Complexity:** Avoids the need to juggle multiple modules and types.
*   **Simplified Timezone Handling:** Makes timezone conversions and timestamp handling easier.
*   **Improved Usability:** Provides a more intuitive and user-friendly API.

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
arw = arrow.get('2013-05-11T21:23:58.970460+07:00')
print(arw)  # Output: <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()
print(utc) # Output: <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift time
utc = utc.shift(hours=-1)
print(utc)  # Output: <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert timezones
local = utc.to('US/Pacific')
print(local)  # Output: <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get a timestamp
print(local.timestamp())  # Output: 1368303838.970460

# Format the time
print(local.format())  # Output: 2013-05-11 13:23:58 -07:00
print(local.format('YYYY-MM-DD HH:mm:ss ZZ')) # Output: 2013-05-11 13:23:58 -07:00

# Humanize the time
print(local.humanize())  # Output: an hour ago
print(local.humanize(locale='ko-kr')) # Output: 한시간 전
```

## Documentation

For comprehensive information, visit the official documentation at [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

We welcome contributions!  Find an issue on the [issue tracker](https://github.com/arrow-py/arrow/issues), fork the repository, and start making changes.  Consider tackling "good first issue" labels.  Submit a pull request after adding tests and running the test suite. For discussions and questions, visit our [discussions](https://github.com/arrow-py/arrow/discussions).

## Support Arrow

Support the Arrow project by contributing financially through [Open Collective](https://opencollective.com/arrow) to help fund its continued development.
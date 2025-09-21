# Arrow: Human-Friendly Dates and Times for Python

**Tired of cumbersome date and time manipulation in Python?** Arrow is a powerful Python library that simplifies working with dates, times, and timestamps.  Check out the original repository for more details: [https://github.com/arrow-py/arrow](https://github.com/arrow-py/arrow)

## Key Features

*   **Simplified Date and Time Handling:** A drop-in replacement for Python's `datetime` module, making date and time operations more intuitive.
*   **Python 3.8+ Compatibility:** Fully compatible with the latest Python versions.
*   **Timezone Awareness:** Timezone-aware and UTC by default, reducing common timezone headaches.
*   **Easy Creation:** Simple creation methods for various input formats.
*   **Flexible Time Shifting:** The `shift` method supports relative offsets, including weeks, for easy time manipulation.
*   **Automatic Formatting and Parsing:**  Intelligent parsing and formatting of strings.
*   **ISO 8601 Support:** Comprehensive support for the ISO 8601 standard.
*   **Timezone Conversion:** Effortlessly convert between timezones.
*   **`dateutil`, `pytz`, and `ZoneInfo` Integration:** Works seamlessly with existing timezone libraries.
*   **Time Span Generation:** Generate time spans, ranges, floors, and ceilings for different time frames.
*   **Humanization:**  Human-readable date and time formats with support for multiple locales.
*   **Extensibility:** Create your own Arrow-derived types.
*   **Type Hinting:**  Fully supports PEP 484-style type hints.

## Why Choose Arrow?

Arrow addresses the usability challenges of Python's built-in date and time modules by:

*   **Consolidating Modules:** Reducing the need to import multiple modules (e.g., `datetime`, `time`, `calendar`, `dateutil`, `pytz`).
*   **Simplifying Data Types:** Working with fewer data types like `date`, `time`, `datetime`, etc.
*   **Making Timezone Operations Easy:** Simplifying and making timezone conversions more intuitive.
*   **Defaulting to Timezone Awareness:**  Encouraging timezone-aware datetime objects.
*   **Adding Functionality:**  Providing functionality like ISO 8601 parsing, timespans, and humanization.

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
arrow.get('2013-05-11T21:23:58.970460+07:00')
# <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()
# <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift the time by one hour
utc = utc.shift(hours=-1)
# <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to a different timezone
local = utc.to('US/Pacific')
# <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get the timestamp
local.timestamp()
# 1368303838.970460

# Format the date and time
local.format()
# '2013-05-11 13:23:58 -07:00'
local.format('YYYY-MM-DD HH:mm:ss ZZ')
# '2013-05-11 13:23:58 -07:00'

# Humanize the date and time
local.humanize()
# 'an hour ago'

# Humanize in a different locale
local.humanize(locale='ko-kr')
# '한시간 전'
```

## Documentation

For comprehensive documentation, visit [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome!  Follow these steps to contribute:

1.  Find an issue on the [issue tracker](https://github.com/arrow-py/arrow/issues), especially those labeled "good first issue".
2.  Fork the repository and create a branch.
3.  Add tests for your changes.
4.  Run tests using `tox` or `make build39 && make test && make lint`.
5.  Submit a pull request.

## Support Arrow

Support the project by donating on the [Open Collective](https://opencollective.com/arrow).
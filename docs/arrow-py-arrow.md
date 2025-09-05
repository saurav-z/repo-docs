# Arrow: Human-Friendly Dates and Times for Python

**Tired of wrestling with Python's built-in datetime modules?** Arrow is a Python library designed to make working with dates and times a breeze, offering a more intuitive and user-friendly experience. Visit the [original repository](https://github.com/arrow-py/arrow) for more information.

## Key Features of Arrow:

*   **Simplified API:** Easily create, manipulate, and format dates and times with less code.
*   **Timezone-Aware by Default:** Work with timezones effortlessly, including UTC support.
*   **Human-Friendly Formatting:** Format dates and times into easy-to-read strings or humanized output (e.g., "2 hours ago").
*   **Intuitive Shifting:** Shift dates and times forward and backward with relative offsets (e.g., days, weeks).
*   **ISO 8601 Support:** Seamlessly parse and format dates and times according to the ISO 8601 standard.
*   **Date and Time Range Operations:** Generates time spans, ranges, floors and ceilings for time frames ranging from microsecond to year.
*   **Locale Support:** Humanize dates and times with a growing list of contributed locales, supporting many different languages.
*   **Python 3.8+ Compatibility:** Compatible with the latest Python versions.
*   **Type Hints:** Full support for PEP 484-style type hints.

## Why Choose Arrow?

Arrow addresses the pain points of Python's standard library by:

*   Reducing the need to import multiple modules (datetime, time, etc.)
*   Simplifying complex timezone conversions and timestamp handling.
*   Providing a more intuitive and user-friendly API.
*   Closing the gaps in functionality with ISO 8601 parsing, timespans, humanization

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

# Get the current UTC time
utc = arrow.utcnow()
print(utc) # Output: <Arrow [2024-10-27T12:34:56.789012+00:00]> (Example)

# Shift the time
utc = utc.shift(hours=-1)
print(utc)  # Output: <Arrow [2024-10-27T11:34:56.789012+00:00]> (Example)

# Convert to a different timezone
local = utc.to('US/Pacific')
print(local) # Output: <Arrow [2024-10-27T04:34:56.789012-07:00]> (Example)

# Format the output
print(local.format())  # Output: 2024-10-27 04:34:56 -07:00 (Example)
print(local.format('YYYY-MM-DD HH:mm:ss ZZ')) # Output: 2024-10-27 04:34:56 -07:00 (Example)

# Humanize the output
print(local.humanize())  # Output: 7 hours ago (Example)
print(local.humanize(locale='ko-kr')) # Output: 7시간 전 (Example)
```

## Documentation

For comprehensive documentation, please visit [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome! To contribute, follow these steps:

1.  Find an issue or feature to work on in the [issue tracker](https://github.com/arrow-py/arrow/issues). Issues marked with the "good first issue" label may be a great place to start!
2.  Fork the repository on GitHub and create a branch for your changes.
3.  Add tests to cover your changes.
4.  Run the test suite and linting checks (using `tox` or `make` commands).
5.  Submit a pull request.

## Support Arrow

If you would like to support the development of Arrow, consider making a financial contribution via the [Arrow collective on Open Collective](https://opencollective.com/arrow).
# Arrow: Human-Friendly Dates and Times for Python

**Simplify your Python date and time manipulations with Arrow, a powerful library that makes working with dates and times intuitive and efficient.**  Learn more and contribute at the [original Arrow repository](https://github.com/arrow-py/arrow).

## Key Features

*   **Intuitive Date and Time Creation:** Easily create Arrow objects from various input formats.
*   **Timezone Awareness:**  Works with timezones, UTC by default, simplifying conversions.
*   **Flexible Time Shifting:** Use the `shift` method for easy relative date and time adjustments.
*   **Automatic Formatting and Parsing:** Format and parse strings automatically, including full support for ISO 8601.
*   **Time Span Generation:** Generate time spans, ranges, floors and ceilings.
*   **Human-Readable Dates and Times:**  Humanize dates and times with locale support.
*   **Drop-in Replacement:** Fully implemented and drop-in replacement for datetime.
*   **Python 3.8+ Support:** Compatible with the latest Python versions.

## Why Choose Arrow?

Arrow solves the complexities of Python's built-in `datetime` and related modules:

*   **Simplifies imports:** Reduces the need to import multiple date/time-related modules.
*   **Streamlines types:** Works with fewer date/time-related object types.
*   **User-friendly Timezones:** Offers intuitive timezone handling.
*   **Feature-rich:** Adds functionality, such as ISO 8601 parsing and humanization, missing from the standard library.

## Quick Start

### Installation

Install Arrow using `pip`:

```bash
pip install -U arrow
```

### Example Usage

```python
import arrow

# Create an Arrow object from a string
arrow.get('2013-05-11T21:23:58.970460+07:00')
# Output: <Arrow [2013-05-11T21:23:58.970460+07:00]>

# Get the current UTC time
utc = arrow.utcnow()
# Output: <Arrow [2013-05-11T21:23:58.970460+00:00]>

# Shift the time back one hour
utc = utc.shift(hours=-1)
# Output: <Arrow [2013-05-11T20:23:58.970460+00:00]>

# Convert to another timezone
local = utc.to('US/Pacific')
# Output: <Arrow [2013-05-11T13:23:58.970460-07:00]>

# Get the timestamp
local.timestamp()
# Output: 1368303838.970460

# Format the date and time
local.format()
# Output: '2013-05-11 13:23:58 -07:00'

# Format with a custom format string
local.format('YYYY-MM-DD HH:mm:ss ZZ')
# Output: '2013-05-11 13:23:58 -07:00'

# Get a human-readable representation
local.humanize()
# Output: 'an hour ago'

# Get a localized human-readable representation
local.humanize(locale='ko-kr')
# Output: '한시간 전'
```

## Documentation

For comprehensive documentation, visit [arrow.readthedocs.io](https://arrow.readthedocs.io).

## Contributing

Contributions are welcome!  To contribute:

1.  Find an issue on the [issue tracker](https://github.com/arrow-py/arrow/issues).
2.  Fork the repository and create a branch.
3.  Add tests for any new changes.
4.  Run tests: `tox && tox -e lint,docs` or `make build39 && make test && make lint`
5.  Submit a pull request.

## Support Arrow

Support the project via [Open Collective](https://opencollective.com/arrow).
# yt-fts: Unlock YouTube Insights with Powerful Full-Text Search

**Quickly search and analyze YouTube video transcripts with `yt-fts`, your command-line companion for in-depth YouTube content exploration.** [View the original repo](https://github.com/NotJoeMartinez/yt-fts)

<p align="center">
  <img src="https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14" alt="yt-fts demo" width="80%"/>
</p>

## Key Features

*   **Full-Text Search:**  Find specific keywords and phrases within YouTube video transcripts.
*   **Semantic Search:** Leverage OpenAI or Gemini embeddings for intelligent, context-aware searches.
*   **Channel and Video Management:** Download, update, list, and delete channels and their transcripts.
*   **Interactive LLM Chatbot:** Engage in conversational Q&A using semantic search results as context.
*   **Video Summarization:** Generate concise summaries with time-stamped links to relevant video sections.
*   **Flexible Export Options:** Export search results and transcripts in various formats (CSV, TXT, VTT).
*   **Advanced Search Syntax:** Utilize SQLite's Enhanced Query Syntax for powerful search queries.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Command Reference

### `download`: Download Subtitles

Download subtitles for a YouTube channel or playlist.

```bash
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
```

**Options:**

*   `-p, --playlist`: Download from a playlist.
*   `-l, --language`: Subtitle language (default: `en`).
*   `-j, --jobs`: Number of parallel download jobs (default: 8).
*   `--cookies-from-browser`: Use browser cookies (e.g., `firefox`) to handle login issues.

### `diagnose`: Diagnose Download Issues

Test your connection to YouTube and troubleshoot common download problems.

```bash
yt-fts diagnose
```

**Options:**

*   `-u, --test-url`: Test a specific URL.
*   `--cookies-from-browser`: Use browser cookies.
*   `-j, --jobs`: Number of parallel jobs for testing.

### `list`: List Saved Data

List saved channels, videos, and transcripts.

```bash
yt-fts list
```

**Options:**

*   `-t, --transcript`: Show transcript for a video.
*   `-c, --channel`: Show videos for a channel.
*   `-l, --library`: Show list of channels in library.

### `update`: Update Subtitles

Update subtitles for all or a specific channel.

```bash
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: Specify the channel to update.
*   `-l, --language`: Subtitle language.
*   `-j, --jobs`: Number of parallel jobs.
*   `--cookies-from-browser`: Use browser cookies.

### `delete`: Delete a Channel

Delete a channel and its associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel name or ID (required).

### `export`: Export Transcripts

Export transcripts for a channel to a file.

```bash
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: Channel name or ID (required).
*   `-f, --format`: Export format (`txt` or `vtt`, default: `txt`).

### `search`: Full Text Search

Search for keywords within video transcripts.

```bash
yt-fts search "search query" --channel "channel name or id"
```

**Options:**

*   `-c, --channel`: Search within a specific channel.
*   `-v, --video-id`: Search within a specific video.
*   `-l, --limit`: Limit the number of results.
*   `-e, --export`: Export search results to CSV.

**Advanced Search Syntax:**  Supports [SQLite's Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries), including AND, OR, and wildcard searches.

### `embeddings`: Semantic Search Setup

Generate embeddings for a channel to enable semantic search. Requires an OpenAI or Gemini API key set in the `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable.

```bash
yt-fts embeddings --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel name or ID.
*   `--api-key`: Your API key.
*   `-i, --interval`: Split transcripts into chunks (default: 30 seconds).

### `vsearch`: Semantic Search

Perform semantic (vector) search within a channel's embeddings.  Requires enabling embeddings with the `embeddings` command.

```bash
yt-fts vsearch "search query" --channel "channel name or id"
```

**Options:**

*   `-c, --channel`: Search within a specific channel.
*   `-v, --video-id`: Search within a specific video.
*   `-l, --limit`: Limit the number of results (default: 10).
*   `-e, --export`: Export search results to CSV.
*   `--api-key`: Your API key.

### `llm`: LLM Chatbot

Engage in an interactive chat session with a language model using semantic search results as context.  Requires semantic search enabled.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: Channel name or ID (required).
*   `--api-key`: Your API key.

### `summarize`: Video Summarization

Get a summary of a YouTube video transcript.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Model to use in summary
*   `--api-key`: Your API key.

### `config`: Show Configuration

Display the current configuration settings, including database and Chroma paths.

```bash
yt-fts config
```

## Practical How-Tos

**Export Search Results:**  Use the `--export` flag with `search` or `vsearch` to save results to a CSV file.

```bash
yt-fts search "python tutorial" --export
yt-fts vsearch "machine learning applications" --export
```

**Delete a Channel:**  Remove a channel and its data using the `delete` command.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:**  Update subtitles for a channel using the `update` command. (Note: This currently only updates full-text search data and does not update semantic search embeddings.)

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export a Channel's Transcript:**  Export a channel's transcript to a file (TXT or VTT format).

```bash
yt-fts export --channel "[channel id/name]" --format vtt
```
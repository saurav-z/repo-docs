# yt-fts: Unleash the Power of YouTube Search with Full Text Search and Semantic Capabilities

**yt-fts** is a command-line tool that empowers you to search within YouTube video subtitles, offering both full-text and semantic search capabilities. Dive deep into your favorite channels and discover insights like never before. ([Original Repo](https://github.com/NotJoeMartinez/yt-fts))

[![yt-fts demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

## Key Features

*   **Full-Text Search:** Quickly search through video transcripts using keywords and phrases, including advanced search syntax.
*   **Semantic Search:** Leverage OpenAI or Gemini's embeddings to find videos based on meaning and context.
*   **LLM Integration:** Chat with an LLM using semantic search results as context for in-depth analysis.
*   **Video Summarization:** Generate concise summaries of YouTube videos with time-stamped URLs.
*   **Flexible Data Management:** Download, update, list, and delete channels and video transcripts.
*   **Export Capabilities:** Export search results and transcripts in various formats.
*   **Diagnostic Tools:** Troubleshoot common download issues with the diagnose command.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### `download`: Download Subtitles

Downloads subtitles for a channel or playlist.

```bash
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"
```

**Options:**

*   `-p, --playlist`: Download all videos from a playlist
*   `-l, --language`: Language of the subtitles to download (default: en)
*   `-j, --jobs`: Number of parallel download jobs (default: 8, recommended: 4-16)
*   `--cookies-from-browser`: Browser to extract cookies from (chrome, firefox, etc.)

### `diagnose`: Diagnose Download Issues

Tests the connection to YouTube and provides recommendations for fixing issues.

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: URL to test with (default: https://www.youtube.com/watch?v=dQw4w9WgXcQ)
*   `--cookies-from-browser`: Browser to extract cookies from
*   `-j, --jobs`: Number of parallel download jobs to test with (default: 8)

### `list`: List Saved Data

Lists saved channels, videos, and transcripts.

```bash
yt-fts list
yt-fts list --channel "3Blue1Brown"
yt-fts list --transcript "dQw4w9WgXcQ"
```

**Options:**

*   `-t, --transcript`: Show transcript for a video
*   `-c, --channel`: Show list of videos for a channel
*   `-l, --library`: Show list of channels in library

### `update`: Update Subtitles

Updates subtitles for all or a specific channel.

```bash
yt-fts update
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: The name or id of the channel to update
*   `-l, --language`: Language of the subtitles to download (default: en)
*   `-j, --jobs`: Number of parallel download jobs (default: 8)
*   `--cookies-from-browser`: Browser to extract cookies from

### `delete`: Delete a Channel

Deletes a channel and all its associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to delete (required)

### `export`: Export Transcripts

Exports transcripts for a channel to a specified format.

```bash
yt-fts export --channel "3Blue1Brown" --format txt
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: The name or id of the channel to export transcripts for (required)
*   `-f, --format`: The format to export transcripts to. Supported formats: txt, vtt (default: txt)

### `search`: Full Text Search

Performs a full-text search across saved transcripts using an SQLite full-text index.

```bash
yt-fts search "[search query]" --channel "[channel name or id]"
yt-fts search "[search query]" --video-id "[video id]"
yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to search in
*   `-v, --video-id`: The id of the video to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export search results to a CSV file

### `embeddings`: Generate Semantic Embeddings

Generates semantic embeddings for a channel's transcripts using OpenAI or Gemini.

```bash
# Make sure API key is set
# export OPENAI_API_KEY="[yourOpenAIKey]"
# or
# export GEMINI_API_KEY="[yourGeminiKey]"
yt-fts embeddings --channel "3Blue1Brown"
yt-fts embeddings --interval 60 --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to generate embeddings for
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)
*   `-i, --interval`: Interval in seconds to split the transcripts into chunks (default: 30)

### `vsearch`: Semantic Search

Performs a semantic search using embeddings, returning results based on similarity.

```bash
yt-fts vsearch "[search query]" --channel "[channel name or id]"
yt-fts vsearch "[search query]" --video-id "[video id]"
yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to search in
*   `-v, --video-id`: The id of the video to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export search results to a CSV file
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `llm`: Chat Bot

Starts an interactive chat session using semantic search results as context.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to use (required)
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `summarize`: Summarize a Video

Summarizes a YouTube video transcript.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
yt-fts summarize "9-Jl0dxWQs8"
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Model to use in summary
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `config`: View Configuration

Shows the current configuration settings.

```bash
yt-fts config
```

## How To

*   **Export Search Results:** Use the `--export` flag with `search` and `vsearch` to save results to a CSV file.
*   **Delete a Channel:**  Use the `delete` command with the channel name or ID.
*   **Update a Channel:** Use the `update` command with the channel name or ID.
*   **Export a Channel's Transcript:** Use the `export` command with the desired format (txt or vtt).
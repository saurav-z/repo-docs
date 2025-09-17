# yt-fts: Your Command-Line YouTube Search Engine

**Quickly and efficiently search the full transcripts of YouTube channels using keywords and semantic search, all from your terminal.**  ([Original Repo](https://github.com/NotJoeMartinez/yt-fts))

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

**Key Features:**

*   **Full Text Search:**  Search entire YouTube video transcripts using keywords and phrases.
*   **Semantic Search:** Leverage AI embeddings (OpenAI, Gemini, or ChromaDB) for more relevant search results.
*   **Channel and Playlist Support:** Download and search subtitles from entire channels or playlists.
*   **Command-Line Interface:**  Easily search YouTube from the command line.
*   **Flexible Export Options:** Export search results to CSV or transcripts to TXT/VTT formats.
*   **LLM Integration:**  Chat directly with a model using your search results as context.
*   **Video Summarization:** Generate concise summaries of YouTube video transcripts.

## Installation

Install yt-fts using pip:

```bash
pip install yt-fts
```

## Commands

### `download`

Download subtitles from a YouTube channel or playlist and store them in a searchable database.

```bash
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
```

**Options:**

*   `-p, --playlist`: Download all videos from a playlist.
*   `-l, --language`:  Specify the subtitle language (default: `en`).
*   `-j, --jobs`:  Set the number of parallel download jobs (default: `8`, recommended: `4-16`).
*   `--cookies-from-browser`: Use cookies from your browser to bypass login requirements.

### `diagnose`

Diagnose and resolve download issues, particularly 403 errors.

```bash
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: URL to test (default: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`).
*   `--cookies-from-browser`: Browser to extract cookies from.
*   `-j, --jobs`: Number of parallel download jobs to test with (default: `8`).

### `list`

List saved channels, videos, and transcripts.

```bash
yt-fts list --channel "3Blue1Brown"
```

**Options:**

*   `-t, --transcript`: Show the transcript for a specific video.
*   `-c, --channel`: Show a list of videos for a specific channel.
*   `-l, --library`: Show a list of all channels in your library.

### `update`

Update subtitles for existing channels.

```bash
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: Update a specific channel.
*   `-l, --language`: Specify the subtitle language (default: `en`).
*   `-j, --jobs`: Number of parallel download jobs (default: `8`).
*   `--cookies-from-browser`: Browser to extract cookies from.

### `delete`

Delete a channel and all its associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: The name or ID of the channel to delete (required).

### `export`

Export transcripts for a channel to TXT or VTT format.

```bash
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: The name or ID of the channel to export (required).
*   `-f, --format`:  Output format: `txt` or `vtt` (default: `txt`).

### `search` (Full Text Search)

Search for keywords within the saved transcripts.  Supports advanced search syntax.

```bash
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"
```

**Options:**

*   `-c, --channel`: Search within a specific channel.
*   `-v, --video-id`: Search within a specific video.
*   `-l, --limit`:  Limit the number of results (default: `10`).
*   `-e, --export`:  Export results to a CSV file.

### Semantic Search and RAG

### `embeddings`

Generate embeddings for a channel to enable semantic (vector) search,  using OpenAI or Gemini.  Requires API key in `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable or with the `--api-key` flag.

```bash
yt-fts embeddings --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`:  The channel to generate embeddings for.
*   `--api-key`:  Your OpenAI or Gemini API key.
*   `-i, --interval`:  Interval (in seconds) to split transcripts (default: `30`).

### `vsearch` (Semantic Search)

Perform semantic (vector) search on channels with enabled embeddings.

```bash
yt-fts vsearch "How do transformers work?" --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Search within a specific channel.
*   `-v, --video-id`: Search within a specific video.
*   `-l, --limit`:  Limit the number of results (default: `10`).
*   `-e, --export`:  Export results to a CSV file.
*   `--api-key`:  Your OpenAI or Gemini API key.

### `llm` (Chat Bot)

Start an interactive chat session with a model, using semantic search results as context.

```bash
yt-fts llm --channel "3Blue1Brown" "Explain backpropagation."
```

**Options:**

*   `-c, --channel`:  The channel to use for context (required).
*   `--api-key`:  Your OpenAI or Gemini API key.

### `summarize`

Summarize a YouTube video transcript and give time stamped URLS

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Model to use in summary
*   `--api-key`:  Your OpenAI or Gemini API key.

### `config`

Display your current yt-fts configuration settings.

```bash
yt-fts config
```

## How-to Guides

**Export Search Results:**

Export search results from both `search` and `vsearch` to a CSV file:

```bash
yt-fts search "interesting topic" --export
yt-fts vsearch "similar topic" --export
```

**Delete a Channel:**

Remove a channel from your library using:

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:**

Refresh subtitles for a channel using the following command:

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export All of a Channel's Transcript:**

Generate a directory with all transcripts for a specific channel:

```bash
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
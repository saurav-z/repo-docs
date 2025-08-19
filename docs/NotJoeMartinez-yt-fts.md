# yt-fts: Supercharge Your YouTube Research with Full-Text Search

**Quickly search and analyze YouTube transcripts with yt-fts, a powerful command-line tool.** [Check out the original repository](https://github.com/NotJoeMartinez/yt-fts).

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

## Key Features

*   **Full-Text Search:** Search within YouTube transcripts using keywords, phrases, and advanced search syntax.
*   **Semantic Search:**  Leverage OpenAI or Gemini embeddings for more relevant search results, understanding the *meaning* of your queries.
*   **LLM-Powered Chatbot:** Interact with a chatbot that uses semantic search to answer questions about your favorite YouTube channels.
*   **Video Summarization:** Quickly summarize YouTube videos.
*   **Channel and Playlist Downloads:** Easily download subtitles from entire channels or playlists.
*   **Flexible Export Options:** Export search results and transcripts in various formats (CSV, TXT, VTT).
*   **Command-Line Interface:**  Simple and efficient command-line commands for all functionalities.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### `download` - Download Subtitles

Download subtitles for a YouTube channel or playlist. Supports parallel downloads and browser cookie integration.

*   `yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"`
*   `yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"`
*   `yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"`

**Options:**

*   `-p, --playlist`: Download playlist
*   `-l, --language`: Subtitle language (default: `en`)
*   `-j, --jobs`: Parallel download jobs (default: 8)
*   `--cookies-from-browser`: Use cookies from your browser (e.g., `firefox`, `chrome`)

### `diagnose` - Diagnose Download Issues

Tests your connection to YouTube and provides recommendations for fixing common download problems.

*   `yt-fts diagnose`
*   `yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox`

**Options:**

*   `-u, --test-url`: URL to test (default: test video)
*   `--cookies-from-browser`: Browser to extract cookies
*   `-j, --jobs`: Number of parallel jobs to test with (default: 8)

### `list` - List Saved Data

List saved channels, videos, and transcripts.

*   `yt-fts list`
*   `yt-fts list --channel "3Blue1Brown"`
*   `yt-fts list --transcript "dQw4w9WgXcQ"`

**Options:**

*   `-t, --transcript`: Show transcript for a video
*   `-c, --channel`: Show list of videos for a channel
*   `-l, --library`: Show list of channels in library

### `update` - Update Subtitles

Update subtitles for all or specific channels in your library.

*   `yt-fts update`
*   `yt-fts update --channel "3Blue1Brown" --jobs 5`

**Options:**

*   `-c, --channel`: Channel to update
*   `-l, --language`: Subtitle language (default: `en`)
*   `-j, --jobs`: Parallel download jobs (default: 8)
*   `--cookies-from-browser`: Browser to extract cookies

### `delete` - Delete a Channel

Delete a channel and all its associated data. Requires confirmation.

*   `yt-fts delete --channel "3Blue1Brown"`

**Options:**

*   `-c, --channel`: The name or id of the channel to delete (required)

### `export` - Export Transcripts

Export transcripts for a channel in different formats.

*   `yt-fts export --channel "3Blue1Brown" --format txt`
*   `yt-fts export --channel "3Blue1Brown" --format vtt`

**Options:**

*   `-c, --channel`: Channel to export (required)
*   `-f, --format`: Export format: `txt`, `vtt` (default: `txt`)

### `search` - Full Text Search

Search for keywords and phrases within saved transcripts. Supports advanced search syntax.

*   `yt-fts search "[search query]"`
*   `yt-fts search "[search query]" --channel "[channel name or id]"`
*   `yt-fts search "[search query]" --video-id "[video id]"`
*   `yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"`
*   `yt-fts search "[search query]" --export --channel "[channel name or id]"`

**Options:**

*   `-c, --channel`: Channel to search in
*   `-v, --video-id`: Video ID to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export results to CSV

**Advanced Search:**

*   `AND`: `yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"`
*   `OR`: `yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"`
*   `Wildcards`: `yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"`

### `embeddings` - Semantic Search Setup

Generate embeddings for a channel to enable semantic search using OpenAI or Gemini.

*   `yt-fts embeddings --channel "3Blue1Brown"`

**Options:**

*   `-c, --channel`: Channel to generate embeddings for
*   `--api-key`: OpenAI or Gemini API key (or set `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable)
*   `-i, --interval`: Chunk size in seconds (default: 30)

### `vsearch` - Semantic Search

Perform semantic (vector) search within a channel that has embeddings enabled.

*   `yt-fts vsearch "[search query]" --channel "[channel name or id]"`
*   `yt-fts vsearch "[search query]" --video-id "[video id]"`
*   `yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"`
*   `yt-fts vsearch "[search query]" --export --channel "[channel name or id]"`

**Options:**

*   `-c, --channel`: Channel to search in
*   `-v, --video-id`: Video ID to search in
*   `-l, --limit`: Number of results (default: 10)
*   `-e, --export`: Export results to CSV
*   `--api-key`: OpenAI or Gemini API key (or set `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable)

### `llm` - Chatbot

Start an interactive chat session with a model using semantic search context.

*   `yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"`

**Options:**

*   `-c, --channel`: Channel to use (required)
*   `--api-key`: OpenAI or Gemini API key (or set `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable)

### `summarize` - Video Summarization

Get a summary of a YouTube video transcript, including timestamped links.

*   `yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"`
*   `yt-fts summarize "9-Jl0dxWQs8"`
*   `yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"`

**Options:**

*   `--model, -m`: Model to use for summarization
*   `--api-key`: OpenAI or Gemini API key (or set `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable)

### `config` - View Configuration

Show configuration settings, including database and Chroma paths.

*   `yt-fts config`

## How To

**Export search results:**  Use the `--export` flag with `search` and `vsearch` to save results to a CSV file in the current directory.

**Delete a channel:** Use the `delete` command: `yt-fts delete --channel "3Blue1Brown"`

**Update a channel:** Use the `update` command for full-text search updates: `yt-fts update --channel "3Blue1Brown"`

**Export all of a channel's transcript:** Use the `export` command to create a directory with the channel ID.
```
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
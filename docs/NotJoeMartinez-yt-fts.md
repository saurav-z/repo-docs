# yt-fts: Unleash the Power of YouTube Search - Full Text and Semantic Search

**Effortlessly search and explore YouTube channels with `yt-fts`, a command-line tool that lets you search video transcripts, even using semantic search!** [View the original repository on GitHub](https://github.com/NotJoeMartinez/yt-fts).

<p align="center">
  <img src="https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14" alt="yt-fts demo" />
</p>

## Key Features

*   **Full-Text Search:** Quickly find videos containing specific keywords or phrases within a channel's transcripts.
*   **Semantic Search:** Leverage OpenAI or Gemini embeddings for more intelligent search results based on meaning and context.
*   **Video Summarization:** Generate concise summaries of YouTube video transcripts.
*   **LLM Chat Bot:** Start interactive chat sessions with a model, using the semantic search results of your initial prompt as the context to answer questions.
*   **Playlist and Channel Support:** Download and search transcripts from both channels and playlists.
*   **Flexible Export Options:** Export search results to CSV and transcripts to TXT or VTT.
*   **Easy Installation:** Install with a simple `pip install yt-fts`.
*   **Advanced Search Syntax:** Supports SQLite's enhanced query syntax for powerful and flexible search capabilities.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### `download`: Download Subtitles

Download subtitles for a channel or playlist. Use `--cookies-from-browser` to handle potential sign-in requirements.

```bash
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
```

**Options:**

*   `-p, --playlist`: Download from a playlist.
*   `-l, --language`: Subtitle language (default: `en`).
*   `-j, --jobs`: Parallel download jobs (default: 8, recommended: 4-16).
*   `--cookies-from-browser`: Use cookies from a browser (chrome, firefox, etc.).

### `diagnose`: Diagnose Download Issues

Test your connection and get recommendations for fixing download problems.

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: Test URL (default: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`).
*   `--cookies-from-browser`: Browser to extract cookies from.
*   `-j, --jobs`: Number of parallel download jobs to test with (default: 8).

### `list`: List Saved Content

List your saved channels, videos, and transcripts.  "(ss)" indicates semantic search is enabled for a channel.

```bash
yt-fts list
yt-fts list --channel "3Blue1Brown"
yt-fts list --transcript "dQw4w9WgXcQ"
```

**Options:**

*   `-t, --transcript`: Show the transcript for a video.
*   `-c, --channel`: Show videos for a specific channel.
*   `-l, --library`: Show the list of channels in your library.

### `update`: Update Subtitles

Update subtitles for all or specific channels.

```bash
yt-fts update
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: Channel name or ID to update.
*   `-l, --language`: Subtitle language (default: `en`).
*   `-j, --jobs`: Number of parallel download jobs (default: 8).
*   `--cookies-from-browser`: Browser to extract cookies from.

### `delete`: Delete Channel Data

Delete a channel and its associated data.  Requires confirmation.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel name or ID to delete (required).

### `export`: Export Transcripts

Export transcripts for a channel.

```bash
yt-fts export --channel "3Blue1Brown" --format txt
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: Channel name or ID to export (required).
*   `-f, --format`: Export format: `txt` or `vtt` (default: `txt`).

### `search`: Full Text Search

Search for keywords in saved transcripts. Supports [SQLite's Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries).

```bash
yt-fts search "[search query]"
yt-fts search "[search query]" --channel "[channel name or id]"
yt-fts search "[search query]" --video-id "[video id]"
yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"
yt-fts search "[search query]" --export --channel "[channel name or id]" 

# Advanced Search Examples
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
```

**Options:**

*   `-c, --channel`: Search within a specific channel.
*   `-v, --video-id`: Search within a specific video.
*   `-l, --limit`: Limit results (default: 10).
*   `-e, --export`: Export results to CSV.

### `embeddings`: Enable Semantic Search

Generate OpenAI or Gemini embeddings for a channel to enable semantic search functionality. Requires an API key (`OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable, or use `--api-key`).

```bash
yt-fts embeddings --channel "3Blue1Brown"
yt-fts embeddings --interval 60 --channel "3Blue1Brown" 
```

**Options:**

*   `-c, --channel`: The name or id of the channel to generate embeddings for
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)
*   `-i, --interval`: Interval in seconds to split the transcripts into chunks (default: 30)

### `vsearch`: Semantic Search

Perform semantic (vector) search on channels with enabled embeddings.  Similar options as `search`, but results are ranked by similarity.

```bash
yt-fts vsearch "[search query]" --channel "[channel name or id]"
yt-fts vsearch "[search query]" --video-id "[video id]"
yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"
yt-fts vsearch "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Search within a specific channel.
*   `-v, --video-id`: Search within a specific video.
*   `-l, --limit`: Limit results (default: 10).
*   `-e, --export`: Export results to CSV.
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `llm`: LLM Chat Bot

Engage in an interactive chat session using the semantic search results as context. The channel must have semantic search enabled.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: Channel name or ID (required).
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `summarize`: Summarize Video Transcripts

Summarize YouTube video transcripts using a LLM.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
yt-fts summarize "9-Jl0dxWQs8"
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: LLM model to use.
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `config`: Show Configuration

Display current configuration settings, including database and Chroma paths.

```bash
yt-fts config
```

## How To

### Export Search Results

Export results from `search` or `vsearch` to a CSV file.

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

### Delete a Channel

Use the `delete` command to remove a channel and its data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

### Update a Channel

Update subtitles for a channel. Currently only updates full text search and does not update semantic search embeddings.

```bash
yt-fts update --channel "3Blue1Brown"
```

### Export a Channel's Transcript

Export the entire transcript of a channel.

```bash
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
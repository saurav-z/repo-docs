# yt-fts: Unleash the Power of YouTube Search 🚀

Tired of endless scrolling to find that perfect YouTube moment? **yt-fts** is a command-line tool that lets you perform full-text and semantic searches within YouTube video transcripts, giving you instant access to the information you need. Find the original repo [here](https://github.com/NotJoeMartinez/yt-fts/).

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

## Key Features

*   **Full-Text Search:** Quickly search across entire YouTube channels for specific keywords or phrases.
*   **Semantic Search:** Leverage the power of OpenAI or Gemini embeddings for more nuanced searches, understanding the meaning behind your queries.
*   **LLM Chat Bot:** Engage in interactive conversations with a chat bot that uses semantic search results as context.
*   **Video Summarization:** Get concise summaries of YouTube videos, with time-stamped links to relevant sections.
*   **Flexible Data Export:** Export search results and transcripts in various formats (CSV, TXT, VTT).
*   **Channel Management:** Download, update, and delete channels easily.
*   **Robust Installation:** Simple installation via `pip`.

## Installation

```bash
pip install yt-fts
```

## Commands

### `download` - Download Subtitles

Download subtitles for a channel or playlist and store them in a searchable database. Use cookies from your browser if you encounter sign-in requests.  Parallelize downloads for faster processing.

```bash
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
```

**Options:**

*   `-p, --playlist`: Download from a playlist
*   `-l, --language`: Subtitle language (default: `en`)
*   `-j, --jobs`: Number of parallel jobs (default: 8, recommended: 4-16)
*   `--cookies-from-browser`: Browser to extract cookies from

### `diagnose` - Diagnose Download Issues

Troubleshoot 403 errors and other download problems to ensure smooth operation.

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: URL to test (default: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)
*   `--cookies-from-browser`: Browser to extract cookies from
*   `-j, --jobs`: Number of parallel download jobs to test with (default: 8)

### `list` - List Saved Content

View saved channels, videos, and transcripts.

```bash
yt-fts list
yt-fts list --channel "3Blue1Brown"
yt-fts list --transcript "dQw4w9WgXcQ"
```

**Options:**

*   `-t, --transcript`: Show transcript for a video
*   `-c, --channel`: List videos for a specific channel
*   `-l, --library`: List channels in the library

### `update` - Update Subtitles

Refresh subtitles for all channels or specific ones.

```bash
yt-fts update
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: Channel to update
*   `-l, --language`: Subtitle language (default: `en`)
*   `-j, --jobs`: Number of parallel jobs (default: 8)
*   `--cookies-from-browser`: Browser to extract cookies from

### `delete` - Delete Channel Data

Remove a channel and all its associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel to delete (required)

### `export` - Export Transcripts

Export transcripts to various formats.

```bash
yt-fts export --channel "3Blue1Brown" --format txt
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: Channel to export (required)
*   `-f, --format`: Output format (`txt`, `vtt`, default: `txt`)

### `search` - Full-Text Search

Search for keywords or phrases within downloaded transcripts. Supports advanced search syntax.

```bash
yt-fts search "[search query]"
yt-fts search "[search query]" --channel "[channel name or id]"
yt-fts search "[search query]" --video-id "[video id]"
yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"
yt-fts search "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Search within a channel
*   `-v, --video-id`: Search within a video
*   `-l, --limit`: Limit search results (default: 10)
*   `-e, --export`: Export results to CSV

**Advanced Search Examples:**

```bash
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
```

### `embeddings` - Semantic Search Setup

Generate embeddings for semantic search using OpenAI or Gemini. Requires API key setup.

```bash
# export OPENAI_API_KEY="[yourOpenAIKey]" OR export GEMINI_API_KEY="[yourGeminiKey]"
yt-fts embeddings --channel "3Blue1Brown"
yt-fts embeddings --interval 60 --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel to generate embeddings for
*   `--api-key`: API key (reads from environment variables if not provided)
*   `-i, --interval`: Transcript chunk interval in seconds (default: 30)

### `vsearch` - Semantic Search

Perform semantic (vector-based) searches on channels with embeddings enabled.

```bash
yt-fts vsearch "[search query]" --channel "[channel name or id]"
yt-fts vsearch "[search query]" --video-id "[video id]"
yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"
yt-fts vsearch "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Search within a channel
*   `-v, --video-id`: Search within a video
*   `-l, --limit`: Limit search results (default: 10)
*   `-e, --export`: Export results to CSV
*   `--api-key`: API key (reads from environment variables if not provided)

### `llm` - Interactive Chat Bot

Engage in an interactive chat session with a model, utilizing semantic search results for context.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: Channel to use (required)
*   `--api-key`: API key (reads from environment variables if not provided)

### `summarize` - Video Summarization

Get concise summaries of YouTube videos.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
yt-fts summarize "9-Jl0dxWQs8"
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Model to use for summarization
*   `--api-key`: API key (reads from environment variables if not provided)

### `config` - View Configuration

Display current configuration settings, including database and chroma paths.

```bash
yt-fts config
```

## How To: Step-by-Step Guides

**1. Export Search Results:**
Export your search results for further analysis and use.

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**2. Delete a Channel:**
Remove a channel and its associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**3. Update a Channel:**
Refresh a channel's data.

```bash
yt-fts update --channel "3Blue1Brown"
```

**4. Export a Channel's Transcript:**
Save a channel's transcript to your local machine.

```bash
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
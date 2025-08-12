# yt-fts: Unleash the Power of YouTube Video Search

**yt-fts** empowers you to perform full-text and semantic searches within YouTube channels, turning video transcripts into a searchable knowledge base. [Explore yt-fts on GitHub](https://github.com/NotJoeMartinez/yt-fts).

[![yt-fts in action](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)

## Key Features

*   **Full-Text Search:** Quickly find specific keywords or phrases within YouTube video transcripts using SQLite's powerful [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries).
*   **Semantic Search (with OpenAI/Gemini):** Leverage the power of embeddings to search based on meaning and context. Requires OpenAI or Gemini API key.
*   **LLM-Powered Chat Bot:** Interact with a chatbot that understands the content of a YouTube channel using the `llm` command.
*   **Video Summarization:** Generate concise summaries of YouTube videos with time-stamped links.
*   **Comprehensive Channel Management:** Download, update, list, delete, and export transcripts for easy management of your YouTube video library.
*   **Flexible Download Options:** Download subtitles from playlists or entire channels, with options for language selection and parallel downloads.
*   **Diagnostic Tools:** Troubleshoot common download issues with the `diagnose` command.
*   **Configurable Settings:** Easily view and manage your application configuration.

## Installation

Install the `yt-fts` package using pip:

```bash
pip install yt-fts
```

## Commands

### `download` - Download Subtitles

*   Downloads subtitles from a YouTube channel or playlist.
*   Use `--cookies-from-browser` to resolve sign-in errors.
*   Use `--playlist` to download a playlist.
*   Use `update` command to add more videos to the database.

```bash
# Download channel
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"

# Download playlist
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
```

**Options:**

*   `-p, --playlist`: Download playlist
*   `-l, --language`: Subtitle language (default: `en`)
*   `-j, --jobs`: Parallel download jobs (default: 8, recommended: 4-16)
*   `--cookies-from-browser`: Browser to extract cookies from

### `diagnose` - Diagnose Download Issues

*   Tests your connection to YouTube and provides troubleshooting recommendations.

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: URL to test (default: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)
*   `--cookies-from-browser`: Browser to extract cookies from
*   `-j, --jobs`: Parallel download jobs (default: 8)

### `list` - List Saved Channels, Videos, and Transcripts

*   Lists the channels, videos, and transcripts saved in your database.
*   The `(ss)` next to the channel name indicates semantic search is enabled.

```bash
# List all channels
yt-fts list

# List videos for a specific channel
yt-fts list --channel "3Blue1Brown"

# Show transcript for a specific video
yt-fts list --transcript "dQw4w9WgXcQ"

# Show library (same as default)
yt-fts list --library
```

**Options:**

*   `-t, --transcript`: Show transcript for a video
*   `-c, --channel`: Show list of videos for a channel
*   `-l, --library`: Show list of channels in library

### `update` - Update Subtitles

*   Updates subtitles for all or specific channels.

```bash
# Update all channels
yt-fts update

# Update specific channel
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: Channel to update
*   `-l, --language`: Subtitle language (default: `en`)
*   `-j, --jobs`: Parallel download jobs (default: 8)
*   `--cookies-from-browser`: Browser to extract cookies from

### `delete` - Delete a Channel

*   Deletes a channel and its associated data from your database.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel to delete (required)

### `export` - Export Transcripts

*   Exports video transcripts for a channel in txt or vtt format.

```bash
# Export to txt format (default)
yt-fts export --channel "3Blue1Brown" --format txt

# Export to vtt format
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: Channel to export (required)
*   `-f, --format`: Export format: txt, vtt (default: `txt`)

### `search` - Full Text Search

*   Searches for a string within the transcripts of saved channels.

```bash
# search in all channels
yt-fts search "[search query]" 

# search in channel 
yt-fts search "[search query]" --channel "[channel name or id]" 

# search in specific video
yt-fts search "[search query]" --video-id "[video id]"

# limit results 
yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"

# export results to csv
yt-fts search "[search query]" --export --channel "[channel name or id]" 
```

**Options:**

*   `-c, --channel`: Channel to search in
*   `-v, --video-id`: Video ID to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export search results to a CSV file

**Advanced Search Syntax:**

Use SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for more advanced searches:

```bash
# AND search
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show" 

# OR SEARCH 
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show" 

# wild cards
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show" 
```

### `embeddings` - Semantic Search Setup

*   Enables semantic search for a channel using OpenAI or Gemini embeddings. Requires an API key.

```bash
# make sure API key is set
# export OPENAI_API_KEY="[yourOpenAIKey]"
# or
# export GEMINI_API_KEY="[yourGeminiKey]"

yt-fts embeddings --channel "3Blue1Brown"

# specify time interval in seconds to split text by default is 30 
# the larger the interval the more accurate the llm response  
# but semantic search will have more text for you to read. 
yt-fts embeddings --interval 60 --channel "3Blue1Brown" 
```

**Options:**

*   `-c, --channel`: Channel to generate embeddings for
*   `--api-key`: API key (if not provided, uses `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable)
*   `-i, --interval`: Transcript chunk interval in seconds (default: 30)

### `vsearch` - Semantic Search

*   Performs vector-based (semantic) searches within a channel after embeddings have been generated.

```bash
# search by channel name
yt-fts vsearch "[search query]" --channel "[channel name or id]"

# search in specific video
yt-fts vsearch "[search query]" --video-id "[video id]"

# limit results 
yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"

# export results to csv
yt-fts vsearch "[search query]" --export --channel "[channel name or id]" 
```

**Options:**

*   `-c, --channel`: Channel to search in
*   `-v, --video-id`: Video ID to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export search results to a CSV file
*   `--api-key`: API key (if not provided, uses `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable)

### `llm` - LLM Chat Bot

*   Starts an interactive chat session using the semantic search results of your initial prompt as the context for answering questions. The channel must have semantic search enabled.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: Channel to use (required)
*   `--api-key`: API key (if not provided, uses `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable)

### `summarize` - Video Summarization

*   Summarizes a YouTube video transcript and generates time-stamped links.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
# or
yt-fts summarize "9-Jl0dxWQs8"

# Use different model
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: LLM model to use
*   `--api-key`: API key (if not provided, uses `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable)

### `config` - Show Configuration

*   Displays the current configuration settings, including database and chroma paths.

```bash
yt-fts config
```

## How To

**Export Search Results:**

Export your `search` and `vsearch` results to a CSV file for easy data analysis:

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a Channel:**

Remove a channel and all associated data with the `delete` command:

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:**

Refresh the transcript data for a channel:

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export a Channel's Transcript:**

Export a channel's transcripts in either .txt or .vtt format:

```bash
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
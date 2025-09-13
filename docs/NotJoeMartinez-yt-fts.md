# yt-fts: Your Command-Line YouTube Transcript Search Engine

**Effortlessly search YouTube video transcripts with `yt-fts`, a powerful command-line tool, enabling you to find specific keywords and phrases within your favorite channels and videos.**  [View the original repo](https://github.com/NotJoeMartinez/yt-fts)

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)

**Key Features:**

*   **Full-Text Search:** Quickly find videos containing specific keywords or phrases within YouTube channel transcripts using the `search` command.
*   **Semantic Search:** Utilize OpenAI or Gemini API embeddings for more nuanced and context-aware searches with the `vsearch` command.
*   **LLM Chat Bot:** Engage in interactive conversations with an LLM, using semantic search results as context using the `llm` command.
*   **Video Summarization:** Generate concise summaries of YouTube video transcripts with timestamped links, using the `summarize` command.
*   **Channel Management:** Download, update, list, and delete channels and their transcripts.
*   **Advanced Search Syntax:** Leverage SQLite's enhanced query syntax for powerful and flexible searches.
*   **Export Functionality:** Export search results to CSV or channel transcripts in txt/vtt formats.
*   **Robust Installation & Diagnostics:** Easy installation via pip and diagnostic tools to troubleshoot download issues.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### `download` - Download Subtitles
Download subtitles for a channel or playlist.

```bash
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
```

**Options:**
*   `-p, --playlist`: Download all videos from a playlist
*   `-l, --language`: Language of the subtitles to download (default: en)
*   `-j, --jobs`: Number of parallel download jobs (default: 8, recommended: 4-16)
*   `--cookies-from-browser`: Browser to extract cookies from (chrome, firefox, etc.)

### `diagnose` - Diagnose Download Issues
Diagnose and troubleshoot 403 errors and other download problems.

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**
*   `-u, --test-url`: URL to test with (default: https://www.youtube.com/watch?v=dQw4w9WgXcQ)
*   `--cookies-from-browser`: Browser to extract cookies from
*   `-j, --jobs`: Number of parallel download jobs to test with (default: 8)

### `list` - List Channels, Videos, and Transcripts
List saved channels, videos, and transcripts.

```bash
yt-fts list
yt-fts list --channel "3Blue1Brown"
yt-fts list --transcript "dQw4w9WgXcQ"
yt-fts list --library
```

**Options:**
*   `-t, --transcript`: Show transcript for a video
*   `-c, --channel`: Show list of videos for a channel
*   `-l, --library`: Show list of channels in library

### `update` - Update Subtitles
Update subtitles for all channels in the library or a specific channel.

```bash
yt-fts update
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**
*   `-c, --channel`: The name or id of the channel to update
*   `-l, --language`: Language of the subtitles to download (default: en)
*   `-j, --jobs`: Number of parallel download jobs (default: 8)
*   `--cookies-from-browser`: Browser to extract cookies from

### `delete` - Delete a Channel
Delete a channel and all its data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**
*   `-c, --channel`: The name or id of the channel to delete (required)

### `export` - Export Transcripts
Export transcripts for a channel to TXT or VTT format.

```bash
yt-fts export --channel "3Blue1Brown" --format txt
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**
*   `-c, --channel`: The name or id of the channel to export transcripts for (required)
*   `-f, --format`: The format to export transcripts to. Supported formats: txt, vtt (default: txt)

### `search` - Full Text Search
Full text search for a string in saved channels.

```bash
yt-fts search "[search query]"
yt-fts search "[search query]" --channel "[channel name or id]"
yt-fts search "[search query]" --video-id "[video id]"
yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"
yt-fts search "[search query]" --export --channel "[channel name or id]"
```

**Options:**
*   `-c, --channel`: The name or id of the channel to search in
*   `-v, --video-id`: The id of the video to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export search results to a CSV file

**Advanced Search Syntax:**

Utilize SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for advanced searches, including prefix queries and boolean operators.

```bash
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
```

### `embeddings` - Semantic Search Setup
Enable semantic search for a channel using OpenAI or Gemini API.
Requires an API key set in the environment variable `OPENAI_API_KEY` or `GEMINI_API_KEY` or via the `--api-key` flag.

```bash
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

### `vsearch` - Semantic Search
Vector search using semantic embeddings.  This requires that you enable semantic search for a channel with `embeddings`.

```bash
yt-fts vsearch "[search query]" --channel "[channel name or id]"
yt-fts vsearch "[search query]" --video-id "[video id]"
yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"
yt-fts vsearch "[search query]" --export --channel "[channel name or id]"
```

**Options:**
*   `-c, --channel`: The name or id of the channel to search in
*   `-v, --video-id`: The id of the video to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export search results to a CSV file
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `llm` - LLM Chat Bot
Starts an interactive chat session using semantic search context.
The channel must have semantic search enabled.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**
*   `-c, --channel`: The name or id of the channel to use (required)
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `summarize` - Video Summarization
Summarizes a YouTube video transcript.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
yt-fts summarize "9-Jl0dxWQs8"
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**
*   `--model, -m`: Model to use in summary
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

```
In this video, 3Blue1Brown explores how large language models (LLMs) like GPT-3 
might store facts within their vast...                                                         

 1 Introduction to Fact Storage in LLMs:                                                                                     
    • The video starts by questioning how LLMs store specific facts and                                                      
      introduces the idea that these facts might be stored in a particular part of the                                       
      network known as multi-layer perceptrons (MLPs).                                                                       
    • 0:00                                                                                                                   
 2 Overview of Transformers and MLPs:                                                                                        
    • Provides a refresher on transformers and explains that the video will focus                                            
```

### `config` - Show Configuration
Show config settings including database and chroma paths.

```bash
yt-fts config
```

## How To

**Export search results:**

Export search results to a CSV file.
```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a channel:**

Delete a channel with the `delete` command.
```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a channel:**
The update command currently only works for full text search and will not update the 
semantic search embeddings. 

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export all of a channel's transcript:**

Export all of a channel's transcript.
```bash
# Export to vtt
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
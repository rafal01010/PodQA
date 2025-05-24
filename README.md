# PodQA - RAG Application Setup Guide

A Retrieval-Augmented Generation (RAG) application for processing YouTube channel content, transcribing audio, and creating a searchable knowledge base.

## Installation & Setup

### 1. Environment Setup

First, create and activate a conda environment:

```bash
# Create conda environment
conda create -n podqa python=3.10

# Activate the environment
conda activate podqa
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. YouTube Video Download

Navigate to the YouTube download directory and download videos as MP3 files:

```bash
cd yt-download
python yt-dl_channel.py
```

**Input Required**: YouTube channel videos tab URL
- Example: `https://www.youtube.com/@TrashTaste/videos`
- The script will download all videos from the channel as MP3 audio files

### 4. Audio Transcription

Navigate to the transcription directory and transcribe the downloaded MP3 files:

```bash
cd ../transcription
python transcribe.py
```

This will process all MP3 files from the `yt-download` folder and generate transcriptions.

### 5. Create Vector Embeddings

From the main PodQA directory, create embeddings and insert them into the vector database:

```bash
# Return to main directory
cd ..

# Create and insert embeddings
python -m srt_embeddings.insert_embeddings
```

### 6. Create Database Indexes

Finally, create indexes for the vector database to enable efficient searching:

```bash
python -m srt_embeddings.create_index
```

## Project Structure

```
PodQA/
├── backend/
│   ├── app.py
│   └── [backend API files]
├── frontend/
│   └── rag-chatbot-frontend/
│       ├── package.json
│       └── [React frontend files]
├── yt-download/
│   ├── yt-dl_channel.py
│   ├── [downloaded MP3 files]
│   └── [transcription files]
├── transcription/
│   └── transcribe.py
├── srt_embeddings/
│   ├── insert_embeddings.py
│   ├── create_index.py
│   └── transcripts_lancedb/    # Vector database files
├── requirements.txt
└── README.md
```

## Running the Application

After completing the setup, you can start the application:

### 1. Start the Backend Server

From the main PodQA directory, run:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

The backend API will be available at `http://localhost:8000`

### 2. Start the Frontend (New Terminal Tab)

Open a new terminal tab/window and navigate to the frontend directory:

```bash
cd frontend/rag-chatbot-frontend
npm start
```

The frontend application will typically start at `http://localhost:3000`

## Usage Workflow

1. **Download**: Use `yt-dl_channel.py` to download YouTube channel content as MP3
2. **Transcribe**: Use `transcribe.py` to convert audio to text
3. **Embed**: Use `insert_embeddings` to create vector representations
4. **Index**: Use `create_index` to optimize database queries
5. **Start Backend**: Run the FastAPI backend server with uvicorn
6. **Start Frontend**: Launch the React frontend application
7. **Query**: Your RAG application is now ready to answer questions about the content

## Optional Configuration

### Gemini LLM Integration

To use the Gemini model as your LLM, configure the following environment variables:

```bash
# Set Gemini API Key
export GEMINI_API_KEY="your_gemini_api_key_here"

# Set Gemini Model Name
export GEMINI_MODEL_NAME="your_preferred_gemini_model"
```

# Evaluation

## Testing Methodology

To evaluate the retrieval performance of the RAG application, I conducted a test using content from the Trash Taste podcast clips channel.

### Test Data Creation

1. **Source Selection**: I analyzed the most popular clips from the Trash Taste podcast clips channel, sorted by popularity.

2. **Question Generation**: From the top clips, I created 50 test questions using only the video titles (without accessing video content). This approach simulates real-world scenarios where users ask questions based on topics they remember or have heard about.

   **Examples of our question transformation process:**
   - Video title: *"The Filthy Frank era of YouTube was Something Else"*
   - Question: *"How does Trash Taste describe the Filthy Frank era?"*
   
   - Video title: *"The Great Pizza Debate of Trash Taste"*
   - Question: *"What pizza does the Trash Taste members like?"*

3. **Evaluation Criteria**: For each question, I sent the query to the chatbot and examined whether the correct source episode appeared in the retrieved sources. The chat was cleared between each individual question to ensure no context carryover affected the results.

## Results

### Hybrid Retrieval
- **Episodes Retrieved**: 34 out of 50 (68% success rate)
- **Highest Scored Matches**: 20 out of 34 retrieved episodes (58.8% precision)

### ModernColBERT Retrieval
- **Episodes Retrieved**: 32 out of 50 (64% success rate)
- **Highest Scored Matches**: 15 out of 32 retrieved episodes (46.9% precision)

While the success rates of 68% and 64% may appear modest, it should be pointed out that these results don't indicate a failure of the retrieval system or stored embeddings. Instead, they reflect the inherent challenge of our specific test methodology.
The test design deliberately created a challenging scenario by generating questions solely from video clip titles without access to the actual content. This approach often leads to situations where the retrieval system correctly identifies semantically relevant content, but not necessarily the exact episode the question was derived from.

**Illustrative Example:**
Question 49: *"What are their opinions on burgers?"* failed retrieval for both systems, but this doesn't represent a system failure. Since Trash Taste has discussed burgers across multiple episodes, the retrievers surfaced episodes with more extensive burger discussions rather than the specific clip-derived episode.


## Detailed Results

**For complete test data and individual question results, see [`retrieval_eval.txt`](retrieval_eval.txt)**

This file contains:
- All 50 test questions used in the evaluation
- Individual retrieval results for each question
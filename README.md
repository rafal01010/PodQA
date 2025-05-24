# Evaluation

## Testing Methodology

To evaluate the retrieval performance of the RAG application, we conducted a test using content from the Trash Taste podcast clips channel.

### Test Data Creation

1. **Source Selection**: We analyzed the most popular clips from the Trash Taste podcast clips channel, sorted by popularity.

2. **Question Generation**: From the top clips, we created 50 test questions using only the video titles (without accessing video content). This approach simulates real-world scenarios where users ask questions based on topics they remember or have heard about.

   **Examples of our question transformation process:**
   - Video title: *"The Filthy Frank era of YouTube was Something Else"*
   - Question: *"How does Trash Taste describe the Filthy Frank era?"*
   
   - Video title: *"The Great Pizza Debate of Trash Taste"*
   - Question: *"What pizza does the Trash Taste members like?"*

3. **Evaluation Criteria**: For each question, we sent the query to the chatbot and examined whether the correct source episode appeared in the retrieved sources. The chat was cleared between each individual question to ensure no context carryover affected the results.

## Results

### Hybrid Retrieval
- **Episodes Retrieved**: 34 out of 50 (68% success rate)
- **Highest Scored Matches**: 20 out of 34 retrieved episodes (58.8% precision)

### ModernColBERT Retrieval
- **Episodes Retrieved**: 32 out of 50 (64% success rate)
- **Highest Scored Matches**: 15 out of 32 retrieved episodes (46.9% precision)

## Detailed Results

**For complete test data and individual question results, see [`retrieval_eval.txt`](retrieval_eval.txt)**

This file contains:
- All 50 test questions used in the evaluation
- Individual retrieval results for each question
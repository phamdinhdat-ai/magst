# Guest Request Queue System

This document explains how the Guest Request Queue System improves performance, reliability, and user experience in the GeneStory chatbot system.

## Overview

The Guest Request Queue System is designed to handle multiple concurrent requests efficiently, prevent system overload, and provide a better user experience during high-demand periods. It's based on the same principles as the main API server's request queue but specifically optimized for guest workflow interactions.

## Key Features

### 1. Request Queuing

- **Configurable concurrency**: Limit the number of guest requests processed simultaneously
- **Fair scheduling**: First-come, first-served request handling
- **Queue size limits**: Prevent memory exhaustion during traffic spikes
- **Timeout handling**: Automatically handle requests that take too long

### 2. Improved Streaming Response

- **Real-time status updates**: Users know where they are in the queue
- **Event-based streaming**: SSE (Server-Sent Events) format with proper headers
- **Nginx buffering prevention**: X-Accel-Buffering header prevents proxy buffering
- **Graceful error handling**: Provides informative error messages

### 3. Performance Optimizations

- **Background processing**: Database updates happen asynchronously
- **Intelligent caching**: Fast responses for common questions
- **Timeouts**: Prevents resource hogging by runaway requests
- **Request history**: Limited tracking of recent requests for diagnostics

## Configuration

The queue system can be configured using environment variables:

- `GUEST_MAX_CONCURRENT_REQUESTS`: Maximum number of simultaneous requests (default: 3)
- `GUEST_MAX_QUEUE_SIZE`: Maximum number of requests in queue (default: 50)
- `GUEST_REQUEST_TIMEOUT_SEC`: Maximum processing time per request (default: 60s)

## API Endpoints

### Chat Endpoints

- `POST /api/v1/guest/chat`: Streaming chat interface using the queue system
- `POST /api/v1/guest/chat/simple`: Non-streaming interface using the queue system

### Queue Management Endpoints

- `GET /api/v1/guest/queue/status`: View current queue statistics
- `GET /api/v1/guest/queue/request/{request_id}`: Check status of a specific request
- `GET /api/v1/guest/health`: Health check including queue status

## Frontend Integration

The frontend needs to handle the SSE streaming responses correctly. Here's a basic example:

```javascript
async function streamChatResponse(prompt) {
  try {
    const response = await fetch('/api/v1/guest/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: prompt }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    
    // Process the stream
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      
      // Process complete SSE messages in buffer
      while (buffer.includes('\n\n')) {
        const newlineIndex = buffer.indexOf('\n\n');
        const message = buffer.slice(0, newlineIndex).trim();
        buffer = buffer.slice(newlineIndex + 2);
        
        if (message && message.startsWith('data:')) {
          try {
            // Extract the JSON part
            const jsonStr = message.slice(5).trim();
            const data = JSON.parse(jsonStr);
            
            // Handle different event types
            switch(data.event) {
              case 'queued':
                updateUI(`Queued at position ${data.data.queue_position}...`);
                break;
              case 'processing':
                updateUI('Processing your request...');
                break;
              case 'answer_chunk':
                updateUI(data.data, true); // append=true
                break;
              case 'error':
                showError(data.data.error);
                break;
              case 'workflow_complete':
                showSuggestedQuestions(data.data.suggested_questions);
                break;
            }
          } catch (e) {
            console.error('Error parsing SSE message:', e);
          }
        }
      }
    }
  } catch (error) {
    console.error('Error with chat stream:', error);
    showError('Connection error occurred');
  }
}
```

## Error Handling

The queue system handles several types of errors:

1. **Queue full**: Returns 429 status with a message to try again later
2. **Request timeout**: Returns a timeout event and fallback response
3. **Workflow errors**: Forwards error messages from the underlying workflow
4. **Service unavailable**: Returns 503 if the workflow service isn't available

## Performance Monitoring

Use the `/api/v1/guest/queue/status` endpoint to monitor:

- Current queue size
- Active requests
- Total, completed, and failed requests
- Request timeouts

## Implementation Details

- The queue system is implemented using Python's `asyncio.Queue`
- Each request is assigned a unique ID for tracking
- Worker tasks process requests from the queue
- Timeouts are implemented with `asyncio.create_task` and `asyncio.sleep`
- Request history is limited to 100 recent requests to prevent memory leaks

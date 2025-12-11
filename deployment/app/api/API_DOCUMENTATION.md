# GeneStory Workflow API Documentation

A comprehensive API for managing guest, customer, and employee interactions with the GeneStory workflow system, featuring AI-powered workflows, document processing, and queue management.

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Authentication](#authentication)
- [API Endpoints](#api-endpoints)
  - [Health & System Monitoring](#health--system-monitoring)
  - [Guest API](#guest-api)
  - [Customer API](#customer-api)
  - [Employee API](#employee-api)
  - [Document API](#document-api)
  - [RAG (Retrieval-Augmented Generation) API](#rag-retrieval-augmented-generation-api)
  - [Product API](#product-api)
  - [Queue Management APIs](#queue-management-apis)
  - [Feedback API](#feedback-api)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Queue System](#queue-system)

## Overview

The GeneStory Workflow API provides a robust platform for:
- **Multi-user support**: Guest, customer, and employee workflows
- **AI-powered interactions**: Intelligent conversation workflows with streaming responses
- **Document processing**: Upload, processing, and RAG-based document querying
- **Queue management**: Fair resource allocation and request processing
- **System monitoring**: Health checks, metrics, and performance monitoring
- **Feedback system**: Universal feedback collection and analytics

## Base URL

```
http://localhost:8000
```

## Authentication

The API uses JWT-based authentication with role-based access control:

- **Guests**: No authentication required for basic operations
- **Customers**: JWT tokens with customer-specific permissions
- **Employees**: JWT tokens with role-based permissions (admin, manager, employee)

### Authentication Headers

```http
Authorization: Bearer <jwt_token>
```

## API Endpoints

### Health & System Monitoring

#### Get Server Health
```http
GET /health
```

Returns overall server health status, available features, and system information.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2024-01-01T12:00:00Z",
  "available_features": ["guest_workflow", "customer_workflow", "employee_workflow"],
  "system_info": {
    "hostname": "server-name",
    "platform": "Linux",
    "python_version": "3.11.0"
  }
}
```

#### System Health (Detailed)
```http
GET /api/system/health?detailed=true
```

Comprehensive system health including database, resources, and queue status.

#### Database Health
```http
GET /api/system/database/health
```

Database connection status and performance metrics.

#### Load Balancer Status
```http
GET /api/system/load-balancer/status
```

Load balancer configuration and health status.

#### Queue Status
```http
GET /api/system/queue/status
```

Queue management system status and statistics.

---

### Guest API

Base path: `/api/v1/guest`

#### Guest Session Management

##### Create Guest Session
```http
POST /api/v1/guest/session
```

**Request Body:**
```json
{
  "session_id": "optional-session-id",
  "metadata": {
    "user_agent": "string",
    "ip_address": "string"
  }
}
```

##### Get Guest Session
```http
GET /api/v1/guest/session/{session_id}
```

#### Guest Workflow Interactions

##### Submit Workflow Query (Streaming)
```http
POST /api/v1/guest/workflow/stream
```

**Request Body:**
```json
{
  "query": "What are the latest developments in gene therapy?",
  "session_id": "uuid-session-id",
  "chat_history": [
    {
      "role": "user",
      "content": "Previous message"
    }
  ],
  "metadata": {
    "priority": false
  }
}
```

**Response:** Server-Sent Events (SSE) stream with real-time workflow progress.

##### Submit Workflow Query (Non-streaming)
```http
POST /api/v1/guest/workflow
```

Similar request body as streaming version, returns complete response.

#### Guest Analytics

##### Get Guest Analytics
```http
GET /api/v1/guest/analytics?session_id={session_id}&days=30
```

Returns interaction statistics and usage patterns.

---

### Customer API

Base path: `/api/v1/customer`

#### Customer Authentication

##### Register Customer
```http
POST /api/v1/customer/register
```

**Request Body:**
```json
{
  "username": "customer_username",
  "email": "customer@example.com",
  "password": "secure_password",
  "full_name": "Customer Name",
  "phone": "+1234567890"
}
```

##### Customer Login
```http
POST /api/v1/customer/login
```

**Request Body:**
```json
{
  "username": "customer_username",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "jwt_token_here",
  "refresh_token": "refresh_token_here",
  "token_type": "bearer",
  "expires_in": 3600,
  "customer": {
    "id": 1,
    "username": "customer_username",
    "email": "customer@example.com",
    "role": "CUSTOMER"
  }
}
```

##### Refresh Token
```http
POST /api/v1/customer/refresh
```

#### Customer Profile Management

##### Get Customer Profile
```http
GET /api/v1/customer/profile
```

##### Update Customer Profile
```http
PUT /api/v1/customer/profile
```

##### Change Password
```http
POST /api/v1/customer/change-password
```

#### Customer Chat Management

##### Create Chat Thread
```http
POST /api/v1/customer/chat/threads
```

##### Get Chat Threads
```http
GET /api/v1/customer/chat/threads?skip=0&limit=20
```

##### Get Chat Messages
```http
GET /api/v1/customer/chat/threads/{thread_id}/messages?skip=0&limit=50
```

##### Send Chat Message
```http
POST /api/v1/customer/chat/threads/{thread_id}/messages
```

#### Customer Workflow

##### Submit Workflow Query (Streaming)
```http
POST /api/v1/customer/workflow/stream
```

##### Submit Workflow Query (Non-streaming)
```http
POST /api/v1/customer/workflow
```

##### Get Workflow History
```http
GET /api/v1/customer/interactions?skip=0&limit=20&interaction_type=workflow
```

---

### Employee API

Base path: `/api/v1/employee`

#### Employee Authentication

##### Register Employee
```http
POST /api/v1/employee/register
```

**Request Body:**
```json
{
  "username": "employee_username",
  "email": "employee@company.com",
  "password": "secure_password",
  "full_name": "Employee Name",
  "role": "employee",
  "department": "Engineering"
}
```

##### Employee Login
```http
POST /api/v1/employee/login
```

#### Employee Management (Admin Only)

##### List Employees
```http
GET /api/v1/employee/employees?skip=0&limit=20&role=employee&status=active
```

##### Get Employee Details
```http
GET /api/v1/employee/employees/{employee_id}
```

##### Update Employee
```http
PUT /api/v1/employee/employees/{employee_id}
```

##### Suspend/Activate Employee
```http
POST /api/v1/employee/employees/{employee_id}/suspend
POST /api/v1/employee/employees/{employee_id}/activate
```

#### Employee Workflow

##### Submit Workflow Query (Streaming)
```http
POST /api/v1/employee/workflow/stream
```

##### Submit Workflow Query (Non-streaming)
```http
POST /api/v1/employee/workflow
```

#### Employee Chat System

##### Create Chat Thread
```http
POST /api/v1/employee/chat/threads
```

##### Get Chat Threads
```http
GET /api/v1/employee/chat/threads?skip=0&limit=20
```

##### Send Chat Message
```http
POST /api/v1/employee/chat/threads/{thread_id}/messages
```

#### Employee Analytics

##### Get Employee Statistics
```http
GET /api/v1/employee/analytics?employee_id={id}&days=30
```

---

### Document API

Base path: `/api/v1/documents`

#### Document Management

##### Upload Document
```http
POST /api/v1/documents/upload
```

**Form Data:**
- `file`: Document file (PDF, DOCX, TXT, etc.)
- `title`: Document title
- `description`: Document description
- `is_public`: Boolean (optional)

##### List Documents
```http
GET /api/v1/documents?skip=0&limit=20&status=active&document_type=pdf
```

##### Get Document Details
```http
GET /api/v1/documents/{document_id}
```

##### Download Document
```http
GET /api/v1/documents/{document_id}/download
```

##### Update Document
```http
PUT /api/v1/documents/{document_id}
```

##### Delete Document
```http
DELETE /api/v1/documents/{document_id}
```

#### Document Processing

##### Process Document
```http
POST /api/v1/documents/{document_id}/process
```

Initiates document processing for vector embeddings and indexing.

##### Get Processing Status
```http
GET /api/v1/documents/{document_id}/status
```

---

### RAG (Retrieval-Augmented Generation) API

Base path: `/api/v1/rag`

#### Document-Enhanced Queries

##### Guest RAG Query
```http
POST /api/v1/rag/guest/query
```

**Request Body:**
```json
{
  "query": "What does the document say about gene therapy?",
  "session_id": "session-uuid",
  "document_id": 123,
  "limit": 3
}
```

##### Customer RAG Query
```http
POST /api/v1/rag/customer/query
```

##### Employee RAG Query
```http
POST /api/v1/rag/employee/query
```

---

### Product API

Base path: `/api/v1/products`

#### Product Management

##### List Products
```http
GET /api/v1/products?skip=0&limit=100&active_only=true
```

##### Get Product by ID
```http
GET /api/v1/products/{product_id}
```

##### Get Product by Index
```http
GET /api/v1/products/index/{product_index}
```

##### Search Products
```http
GET /api/v1/products/search?q=gene therapy&skip=0&limit=20
```

##### Filter Products by Type
```http
GET /api/v1/products/type/{product_type}?skip=0&limit=20
```

##### Filter Products by Subject
```http
GET /api/v1/products/subject/{subject}?skip=0&limit=20
```

##### Filter Products by Technology
```http
GET /api/v1/products/technology/{technology}?skip=0&limit=20
```

##### Filter Products by Price Range
```http
GET /api/v1/products/price-range?min_price=100&max_price=1000&skip=0&limit=20
```

#### Product Metadata

##### Get Product Types
```http
GET /api/v1/products/meta/types
```

##### Get Technologies
```http
GET /api/v1/products/meta/technologies
```

#### Product Management (Admin Only)

##### Create Product
```http
POST /api/v1/products
```

##### Update Product
```http
PUT /api/v1/products/{product_id}
```

##### Delete Product
```http
DELETE /api/v1/products/{product_id}?hard_delete=false
```

---

### Queue Management APIs

#### Customer Queue

##### Get Customer Queue Status
```http
GET /api/v1/customer/queue/status
```

##### Get Customer Request Status
```http
GET /api/v1/customer/queue/request/{request_id}
```

##### Cancel Customer Request
```http
POST /api/v1/customer/queue/request/{request_id}/cancel
```

#### Employee Queue

##### Get Employee Queue Status
```http
GET /api/v1/employee/queue/status
```

##### Get Employee Request Status
```http
GET /api/v1/employee/queue/request/{request_id}
```

#### Document Queue

##### Get Document Queue Status
```http
GET /api/v1/documents/queue/status
```

##### Get Document Request Status
```http
GET /api/v1/documents/queue/request/{request_id}
```

##### Cancel Document Request
```http
POST /api/v1/documents/queue/request/{request_id}/cancel
```

---

### Feedback API

Base path: `/api/v1/feedback`

#### Universal Feedback System

##### Submit Feedback
```http
POST /api/v1/feedback/
```

**Request Body:**
```json
{
  "interaction_id": "uuid-interaction-id",
  "feedback_type": "GENERAL",
  "rating": 5,
  "feedback_text": "Great response!",
  "was_helpful": true,
  "message_id": "optional-message-id"
}
```

##### Submit Employee Interaction Feedback
```http
POST /api/v1/employee/interactions/{interaction_id}/feedback
```

---

## Data Models

### User Roles

- **Guest**: Unauthenticated users with limited access
- **Customer**: Registered users with basic features
- **Premium Customer**: Enhanced features and priority support
- **Employee**: Staff members with internal tools access
- **Manager**: Management-level access and oversight
- **Admin**: Full system administration rights

### Document Types

- **PDF**: Portable Document Format
- **DOCX**: Microsoft Word documents
- **TXT**: Plain text files
- **MD**: Markdown files

### Document Status

- **PENDING**: Awaiting processing
- **PROCESSING**: Currently being processed
- **COMPLETED**: Processing completed successfully
- **FAILED**: Processing failed
- **ARCHIVED**: Archived document

### Interaction Types

- **WORKFLOW**: AI workflow interactions
- **CHAT**: Chat conversations
- **DOCUMENT_QUERY**: Document-based queries
- **FEEDBACK**: User feedback submissions

## Error Handling

The API uses standard HTTP status codes and returns detailed error information:

```json
{
  "detail": "Error description",
  "error_code": "SPECIFIC_ERROR_CODE",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Common Error Codes

- **400**: Bad Request - Invalid input data
- **401**: Unauthorized - Authentication required
- **403**: Forbidden - Insufficient permissions
- **404**: Not Found - Resource not found
- **422**: Validation Error - Input validation failed
- **429**: Too Many Requests - Rate limit exceeded
- **500**: Internal Server Error - Server-side error

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Guest users**: 10 requests per minute
- **Customers**: 30 requests per minute
- **Premium customers**: 60 requests per minute
- **Employees**: 100 requests per minute

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 25
X-RateLimit-Reset: 1640995200
```

## Queue System

The API uses a sophisticated queue system to manage resource allocation:

### Queue Types

1. **Customer Request Queue**: Manages customer workflow requests
2. **Employee Request Queue**: Handles employee workflow requests
3. **Document Processing Queue**: Manages document upload and processing
4. **Guest Request Queue**: Handles guest workflow requests

### Queue Features

- **Fair scheduling**: FIFO with priority support
- **Timeout handling**: Automatic request timeout
- **Cancellation**: Request cancellation support
- **Monitoring**: Real-time queue statistics
- **Background processing**: Non-blocking request processing

### Queue Statistics

Each queue provides statistics including:
- Current queue size
- Active requests count
- Average processing time
- Total processed requests
- Failed request count

## Streaming Responses

Many endpoints support real-time streaming using Server-Sent Events (SSE):

```javascript
const eventSource = new EventSource('/api/v1/customer/workflow/stream');

eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

eventSource.addEventListener('workflow_complete', function(event) {
  console.log('Workflow completed:', event.data);
  eventSource.close();
});
```

## WebSocket Support

Real-time communication is supported through WebSocket connections for:
- Live chat systems
- Real-time notifications
- System monitoring updates

## Security Features

- **JWT Authentication**: Secure token-based authentication
- **Role-based Access Control**: Granular permission system
- **Rate Limiting**: Abuse prevention
- **Input Validation**: Comprehensive data validation
- **CORS Configuration**: Cross-origin request handling
- **Password Security**: Secure password hashing and policies

## Monitoring and Analytics

The API provides comprehensive monitoring:
- **Health checks**: Multi-level health monitoring
- **Performance metrics**: Response times and throughput
- **Usage analytics**: User interaction patterns
- **Error tracking**: Detailed error logging and tracking
- **Resource monitoring**: System resource utilization

## Development and Testing

### Test Endpoints

Several test endpoints are available for development:
- `/test`: Stream test page
- `/api/system/test`: System component testing
- `/health`: Basic health check

### Environment Configuration

Key environment variables:
- `DATABASE_URL`: Database connection string
- `GUEST_MAX_CONCURRENT_REQUESTS`: Guest queue configuration
- `GUEST_MAX_QUEUE_SIZE`: Maximum queue size
- `GUEST_REQUEST_TIMEOUT_SEC`: Request timeout settings

This comprehensive API documentation covers all major endpoints and features of the GeneStory Workflow API system. For additional details or specific use cases, refer to the individual endpoint documentation or contact the development team.

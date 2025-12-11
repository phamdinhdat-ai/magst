# MCP Server Docker Deployment

This directory contains the Docker setup for the Model Context Protocol (MCP) server with PostgreSQL vector database support.

## Services

- **PostgreSQL with pgvector**: Vector database for embeddings storage
- **PgAdmin**: Web-based PostgreSQL administration
- **MCP Server**: FastMCP server with retrieval and memory capabilities

## Quick Start

1. **Copy environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit the .env file** with your preferred settings:
   ```bash
   # Database Configuration
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=your_secure_password_here
   POSTGRES_DB=vectordb

   # PgAdmin Configuration
   PGADMIN_DEFAULT_EMAIL=admin@example.com
   PGADMIN_DEFAULT_PASSWORD=admin_password_here

   # MCP Server Configuration
   MCP_PORT=50051
   DEVICE=cpu  # or cuda for GPU support
   ```

3. **Start the services**:
   ```bash
   docker-compose up -d
   ```

4. **Check service status**:
   ```bash
   docker-compose ps
   ```

## Service Endpoints

- **MCP Server**: `localhost:50051`
- **PostgreSQL**: `localhost:5433`
- **PgAdmin**: `http://localhost:8888`

## MCP Tools Available

The MCP server provides the following tools:

### 1. Vector Retrieval
- `retrieve_vectorstore_with_reranker`: Advanced retrieval with reranking
- `retrieve_vectorstore`: Simple vector retrieval
- `configure_retrieval_settings`: Configure retrieval parameters

### 2. Document Ingestion
- `ingest_documents`: Add documents to a collection
- `file_to_documents`: Convert file content to documents

### 3. Memory Management
- `memory_saver`: Save conversation context
- `memory_loader`: Load relevant conversation history

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | postgres | Database username |
| `POSTGRES_PASSWORD` | - | Database password |
| `POSTGRES_DB` | vectordb | Database name |
| `MCP_PORT` | 50051 | MCP server port |
| `MCP_HOST` | 0.0.0.0 | MCP server host |
| `DEVICE` | cpu | Device for AI models (cpu/cuda) |
| `EMBEDDING_MODEL` | AITeamVN/Vietnamese_Embedding_v2 | Embedding model |
| `RERANKING_MODEL` | AITeamVN/Vietnamese_Reranker | Reranking model |

### GPU Support

To enable GPU support:

1. Install nvidia-docker2
2. Update `.env`:
   ```
   DEVICE=cuda
   ```
3. Add GPU support to docker-compose.yaml:
   ```yaml
   mcp-server:
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: 1
               capabilities: [gpu]
   ```

## Development

### Building Locally

```bash
# Build the MCP server image
docker-compose build mcp-server

# Rebuild after code changes
docker-compose up -d --build mcp-server
```

### Logs

```bash
# View all service logs
docker-compose logs

# View specific service logs
docker-compose logs mcp-server
docker-compose logs db
```

### Database Access

#### Via PgAdmin
1. Open http://localhost:8888
2. Login with credentials from .env file
3. Add server:
   - Host: `db`
   - Port: `5432`
   - Username/Password: from .env file

#### Via Command Line
```bash
# Connect to PostgreSQL container
docker exec -it local_pgdb psql -U postgres -d vectordb

# List collections
\dt
```

## Health Checks

The services include health checks:

- **Database**: Checks PostgreSQL availability
- **MCP Server**: HTTP health endpoint at `/health`

Check health status:
```bash
docker-compose ps
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in docker-compose.yaml if needed
2. **Memory issues**: Increase Docker memory limits for large models
3. **Permission errors**: Check file permissions in mounted volumes

### Reset Database

```bash
# Stop services and remove volumes
docker-compose down -v

# Restart
docker-compose up -d
```

### View Model Loading Progress

```bash
# Monitor MCP server startup
docker-compose logs -f mcp-server
```

## Production Deployment

For production:

1. Use environment-specific .env files
2. Configure proper secrets management
3. Set up SSL/TLS termination
4. Configure resource limits
5. Set up monitoring and logging

### Resource Requirements

- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **With GPU**: Additional GPU memory for models

## API Usage

Once running, the MCP server can be used with MCP clients:

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print("Available tools:", [tool.name for tool in tools.tools])
```

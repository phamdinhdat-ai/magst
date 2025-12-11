# GenStory Deployment

This directory contains all the necessary files to deploy the GenStory chatbot service using Docker and Docker Compose.

## Components

- **Docker Compose**: Orchestrates the deployment of multiple services
- **PostgreSQL Database**: Stores application data
- **GenStory Service**: The main application service

## Quick Start

### Prerequisites

- Docker and Docker Compose installed on your system
- Proper environment configuration in `.env` file

### Running the Application

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Alternative: Using the run.sh Script

You can also use the provided `run.sh` script for managing individual containers:

```bash
# Build the Docker image
./run.sh build

# Run the container in detached mode
./run.sh run-detached

# View container logs
./run.sh logs

# Access container shell
./run.sh shell

# Stop the container
./run.sh stop
```

## Environment Configuration

Make sure to properly configure the `.env` file with the following settings:

- Database connection information
- API configuration
- Security settings
- LLM provider settings
- Performance optimization parameters

## Health Checks

The service provides a `/health` endpoint that returns a status message. This endpoint is used by Docker to verify the service's health.

## Scaling

To scale the service, adjust the following parameters in `.env`:

- `UVICORN_WORKERS`: Number of worker processes
- `LOAD_BALANCER_ENABLED`: Enable/disable load balancing
- `QUEUE_MANAGER_MAX_TASKS`: Maximum concurrent tasks

## Troubleshooting

If you encounter issues:

1. Check container logs: `docker-compose logs`
2. Verify database connectivity
3. Ensure environment variables are properly set
4. Check available disk space and system resources

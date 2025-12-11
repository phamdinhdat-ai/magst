# Docker Deployment Guide for GST Chatbot

This guide explains how to deploy the GST Chatbot application using Docker.

## Requirements

- Docker
- Docker Compose (optional, for using docker-compose.yml)

## Available Scripts

The `run.sh` script provides several commands to manage your Docker container:

```bash
# Build the Docker image
./run.sh build

# Run the container in foreground mode
./run.sh run

# Run the container in background mode
./run.sh run-detached

# Stop the running container
./run.sh stop

# Restart the container
./run.sh restart

# View container logs
./run.sh logs

# Open a shell inside the container
./run.sh shell

# Check container and image status
./run.sh status

# Clean up container and image
./run.sh cleanup
```

## Using Docker Compose

Alternatively, you can use Docker Compose:

```bash
# Build and start the container
docker-compose up -d

# Stop the container
docker-compose down

# View logs
docker-compose logs -f
```

## Environment Variables

The application uses the environment variables defined in the `.env` file. Make sure this file is present in the deployment directory.

## Ports

The application exposes port 8000 by default.

## Customization

You can modify the following settings in the `run.sh` script:

- `IMAGE_NAME`: The name of the Docker image
- `CONTAINER_NAME`: The name of the Docker container
- `VERSION`: The version tag for the Docker image
- `PORT`: The port to expose on the host

## Troubleshooting

If you encounter any issues:

1. Check container logs: `./run.sh logs`
2. Check container status: `./run.sh status`
3. Try rebuilding the image: `./run.sh build`

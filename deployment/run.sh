#!/bin/bash

# Configuration
IMAGE_NAME="gst-service"
CONTAINER_NAME="gst-chatbot-container"
VERSION="v0.0.6"
PORT="8000"

# Color codes for pretty output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to display usage information
show_usage() {
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  $0 ${GREEN}build${NC}              - Build Docker image"
    echo -e "  $0 ${GREEN}run${NC}                - Run container in foreground"
    echo -e "  $0 ${GREEN}run-detached${NC}       - Run container in background"
    echo -e "  $0 ${GREEN}stop${NC}               - Stop the container"
    echo -e "  $0 ${GREEN}restart${NC}            - Restart the container"
    echo -e "  $0 ${GREEN}logs${NC}               - Show container logs"
    echo -e "  $0 ${GREEN}shell${NC}              - Open shell inside container"
    echo -e "  $0 ${GREEN}status${NC}             - Check container status"
    echo -e "  $0 ${GREEN}cleanup${NC}            - Remove container and image"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

# Function to build the Docker image
build_image() {
    echo -e "${GREEN}Building Docker image: $IMAGE_NAME:$VERSION${NC}"
    docker build -t "$IMAGE_NAME:$VERSION" -t "$IMAGE_NAME:latest" .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Build successful!${NC}"
    else
        echo -e "${RED}Build failed!${NC}"
        exit 1
    fi
}

# Function to run the container
run_container() {
    local detached=$1
    
    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        echo -e "${YELLOW}Container already exists. Stopping and removing...${NC}"
        docker stop "$CONTAINER_NAME" > /dev/null
        docker rm "$CONTAINER_NAME" > /dev/null
    fi
    
    # Run mode (detached or foreground)
    if [ "$detached" = true ]; then
        echo -e "${GREEN}Running container in detached mode${NC}"
        docker run --name "$CONTAINER_NAME" \
            -v $(pwd):/app \
            -p $PORT:8000 \
            --network=host \
            --env-file .env \
            -d \
            "$IMAGE_NAME:$VERSION"
    else
        echo -e "${GREEN}Running container in foreground mode${NC}"
        docker run --name "$CONTAINER_NAME" \
            -p $PORT:8000 \
            --network=host \
            --env-file .env \
            "$IMAGE_NAME:$VERSION"
    fi
}

# Function to stop the container
stop_container() {
    if docker ps --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        echo -e "${GREEN}Stopping container: $CONTAINER_NAME${NC}"
        docker stop "$CONTAINER_NAME"
    else
        echo -e "${YELLOW}Container is not running${NC}"
    fi
}

# Function to show container logs
show_logs() {
    if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        echo -e "${GREEN}Showing logs for container: $CONTAINER_NAME${NC}"
        docker logs -f "$CONTAINER_NAME"
    else
        echo -e "${RED}Container does not exist${NC}"
    fi
}

# Function to open a shell in the container
open_shell() {
    if docker ps --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        echo -e "${GREEN}Opening shell in container: $CONTAINER_NAME${NC}"
        docker exec -it "$CONTAINER_NAME" bash
    else
        echo -e "${RED}Container is not running${NC}"
    fi
}

# Function to check container status
check_status() {
    echo -e "${GREEN}Container status:${NC}"
    docker ps -a --filter "name=$CONTAINER_NAME" --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
    
    echo -e "\n${GREEN}Image status:${NC}"
    docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.CreatedSince}}"
}

# Function to clean up container and image
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    
    if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        echo -e "${YELLOW}Removing container: $CONTAINER_NAME${NC}"
        docker stop "$CONTAINER_NAME" > /dev/null 2>&1
        docker rm "$CONTAINER_NAME"
    fi
    
    if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^$IMAGE_NAME"; then
        echo -e "${YELLOW}Removing image: $IMAGE_NAME${NC}"
        docker rmi "$IMAGE_NAME:$VERSION" "$IMAGE_NAME:latest"
    fi
    
    echo -e "${GREEN}Cleanup complete!${NC}"
}

# Main script execution
case "$1" in
    build)
        build_image
        ;;
    run)
        run_container false
        ;;
    run-detached)
        run_container true
        ;;
    stop)
        stop_container
        ;;
    restart)
        stop_container
        run_container true
        ;;
    logs)
        show_logs
        ;;
    shell)
        open_shell
        ;;
    status)
        check_status
        ;;
    cleanup)
        cleanup
        ;;
    *)
        show_usage
        ;;
esac

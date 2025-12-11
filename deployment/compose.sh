#!/bin/bash

# Configuration
COMPOSE_FILE="docker-compose.yaml"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to display usage information
show_usage() {
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  $0 ${GREEN}up${NC}                - Start all services"
    echo -e "  $0 ${GREEN}down${NC}              - Stop all services"
    echo -e "  $0 ${GREEN}build${NC}             - Build all services"
    echo -e "  $0 ${GREEN}logs${NC}              - Show logs from all services"
    echo -e "  $0 ${GREEN}restart${NC}           - Restart all services"
    echo -e "  $0 ${GREEN}status${NC}            - Show status of all services"
    echo -e "  $0 ${GREEN}ps${NC}                - List running containers"
    echo -e "  $0 ${GREEN}exec <service> <cmd>${NC} - Execute command in service container"
}

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose is not installed or not in PATH${NC}"
    echo -e "${YELLOW}Try using 'docker compose' instead of 'docker-compose'${NC}"
    if command -v docker &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
        exit 1
    fi
else
    COMPOSE_CMD="docker-compose"
fi

# Main script execution
case "$1" in
    up)
        echo -e "${GREEN}Starting all services...${NC}"
        $COMPOSE_CMD -f $COMPOSE_FILE up -d
        ;;
    down)
        echo -e "${YELLOW}Stopping all services...${NC}"
        $COMPOSE_CMD -f $COMPOSE_FILE down
        ;;
    build)
        echo -e "${GREEN}Building all services...${NC}"
        $COMPOSE_CMD -f $COMPOSE_FILE build
        ;;
    logs)
        echo -e "${GREEN}Showing logs from all services...${NC}"
        $COMPOSE_CMD -f $COMPOSE_FILE logs -f
        ;;
    restart)
        echo -e "${YELLOW}Restarting all services...${NC}"
        $COMPOSE_CMD -f $COMPOSE_FILE restart
        ;;
    status|ps)
        echo -e "${GREEN}Service status:${NC}"
        $COMPOSE_CMD -f $COMPOSE_FILE ps
        ;;
    exec)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${RED}Error: Missing service name or command${NC}"
            show_usage
            exit 1
        fi
        echo -e "${GREEN}Executing command in $2 container...${NC}"
        $COMPOSE_CMD -f $COMPOSE_FILE exec $2 ${@:3}
        ;;
    *)
        show_usage
        ;;
esac

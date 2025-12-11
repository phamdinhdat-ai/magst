#!/bin/bash

# MCP Server Docker Setup Script

echo "🚀 Setting up MCP Server with Docker..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install docker-compose."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created. Please edit it with your preferred settings."
    echo "📖 Key settings to review:"
    echo "   - POSTGRES_PASSWORD: Set a secure password"
    echo "   - PGADMIN_DEFAULT_EMAIL: Your admin email"
    echo "   - PGADMIN_DEFAULT_PASSWORD: Admin password"
    echo "   - DEVICE: 'cpu' or 'cuda' for GPU support"
    echo ""
    read -p "Press Enter to continue after editing .env file..."
fi

# Pull images
echo "📦 Pulling Docker images..."
docker-compose pull

# Build MCP server
echo "🔨 Building MCP server..."
docker-compose build mcp-server

# Start services
echo "▶️  Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to start..."
sleep 10

# Check service status
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "🎉 MCP Server setup complete!"
echo ""
echo "📍 Service URLs:"
echo "   MCP Server: localhost:50051"
echo "   PgAdmin: http://localhost:8888"
echo "   PostgreSQL: localhost:5433"
echo ""
echo "🔧 Useful commands:"
echo "   docker-compose logs -f mcp-server  # View MCP server logs"
echo "   docker-compose stop               # Stop all services"
echo "   docker-compose down -v            # Stop and remove data"
echo ""
echo "📚 See README.md for more information."

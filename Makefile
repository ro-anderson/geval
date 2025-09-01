.PHONY: run app clean docker-clean help

SHELL=/bin/bash

# Check if 'docker compose' is available, otherwise fall back to 'docker-compose'
ifeq ($(shell docker compose version 2>/dev/null),)
  DOCKER_COMPOSE := docker-compose
else
  DOCKER_COMPOSE := docker compose
endif

## Build (if needed) and run the RAG service in Docker
run: build
	$(DOCKER_COMPOSE) up web

## Build (if needed) and run the RAG service in Docker
app: build
	$(DOCKER_COMPOSE) up web

## Build the Docker images
build:
	$(DOCKER_COMPOSE) build

## Remove Python cache files
clean:
	find . -name "__pycache__" -type d -exec rm -r {} \+

## Remove Docker containers, networks, and volumes
docker-clean:
	$(DOCKER_COMPOSE) down -v --remove-orphans

## Display help information
help:
	@echo "Available commands:"
	@echo "  make run           - Build (if needed) and run the RAG service in Docker"
	@echo "  make app           - Build (if needed) and run the RAG service in Docker"
	@echo "  make clean         - Remove Python cache files"
	@echo "  make docker-clean  - Remove Docker containers, networks, and volumes"
	@echo "  make help          - Display this help information"

# Default target
.DEFAULT_GOAL := help

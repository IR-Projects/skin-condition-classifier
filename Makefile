.PHONY: help lint test run docker-build docker-run compose-up compose-down

help:
	@echo "Available commands:"
	@echo "  make lint         - Run ruff checks"
	@echo "  make test         - Run pytest"
	@echo "  make run          - Start API locally with uvicorn"
	@echo "  make docker-build - Build the API Docker image"
	@echo "  make docker-run   - Run the API container with volume mount"
	@echo "  make compose-up   - Start services using docker compose"
	@echo "  make compose-down - Stop docker compose services"

lint:
	ruff check .

test:
	pytest

run:
	uvicorn api.main:app --reload

docker-build:
	docker build -t skin-api .

docker-run:
	docker run --rm -p 8000:8000 -v "${PWD}/models:/app/models" -e MODEL_PATH="models/efficientnet_b0_best.pt" skin-api

compose-up:
	docker compose up

compose-down:
	docker compose down
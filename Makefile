.PHONY: install run docker-up docker-down test clean lint help

# ── Default target ────────────────────────────────────────────────────────
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Install ───────────────────────────────────────────────────────────────
install: ## Install Python dependencies
	pip install -r requirements.txt

# ── Run ───────────────────────────────────────────────────────────────────
run: ## Start the Streamlit app locally
	streamlit run app.py

# ── Docker ────────────────────────────────────────────────────────────────
docker-up: ## Start all services with Docker Compose
	docker compose up --build -d

docker-down: ## Stop all Docker Compose services
	docker compose down

# ── Testing ───────────────────────────────────────────────────────────────
test: ## Run the test suite with pytest
	pytest tests/ -v

# ── Linting ───────────────────────────────────────────────────────────────
lint: ## Run ruff linter (install ruff first)
	ruff check .

# ── Cleanup ───────────────────────────────────────────────────────────────
clean: ## Remove caches, ChromaDB data, and uploaded PDFs
	rm -rf chroma_db/ uploaded_pdfs/ __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned."

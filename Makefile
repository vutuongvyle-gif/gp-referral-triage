.PHONY: setup up down mcp critic app ingest data eval test lint

setup:   ## install everything
	uv sync --all-groups
	uv pip install pip
	uv run python -m spacy download en_core_web_lg

up:      ## start Postgres, HAPI FHIR and Phoenix
	docker compose up -d

down:
	docker compose down

mcp:     ## MCP tool server
	uv run python -m referral_triage.mcp_servers.server

critic:  ## Critic A2A service
	uv run python -m referral_triage.a2a_critic.server

app:     ## GP review UI
	uv run streamlit run app/review_app.py

ingest:  ## build the criteria vector store
	uv run python -m referral_triage.rag.ingest

data:    ## generate and load synthetic patients
	uv run python scripts/load_fhir.py

eval:
	uv run python -m evals.run

test:
	uv run pytest -m "not eval"

lint:
	uv run ruff check . && uv run ruff format --check .
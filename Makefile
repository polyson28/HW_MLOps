IMAGE ?= dockerhubuser/hw-mlops:latest

.PHONY: docker-push test lint

docker-push:
	docker build -t $(IMAGE) .
	docker push $(IMAGE)

test:
	poetry run pytest -q

lint:
	poetry run ruff check .
	poetry run black --check .

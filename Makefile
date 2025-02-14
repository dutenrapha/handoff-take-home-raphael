IMAGE_NAME=ai_ml_evaluator
DATA_DIR=$(shell pwd)/data
REPORTS_DIR=$(shell pwd)/reports
CONFIG_FILE=$(shell pwd)/config.yaml

build:
	docker-compose build

evaluation:
	docker-compose up ai_ml_evaluator

ui:
	docker-compose up streamlit_app

clean:
	docker-compose down
	docker container prune -f

fclean: clean
	docker image prune -f
	rm -f *.csv *.json

.PHONY: build evaluation ui all clean fclean

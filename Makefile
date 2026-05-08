.PHONY: build download engineer train shell tensorboard run-backend run-frontend run-app

build:
	docker compose build

download:
	docker compose run --rm training \
		python app/pipelines/download_pipeline.py

engineer:
	docker compose run --rm training \d 
		python app/pipelines/engineering_pipeline.py


segment:
	docker compose run --rm training \
		python app/pipelines/segmentation_pipeline.py


refine:
	docker compose run --rm training \
		python app/pipelines/segmentation_refinement_pipeline.py


seg-stats:
	docker compose run --rm training \
		python app/pipelines/segmentation_statistics_pipeline.py


train-unet-resnet:
	docker compose run --rm training \
		python app/pipelines/train_unet_resnet_pipeline.py

train-unet-efficient:
	docker compose run --rm training \
		python app/pipelines/train_unet_efficientnet_pipeline.py


train:
	docker compose run --rm training \
		python app/pipelines/train_pipeline.py


ensemble:
	docker compose run --rm training \
		python app/pipelines/ensemble_segmentation_pipeline.py


train-classifier:
	docker compose run --rm training \
		python app/pipelines/train_classifier_pipeline.py

infer:
	docker compose run --rm training \
		python app/pipelines/inference_pipeline.py

shell:
	docker compose run --rm training bash

tensorboard:
	docker compose run --rm -p 6006:6006 training tensorboard --logdir=/workspace/runs --bind_all

run-backend:
	cd app/api && uvicorn main:app --host 127.0.0.1 --port 8000 --reload

run-frontend:
	cd ../lesion-inference-ui && npm run dev

run-app:
	$(MAKE) -j 2 run-backend run-frontend
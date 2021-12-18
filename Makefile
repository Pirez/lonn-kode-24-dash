
build:
	docker build -t dashboard .

run:
	docker run --net host --restart always -e test=1 -it -d -p 8000:8000 dashboard

pip:
	pip3 freeze > requirements.txt
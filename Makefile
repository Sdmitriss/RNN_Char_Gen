build:
	docker build --tag rnn .

run:
	docker run --rm -it -p 8000:8000 \
		-v ./model:/app/model \
		-v ./output:/app/output \
		rnn

	
  
      


.PHONY: test venv
download_v6:
	mkdir data/v6
	curl -L "https://app.roboflow.com/ds/21JxaNFWvy?key=dWDI4e1efw" > data/v6/roboflow.zip
	unzip data/v6/roboflow.zip -d data/v6/
run:
	venv/bin/python3 learn.py

train:
	venv/bin/python3 train.py

build0:
	# sudo apt-get install python3.10
	# sudo apt install python3-pip
	# sudo apt-get install unzip
	mkdir data/v1;
	curl -L "https://app.roboflow.com/ds/vZpxy3h6AT?key=xqnkFTwCdy" > data/v1/roboflow.zip; 
	unzip data/v1/roboflow.zip -d data/v1/; 
	rm data/v1/roboflow.zip
	# cp -r data/v3/train/* data/v4/train
	# cp -r data/v3/valid/* data/v4/valid
	# cp -r data/v3/test/* data/v4/test
	# curl -L "https://app.roboflow.com/ds/h8uohoV1Fo?key=y4yt1YrRNl" > v3/roboflow.zip;
	# unzip v3/roboflow.zip -d v3/;
build:
	python3 -m venv venv
	venv/bin/pip3 install -r requirements.txt
	venv/bin/pip3 install -U ultralytics
predict:
	venv/bin/python3 predict.py

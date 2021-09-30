# Service directory

Dockerfile can be used for building on both ARM64 and x86 systems. \
The web application resides in `app` directory - those are flask related files. \
Microservices are in `microservices` directory.

Important to note that to run the container, volumes need to be as follows, assuming run is executed from this folder: 
`docker run -it -p 80:80 -v $PWD/app:/app -v $PWD/microservices:/app/microservices $IMAGE_NAME`
Network port necessary for web service is 80. 
The microservices folder needs to go in as `/app/microservices`.
FROM demosense/raspberrypi3-opencv:latest
COPY . /usr/src/app
WORKDIR /usr/src/app
RUN pip install imutils
CMD ["python", "motion-detector.py"]
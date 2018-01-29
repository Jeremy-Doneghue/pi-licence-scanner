FROM demosense/raspberrypi3-opencv:latest
RUN pip install imutils
COPY . /usr/src/app
CMD ["python", "/usr/src/app/motion-detector.py"]
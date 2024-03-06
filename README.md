This groundbreaking project integrates real-time object detection tech for the visually impaired. A strategically placed camera identifies objects, providing instant voice alerts for enhanced safety and independence. Beyond obstacles, it offers comprehensive object identification, promoting non-contact navigation for a dignified experience. Inclusive for all, it fosters a compassionate community, transforming societal norms.

How to run this code?

Step 1: Create a directory in your local machine and cd into it

mkdir ~/Desktop/Real-Time-Object-Detection-with-Voice-Output
cd ~/Desktop/Real-Time-Object-Detection-with-Voice-Output

Step 2: Clone the repository and cd into the folder:

git clone https://github.com/bmanasa123/Real-Time-Object-Detection-with-Voice-Output.git
cd Real-Time-Object-Detection-with-Voice-Output

Step 3: Install all the necessary libraries. I used Windows for this project. These are some of the libraries I had to install:

brew install opencv
pip install opencv-python
pip install opencv-contrib-python
pip install opencv-python-headless
pip install opencv-contrib-python-headless
pip install matplotlib
pip install imutils
pip install cap                
pip install pyttsx3
pip install blinker

Make sure to download and install opencv and and opencv-contrib releases for OpenCV 3.3. This ensures that the deep neural network (dnn) module is installed. You must have OpenCV 3.3 (or newer) to run this code.

Step 4: After installing above libraries runn the below command in command prompt 

python main.py 
then you will be redirected to a frame that will detect all objects and pronounces.


# This code opens the folder where your test images a stored and then tests them.

import os, os.path, time
import subprocess


print ('Testing Images')
path= 'test/'
for file in os.listdir("test/"):   # Look into the given path
    if file.endswith(".jpg"):	   # Look for files with .jpg as extension
        print (str(file)) 
        subprocess.call(['python','test_core.py' ,str(path+file)], shell=True) # Calling a system process to run the python script. 
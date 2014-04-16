You will have to 
$ssh -X username@gpu01.cc.iitk.ac.in 
to run the code on the gpu servers and open the window containing disparity map at the end.

To run the code :

Go to the opencv_build folder in your home folder on the gpu server.
Copy the file main_code.cu and the images 21.tiff and 22.tiff
Also copy another set of images i.e., w1.tiff and w2.tiff(to check and as per the requirement).

$ opencv main_code.cu -o main_code
			or
$nvcc main_code.cu `pkg-config opencv --cflags --libs` -o main_code

then
$ main_code 22.tiff 21.tiff
for the 22.tiff and 21.tiff images from the given dataset.

or

$main_code w1.tiff w2.tiff
for the other set of images from the EPFL website.

Before running "22.tiff and 21.tiff" just check for the value of the contrastThreshold (w.r.t SIFT). 
It should be "0.02" for the "22.tiff and 21.tiff" pair of images. 
For other pair of images "w1.tiff and w2.tiff" , just adjust the value to 0.09 so that the number of identied "SIFT feature points" should be below 1000.

Because if is greater than 1000 , then sometimes due to space constrains it gives "Segmentation Fault".
Now after you run this command , a window would appear which would display the disparity map of the stereo images.

Note : The code works perfectly fine on our system and on our gpu account. If it doesen't works properly when you run, even after following all these instructions, kindly contact us. 

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h> 
#include <time.h>

#include <stdio.h>
#include <cuda.h>

using namespace std;
using namespace cv;

typedef struct met{
  int p[9];
}met;

// Kernel that executes on the CUDA device
__global__ void keygen( Point *a,Point *b, int N, double *neg_ans,met *mat,double *ans, double *arr,int l,int n1, int n2)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
  {
    int i = idx % n1;
    int j = (int)(idx/n1);
    mat[idx].p[0] = a[i].x * b[j].x;    //pass mat as structure contataining n1*n2 arrays of 9 elements.
    mat[idx].p[1] = a[i].x * b[j].y;
    mat[idx].p[2] = a[i].x;
    mat[idx].p[3] = b[j].x * a[i].y;
    mat[idx].p[4] = b[j].y * a[i].y;
    mat[idx].p[5] = a[i].y;
    mat[idx].p[6] = b[j].x;
    mat[idx].p[7] = b[j].y;
    mat[idx].p[8]= 1;
    neg_ans[idx] = 0;
    for (int q = 0; q < 9; q++)
    {
      neg_ans[idx] = neg_ans[idx] + arr[q]*mat[idx].p[q];   //pass arr as original a.
    }
    neg_ans[idx] = (-1)*neg_ans[idx];      //pass neg_idx
    ans[idx] = (double)(neg_ans[idx]/mat[idx].p[l]);      //pass l,ans also
  }
}

const double THRESHOLD = 400;

 
int main(int argc, char** argv) {

	  if (argc < 2) {
	    cerr << "Too few arguments" << endl;
	    return -1;
	  }
	  
	  const char* filename = argv[1];
	  const char* filename2 = argv[2];
	  
	  printf("load file:%s\n", filename);
	  printf("load file:%s\n", filename2);
	  
	  // initialize detector and extractor
	  FeatureDetector* detector;
	  detector = new SiftFeatureDetector(
	                                     0, // nFeatures
	                                     4, // nOctaveLayers
	                                     0.02, // contrastThreshold
	                                     10, //edgeThreshold
	                                     1.6 //sigma
	                                     );
	  
	  DescriptorExtractor* extractor;
	  extractor = new SiftDescriptorExtractor();
	  
	  // Compute keypoints and descriptor from the source image in advance
	  vector<KeyPoint> keypoints2,keypoints1;
	  Mat descriptors2,descriptors1;
	  
	  Mat originalGrayImage = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	  resize(originalGrayImage,originalGrayImage,Size(800,800),0,0,INTER_CUBIC);

	  if (!originalGrayImage.data) {
	    cerr << "gray image load error  www" << endl;
	    return -1;
	  }

	  // Mat originalColorImage = imread(filename, CV_LOAD_IMAGE_ANYCOLOR|CV_LOAD_IMAGE_ANYDEPTH);
	  // resize(originalGrayImage,originalGrayImage,Size(600,600),0,0,INTER_CUBIC);

	  // if (!originalColorImage.data) {
	  //   cerr << "color image open error" << endl;
	  //   return -1;
	  // }
	  
	  detector->detect(originalGrayImage, keypoints2);
	  extractor->compute(originalGrayImage, keypoints2, descriptors2);
	  //printf("original image:%d keypoints are found.\n", (int)keypoints2.size());


	  Mat originalGrayImage2 = imread(filename2, CV_LOAD_IMAGE_GRAYSCALE);
	  resize(originalGrayImage2,originalGrayImage2,Size(800,800),0,0,INTER_CUBIC);

	  if (!originalGrayImage2.data) {
	    cerr << "gray image load error  22" << endl;
	    return -1;
	  }

	  // Mat originalColorImage2 = imread(filename2, CV_LOAD_IMAGE_ANYCOLOR|CV_LOAD_IMAGE_ANYDEPTH);
	  // if (!originalColorImage2.data) {
	  //   cerr << "color image open error" << endl;
	  //   return -1;
	  // }
	  
	  detector->detect(originalGrayImage2, keypoints1);
	  extractor->compute(originalGrayImage2, keypoints1, descriptors1);
	  //printf("original image:%d keypoints are found.\n", (int)keypoints1.size());
	  
	  //while (1) {
	    //capture >> frame;
	    
	    // load gray scale image from camera
	     Size size = originalGrayImage2.size();
	    // Mat grayFrame(size, CV_8UC1);
	    // cvtColor(frame, grayFrame, CV_BGR2GRAY);
	    // if (!grayFrame.data) {
	    //   cerr << "cannot find image file1" << endl;
	    //   exit(-1);
	    // }
	    
	    // Create a image for displaying mathing keypoints
	     Size sz = Size(size.width + originalGrayImage.size().width, size.height + originalGrayImage.size().height);
	     Mat matchingImage = Mat::zeros(sz, CV_8UC3);
	    
	    // // Draw camera frame
	     Mat roi1 = Mat(matchingImage, Rect(0, 0, size.width, size.height));
	     originalGrayImage2.copyTo(roi1);
	     // Draw original image
	     Mat roi2 = Mat(matchingImage, Rect(size.width, size.height, originalGrayImage.size().width, originalGrayImage.size().height));
	     originalGrayImage.copyTo(roi2);
	    
	    vector<DMatch> matches;
	    
	    // Detect keypoints
	    //detector->detect(grayFrame, keypoints1);
	    //extractor->compute(grayFrame, keypoints1, descriptors1);
	    
	    //printf("image1:%zd keypoints are found.\n", keypoints1.size());
	    
	    for (int i=0; i<keypoints1.size(); i++){
	      KeyPoint kp = keypoints1[i];
	      circle(matchingImage, kp.pt, cvRound(1), Scalar(255,255,0), 1, 8, 0);
	    }

	    for (int i=0; i<keypoints2.size(); i++){
	      KeyPoint kp_2 = keypoints2[i];
	      Point p = kp_2.pt;
	      circle(matchingImage, Point(size.width + p.x, size.height + p.y), cvRound(1), Scalar(0,0,0), 1, 8, 0);
	    }

	    double a[9] = { -9.41091199e-08,5.18249150e-06,-6.01573551e-03,-5.33521515e-06,-3.27683870e-08,1.53004085e-02,6.35805225e-03,-1.58209024e-02,1.00000000};
	    //float a[9] = { -0.00941091199,0.00518249150,-0.00601573551,-0.00533521515,-0.00327683870,0.00153004085,0.00635805225,-0.00158209024,1.00000000};
	    int i,j;

	    double accu_val[5][9];
	    long int accu[5][9];

	    int key[keypoints1.size()][keypoints2.size()];
	    int value[keypoints1.size()][keypoints2.size()];
	    int main[keypoints1.size()][keypoints2.size()];

	    for (i=0;i<9;i++){
	      double p = a[i]/2.5;
	      for (j=0;j<5;j++){
	          accu[j][i] = 0;
	          accu_val[j][i] = p*(j+1);
	      }
	    }

	    for (i=0;i<9;i++){
	          printf(" %lf\n",a[i]);
	    }

	    int l,k;
    	int n1 = keypoints1.size();
    	int n2 = keypoints2.size();
    	size_t n1_s = n1 * sizeof( Point );
    	size_t n2_s = n2 * sizeof( Point );
    	int block_size, n_blocks;
    	const int N = n1*n2; // Number of elements in arrays

    /*	for (i = 0; i < 1000; i++)
    	{
    		for (j = 0; j < 1000; j++)
    		{
    			;
    		}
    	}*/





	    for (l=0;l<9;l++)
	    {
	      //start CUDA
      
	      met *mat,*mat_d;// = (met*)malloc(N*sizeof(met));
	      double *neg_ans,*neg_ans_d;//= (int *)malloc(N*sizeof(int));
	      double *ans,*ans_d;// = (float*)malloc(N*sizeof(int));
	      Point *a_1_h,*a_2_h, *a_1_d, *a_2_d; // Pointer to host & device arrays
	      double *a_d;
	      printf("here 1 : \n");
	      a_1_h = (Point *)malloc(n1_s);    // Allocate array on host
	      cudaMalloc( (void **)&a_1_d, n1_s ); // Allocate array on device
	      printf("here 2 : \n");
	      neg_ans = (double *)malloc(N*sizeof(int));    // Allocate array on host
	      cudaMalloc( (void **)&neg_ans_d, N*sizeof(int) ); // Allocate array on device
	      printf("here 3 : \n");

	      mat = (met *)malloc(N*sizeof(met));    // Allocate array on host
	      cudaMalloc( (void **)&mat_d, N*sizeof(met)); // Allocate array on device      
	      printf("here 4 : \n");

	      ans = (double *)malloc(N*sizeof(ans));    // Allocate array on host
	      cudaMalloc( (void **)&ans_d, N*sizeof(double)); // Allocate array on device      
	      printf("here 5 : \n");

	      a_2_h = (Point *)malloc(n2_s);    // Allocate array on host
	      cudaMalloc( (void **)&a_2_d, n2_s ); // Allocate array on device
	      printf("here 6 : \n");

	      cudaMalloc( (void **)&a_d, 9*sizeof(double) ); // Allocate array on device
	      printf("here array : \n");
	      // Initialize host array and copy it to CUDA device
	      for ( i = 0; i < n1; i++ )
	          a_1_h[i] = keypoints1[i].pt;
	      printf("here gcsa : \n");
	      for ( j = 0; j < n2; j++ )
	          a_2_h[j] = keypoints2[j].pt;
	      printf("here asdf : \n");
	      /*for (k = 0; k < 9; k++)
	      {
	        a_d[k] = a[k];
	      }*/
	      printf("here 7 : \n");
	      cudaMemcpy( a_1_d, a_1_h, n1_s, cudaMemcpyHostToDevice );
	      cudaMemcpy( a_2_d, a_2_h, n2_s, cudaMemcpyHostToDevice );
	      cudaMemcpy( a_d, a, 9*sizeof(double), cudaMemcpyHostToDevice );
	      // Do calculation on device:
	      block_size = 256;
	      n_blocks  = N / block_size + ( N % block_size == 0 ? 0 : 1 );
	      printf("Entering GPU code : %d\n",l);
	      keygen <<< n_blocks, block_size >>> ( a_1_d, a_2_d, N, neg_ans_d, mat_d, ans_d,a_d,l,n1,n2);
	      printf("Exiting GPU code : %d\n",l);
	      // Retrieve result from device and store it in host array
	      cudaMemcpy( a, a_d, sizeof( double ) * 9, cudaMemcpyDeviceToHost );
	      cudaMemcpy( ans, ans_d, sizeof( double ) * N, cudaMemcpyDeviceToHost );
	      // Print results
	      for ( int i = 0; i < 9; i++ )
	          printf( "l = %d\t i = %d\t i = %lf\n",l,i, a[i] ); // Cleanup
	      free( a_1_h );
	      printf("it is here \n");
	      free( a_2_h );
	      cudaFree( a_1_d );
	      cudaFree( a_2_d );
	      free( mat );
	      free( neg_ans );
	      cudaFree(mat_d);
	      cudaFree(neg_ans_d);
	      cudaFree(ans_d);
	      cudaFree(a_d);
	      for (i=0;i<keypoints1.size();i++)
	      {
	           for (j=0;j<keypoints2.size();j++)
	           {
	              double val_ans = ans[i*keypoints1.size() + j];
	              int index = 0;
	              double close_aux;
	              double close = abs(val_ans - accu_val[0][l]);
	              for (k=0;k<5;k++)
	              {
	                close_aux = abs(val_ans - accu_val[k][l]);
	                if (close_aux < close){
	                  index = k;
	                  close = close_aux;
	                }
	              }
	              //printf("%d %lf\n",index,close);
	              accu[index][l] ++;
	              key[i][j] = index;
	              value[i][j] = l;
	              //printf("%E \n",ans);
	              //printf("%d %d %d %d %d %d %lf  \n",i,l,x_1,x_2,y_1,y_2,ans);
	          }
	      }

	      int in = 0;
	      int max_aux ;
	      int m;
	      int max = accu[0][l];
	      printf("it is here 2\n");
	      for (m=0;m<5;m++){
	        max_aux = accu[m][l];
	        if (max_aux > max){
	          in = m;
	          max = max_aux;
	        }
	      }
	      a[l] = accu_val[in][l];
	    }

	    printf("it is here 3\n");

	    int count = 0;
	    for (i=0;i<keypoints1.size();i++){
	      for (j=0;j<keypoints2.size();j++){
	        if (accu_val[key[i][j]][value[i][j]] == a[value[i][j]]){
	        //if (10){
	          count++;
	          main[i][j] = 1;
	        }
	      }
	    }

	    printf("priii %d\n",count);

	    for (i=0;i<9;i++){
	      for (j=0;j<5;j++){
	          printf("%ld ",accu[j][i]);
	      }
	    }

	    for (i=0;i<9;i++){
	          printf("  %E\n",a[i]);
	    }
	

		Mat F = Mat(3, 3, CV_64F, &a);
		cout << "F = "<< endl << " "  << F << endl << endl;

		//-- Step 5: calculate Essential Matrix

		double data[] = {1189.46 , 0.0, 805.49, 
		                0.0, 1191.78, 597.44,
		                0.0, 0.0, 1.0};//Camera Matrix
		Mat K(3, 3, CV_64F, data);
		Mat_<double> E = K.t() * F * K; //according to HZ (9.12)

		//-- Step 6: calculate Rotation Matrix and Translation Vector
		Matx34d P;
		Matx34d P1;
		//decompose E to P' , HZ (9.19)
		SVD svd(E,SVD::MODIFY_A);
		Mat svd_u = svd.u;
		Mat svd_vt = svd.vt;
		Mat svd_w = svd.w;
		Matx33d W(0,-1,0,1,0,0,0,0,1);//HZ 9.13
		Mat_<double> R = svd_u * Mat(W) * svd_vt; //HZ 9.19
		Mat_<double> t = svd_u.col(2); //u3

		// if (!CheckCoherentRotation (R)) {
		// std::cout<<"resulting rotation is not coherent\n";
		// P1 = 0;
		// return 0;
		// }

		P1 = Matx34d(R(0,0),R(0,1),R(0,2),t(0),
		             R(1,0),R(1,1),R(1,2),t(1),
		             R(2,0),R(2,1),R(2,2),t(2));

		//-- Step 7: Reprojection Matrix and rectification data
		Mat R1, R2, P1_, P2_, Q;
		Rect validRoi[2];
		double dist[] = { -0.03432, 0.05332, -0.00347, 0.00106, 0.00000};
		Mat D(1, 5, CV_64F, dist);

		//stereoRectify(K, D, K, D, img_1.size(), R, t, R1, R2, P1_, P2_, Q, CV_CALIB_ZERO_DISPARITY, 1, img_1.size(),  &validRoi[0], &validRoi[1] );
		//triangulatePoints(P1_,P2_,imgpts1,imgpts2,Q);
		Mat xyz;
		Mat disp, disp8;

	    StereoBM bm;
	    StereoSGBM sgbm;
	    StereoVar var;

	    Size img_size = originalGrayImage.size();

	    int numberOfDisparities = 0;
	    int SADWindowSize = 0;

	    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;
	    printf("number of disparity %d  \n",numberOfDisparities);

	    Rect roi_1, roi_2;

	    bm.state->roi1 = roi_1;
	    bm.state->roi2 = roi_2;
	    bm.state->preFilterCap = 31;
	    bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
	    bm.state->minDisparity = 0;
	    bm.state->numberOfDisparities = numberOfDisparities;
	    bm.state->textureThreshold = 10;
	    bm.state->uniquenessRatio = 15;
	    bm.state->speckleWindowSize = 100;
	    bm.state->speckleRange = 32;
	    bm.state->disp12MaxDiff = 1;

	    bm(originalGrayImage, originalGrayImage2, disp);
	    disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));

	    //new_d = cv.CreateImage( (400,400), disp.depth, disp.nChannels );
		//cv.Resize( disp8, new_de , interpolation=cv.CV_INTER_CUBIC );
        

        imshow("3-D reconstruction", disp8);

        printf("press any key to continue...");
        waitKey(0);
        printf("\n");
}

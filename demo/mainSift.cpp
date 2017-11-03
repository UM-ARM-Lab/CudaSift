//********************************************************//
// CUDA SIFT extractor by Marten Björkman aka Celebrandil //
//              celle @ csc.kth.se                       //
//********************************************************//

#include <iostream>
#include <cmath>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cudaSift/image.h>
#include <cudaSift/sift.h>
#include <cudaSift/utils.h>

int ImproveHomography(cudaSift::SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);
void PrintMatchData(cudaSift::SiftData &siftData1, cudaSift::SiftData &siftData2, cudaSift::Image &img);
void MatchAll(cudaSift::SiftData &siftData1, cudaSift::SiftData &siftData2, float *homography);

double ScaleUp(cudaSift::Image &res, cudaSift::Image &src);

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  int devNum = 0;
  if (argc>1)
    devNum = std::atoi(argv[1]);

  // Read images using OpenCV
  cv::Mat limg, rimg;
  cv::imread("data/left.pgm", 0).convertTo(limg, CV_32FC1);
  cv::imread("data/righ.pgm", 0).convertTo(rimg, CV_32FC1);
  unsigned int w = limg.cols;
  unsigned int h = limg.rows;
  std::cout << "Image size = (" << w << "," << h << ")" << std::endl;

  // Extract Sift features from images
  cudaSift::SiftData siftData1, siftData2;
  float initBlur = 1.0f;
  float thresh = 3.5f;
  cudaSift::InitSiftData(siftData1, 32768, true, true);
  cudaSift::InitSiftData(siftData2, 32768, true, true);

  const bool useStreams = false;// use streams?  cudaSift assumes user manages these!

  // Default stream is 0
  cudaStream_t stream1 = 0;
  cudaStream_t stream2 = 0;

  if (useStreams) {
    safeCall(cudaStreamCreate(&stream1));
    safeCall(cudaStreamCreate(&stream2));
  }

  siftData1.stream = stream1;// if both are 0 they just operate on the same stream,
  siftData2.stream = stream2;// AKA if not using streams this can be skipped

  // Initial Cuda images and download images to device
  std::cout << "Initializing data..." << std::endl;
  cudaSift::InitCuda(devNum);
  cudaSift::Image img1, img2;
  img1.Allocate(w, h, cudaSift::iAlignUp(w, 128), false, NULL, (float*)limg.data);
  img2.Allocate(w, h, cudaSift::iAlignUp(w, 128), false, NULL, (float*)rimg.data);

  img1.stream = siftData1.stream;// if not using streams, these will both be zero
  img2.stream = siftData2.stream;// AKA this step could be skipped

  img1.Download();
  img2.Download();

  // A bit of benchmarking
  for (thresh=1.00f;thresh<=4.01f;thresh+=0.50f) {
    for (int i=0;i<10;i++) {
      cudaSift::ExtractSift(siftData1, img1, 5, initBlur, thresh, 0.0f, false);
      cudaSift::ExtractSift(siftData2, img2, 5, initBlur, thresh, 0.0f, false);
    }

    // Match Sift features and find a homography
    for (int i=0;i<1;i++)
      cudaSift::MatchSiftData(siftData1, siftData2);
    float homography[9];
    int numMatches;
    cudaSift::FindHomography(siftData1, homography, &numMatches, 10000, 0.00f, 0.80f, 5.0);
    // ImproveHomography is in demo/geomFuncs.cpp
    int numFit = ImproveHomography(siftData1, homography, 5, 0.00f, 0.80f, 3.0);

    std::cout << "Number of original features: " <<  siftData1.numPts << " " << siftData2.numPts << std::endl;
    std::cout << "Number of matching features: " << numFit << " " << numMatches << " " << 100.0f*numFit/std::min(siftData1.numPts, siftData2.numPts) << "% " << initBlur << " " << thresh << std::endl;
  }

  // Print out and store summary data
  PrintMatchData(siftData1, siftData2, img1);
  cv::imwrite("data/limg_pts.pgm", limg);

  // Free Sift data from device
  cudaSift::FreeSiftData(siftData1);
  cudaSift::FreeSiftData(siftData2);

  // User is responsible for managing the streams
  if (useStreams) {
    safeCall(cudaStreamDestroy(stream1));
    safeCall(cudaStreamDestroy(stream2));
  }

  std::cout << "SIFT Correspondences saved to 'data/limg_pts.pgm'." << std::endl;
  // Display the results
  {
    // Displaying `limg` directly is not ideal only in that the pgm conversion will not
    // have taken place.  Simply re-read it and display that.
    cv::Mat img;
    img = cv::imread("data/limg_pts.pgm", cv::IMREAD_COLOR);
    // display the image
    cv::imshow("SIFT Correspondences", img);
    cv::waitKey(0);
  }

  return 0;
}

void MatchAll(cudaSift::SiftData &siftData1, cudaSift::SiftData &siftData2, float *homography)
{
#ifdef MANAGEDMEM
  cudaSift::SiftPoint *sift1 = siftData1.m_data;
  cudaSift::SiftPoint *sift2 = siftData2.m_data;
#else
  cudaSift::SiftPoint *sift1 = siftData1.h_data;
  cudaSift::SiftPoint *sift2 = siftData2.h_data;
#endif
  int numPts1 = siftData1.numPts;
  int numPts2 = siftData2.numPts;
  int numFound = 0;
  for (int i=0;i<numPts1;i++) {
    float *data1 = sift1[i].data;
    std::cout << i << ":" << sift1[i].scale << ":" << (int)sift1[i].orientation << std::endl;
    bool found = false;
    for (int j=0;j<numPts2;j++) {
      float *data2 = sift2[j].data;
      float sum = 0.0f;
      for (int k=0;k<128;k++)
	sum += data1[k]*data2[k];
      float den = homography[6]*sift1[i].xpos + homography[7]*sift1[i].ypos + homography[8];
      float dx = (homography[0]*sift1[i].xpos + homography[1]*sift1[i].ypos + homography[2]) / den - sift2[j].xpos;
      float dy = (homography[3]*sift1[i].xpos + homography[4]*sift1[i].ypos + homography[5]) / den - sift2[j].ypos;
      float err = dx*dx + dy*dy;
      if (err<100.0f)
	found = true;
      if (err<100.0f || j==sift1[i].match) {
	if (j==sift1[i].match && err<100.0f)
	  std::cout << " *";
	else if (j==sift1[i].match)
	  std::cout << " -";
	else if (err<100.0f)
	  std::cout << " +";
	else
	  std::cout << "  ";
	std::cout << j << ":" << sum << ":" << (int)sqrt(err) << ":" << sift2[j].scale << ":" << (int)sift2[j].orientation << std::endl;
      }
    }
    std::cout << std::endl;
    if (found)
      numFound++;
  }
  std::cout << "Number of founds: " << numFound << std::endl;
}

void PrintMatchData(cudaSift::SiftData &siftData1, cudaSift::SiftData &siftData2, cudaSift::Image &img)
{
  int numPts = siftData1.numPts;
#ifdef MANAGEDMEM
  cudaSift::SiftPoint *sift1 = siftData1.m_data;
  cudaSift::SiftPoint *sift2 = siftData2.m_data;
#else
  cudaSift::SiftPoint *sift1 = siftData1.h_data;
  cudaSift::SiftPoint *sift2 = siftData2.h_data;
#endif
  float *h_img = img.h_data;
  int w = img.width;
  int h = img.height;
  std::cout << std::setprecision(3);
  for (int j=0;j<numPts;j++) {
    int k = sift1[j].match;
    if (sift1[j].match_error<5) {
      float dx = sift2[k].xpos - sift1[j].xpos;
      float dy = sift2[k].ypos - sift1[j].ypos;
#if 0
      if (false && sift1[j].xpos>550 && sift1[j].xpos<600) {
	std::cout << "pos1=(" << (int)sift1[j].xpos << "," << (int)sift1[j].ypos << ") ";
	std::cout << j << ": " << "score=" << sift1[j].score << "  ambiguity=" << sift1[j].ambiguity << "  match=" << k << "  ";
	std::cout << "scale=" << sift1[j].scale << "  ";
	std::cout << "error=" << (int)sift1[j].match_error << "  ";
	std::cout << "orient=" << (int)sift1[j].orientation << "," << (int)sift2[k].orientation << "  ";
	std::cout << " delta=(" << (int)dx << "," << (int)dy << ")" << std::endl;
      }
#endif
      int len = (int)(fabs(dx)>fabs(dy) ? fabs(dx) : fabs(dy));
      for (int l=0;l<len;l++) {
	int x = (int)(sift1[j].xpos + dx*l/len);
	int y = (int)(sift1[j].ypos + dy*l/len);
	h_img[y*w+x] = 255.0f;
      }
    }
    int x = (int)(sift1[j].xpos+0.5);
    int y = (int)(sift1[j].ypos+0.5);
    int s = std::min(x, std::min(y, std::min(w-x-2, std::min(h-y-2, (int)(1.41*sift1[j].scale)))));
    int p = y*w + x;
    p += (w+1);
    for (int k=0;k<s;k++)
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] = h_img[p+k*w] = 0.0f;
    p -= (w+1);
    for (int k=0;k<s;k++)
      h_img[p-k] = h_img[p+k] = h_img[p-k*w] =h_img[p+k*w] = 255.0f;
  }
  std::cout << std::setprecision(6);
}



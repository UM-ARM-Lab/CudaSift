#ifndef CUDA_SIFT_SIFT_H
#define CUDA_SIFT_SIFT_H

#include <cudaSift/image.h>

namespace cudaSift {

typedef struct {
  float xpos;
  float ypos;
  float scale;
  float sharpness;
  float edgeness;
  float orientation;
  float score;
  float ambiguity;
  int match;
  float match_xpos;
  float match_ypos;
  float match_error;
  float subsampling;
  float empty[3];
  float data[128];
} SiftPoint;

// original definitions
// __constant__ float d_Threshold[2];
// __constant__ float d_Scales[8], d_Factor;
// __constant__ float d_EdgeLimit;
// __constant__ int d_MaxNumPoints;

// __device__ unsigned int d_PointCounter[1];
// __constant__ float d_Kernel1[5];
// __constant__ float d_Kernel2[12*16];

#define KPARAMS_THRESHOLD_SIZE 2
#define KENREL_THRESHOLD_SIZE_BYTES (sizeof(float)*KPARAMS_THRESHOLD_SIZE)

#define KPARAMS_SCALES_SIZE 8
#define KPARAMS_SCALES_SIZE_BYTES (sizeof(float)*KPARAMS_SCALES_SIZE)

#define KPARAMS_FACTOR_SIZE_BYTES (sizeof(float))

#define KPARAMS_EDGE_LIMIT_SIZE_BYTES (sizeof(float))

#define KPARAMS_MAX_NUM_POINTS_SIZE_BYTES (sizeof(int))

#define KPARAMS_POINT_COUNTER_SIZE 1
#define KPARAMS_POINT_COUNTER_SIZE_BYTES (sizeof(unsigned int)*KPARAMS_POINT_COUNTER_SIZE)

#define KPARAMS_KERNEL_1_SIZE 5
#define KPARAMS_KERNEL_1_SIZE_BYTES (sizeof(float)*KPARAMS_KERNEL_1_SIZE)

#define KPARAMS_KERNEL_2_SIZE (12*16)
#define KPARAMS_KERNEL_2_SIZE_BYTES (sizeof(float)*KPARAMS_KERNEL_2_SIZE)

typedef struct {
  float d_Threshold[KPARAMS_THRESHOLD_SIZE];
  float d_Scales[KPARAMS_SCALES_SIZE], d_Factor;
  float d_EdgeLimit;
  int d_MaxNumPoints;

  unsigned int d_PointCounter[KPARAMS_POINT_COUNTER_SIZE];
  float d_Kernel1[KPARAMS_KERNEL_1_SIZE];
  float d_Kernel2[KPARAMS_KERNEL_2_SIZE];
} SiftKernelParams;

typedef struct {
  int numPts;         // Number of available Sift points
  int maxPts;         // Number of allocated Sift points
#ifdef MANAGEDMEM
  SiftPoint *m_data;    // Managed data
#else
  SiftPoint *h_data;  // Host (CPU) data
  SiftPoint *d_data;  // Device (GPU) data

  SiftKernelParams *h_KernelParams;
  SiftKernelParams *d_KernelParams;

  /* Set this via InitSiftData last parameter, if unspecified will be stream 0 AKA
   * default CUDA stream
   *
   * NOTE: it is *ASSUMED* that the user is creating / destroying streams on their own,
   *       this parameter is here so that memory copies / kernel launches can be
   *       specified on this stream, but no creation / destruction will ever be
   *       performed by this library.
   */
  cudaStream_t stream;
#endif
} SiftData;

void InitCuda(int devNum = 0);
void ExtractSift(SiftData &siftData, Image &img, int numOctaves, double initBlur, float thresh, float lowestScale = 0.0f, bool scaleUp = false);
void InitSiftData(SiftData &data, int num = 1024, bool host = false, bool dev = true, cudaStream_t stream = 0);// stream 0 is default stream
void FreeSiftData(SiftData &data);
void PrintSiftData(SiftData &data);
double MatchSiftData(SiftData &data1, SiftData &data2);
double FindHomography(SiftData &data,  float *homography, int *numMatches, int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, float thresh = 5.0f);

} // namespace cudaSift

#endif // CUDA_SIFT_SIFT_H

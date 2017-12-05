//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

#ifndef CUDA_SIFT_IMAGE_H
#define CUDA_SIFT_IMAGE_H

#include <cuda_runtime.h>

namespace cudaSift {

class Image {
public:
  int width, height;
  int pitch;
  float *h_data;
  float *d_data;
  float *t_data;
  bool d_internalAlloc;
  bool h_internalAlloc;

  /* Assumed to be the same as whatever SiftData object is being used with this Image.
   * see comment on `stream` member of SiftData in `sift.h`.
   */
  cudaStream_t stream;
public:
  Image();
  // ! DANGER ! desctructor does nothing, user is responsible for calling Destroy!
  // this is different than the upstream
  ~Image() { }
  void Allocate(int width, int height, int pitch, bool withHost, float *devMem = NULL, float *hostMem = NULL, cudaStream_t stream = 0);
  double Download();
  double Readback();
  double InitTexture();
  double CopyToTexture(Image &dst, bool host);
  void Destroy();
};

int iDivUp(int a, int b);
int iDivDown(int a, int b);
int iAlignUp(int a, int b);
int iAlignDown(int a, int b);
void StartTimer(unsigned int *hTimer);
double StopTimer(unsigned int hTimer);

} // namespace cudaSift

#endif // CUDA_SIFT_IMAGE_H

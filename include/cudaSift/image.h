//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

#ifndef CUDA_SIFT_IMAGE_H
#define CUDA_SIFT_IMAGE_H

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
public:
  Image();
  ~Image();
  void Allocate(int width, int height, int pitch, bool withHost, float *devMem = NULL, float *hostMem = NULL);
  double Download();
  double Readback();
  double InitTexture();
  double CopyToTexture(Image &dst, bool host);
};

int iDivUp(int a, int b);
int iDivDown(int a, int b);
int iAlignUp(int a, int b);
int iAlignDown(int a, int b);
void StartTimer(unsigned int *hTimer);
double StopTimer(unsigned int hTimer);

} // namespace cudaSift

#endif // CUDA_SIFT_IMAGE_H

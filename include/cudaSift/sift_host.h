#ifndef CUDA_SIFT_SIFT_HOST_H
#define CUDA_SIFT_SIFT_HOST_H

#include <cudaSift/utils.h>
#include <cudaSift/image.h>

namespace cudaSift {

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

void ExtractSiftLoop(SiftData &siftData, Image &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub);
void ExtractSiftOctave(SiftData &siftData, Image &img, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp);
double ScaleDown(SiftData &siftData, Image &res, Image &src, float variance);
double ScaleUp(Image &res, Image &src);
double ComputeOrientations(cudaTextureObject_t texObj, SiftData &siftData, int fstPts, int totPts);
double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, int fstPts, int totPts, float subsampling);
double RescalePositions(SiftData &siftData, float scale);
double LowPass(SiftData &siftData, Image &res, Image &src, float scale);
double LaplaceMulti(SiftData &siftData, cudaTextureObject_t texObj, Image &baseImage, Image *results, float baseBlur, float diffScale, float initBlur);
double FindPointsMulti(Image *sources, SiftData &siftData, float thresh, float edgeLimit, float scale, float factor, float lowestScale, float subsampling);

} // namespace cudaSift

#endif // CUDA_SIFT_SIFT_HOST_H

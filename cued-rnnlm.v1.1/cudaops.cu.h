#ifndef _CUDAOPS_CU_H__
#define _CUDAOPS_CU_H__
#include "DataType.h"

extern "C" void cusigmoid (real *us, int nrows, int ncols);
extern "C" void cutanh (real *us, int nrows, int ncols);
extern "C" void cudropout (float *us, float *dropoutmask, float dropoutrate, bool evalmode, int nrows, int ncols);
extern "C" void cugendropoutmask (float *us, float dropoutrate, int nrows, int ncols);
extern "C" void cugenEmbeddropoutmask (float *us, float dropoutrate, int nrows, int ncols);
extern "C" void cugenvardropoutmask (float *us, int mbidx, float dropoutrate, int nrows, int ncols);
extern "C" void curelu (real *us, int nrows, int ncols, float ratio);
extern "C" void cugradcutoff (real *us, int nrows, int ncols, real gradcutoff);
extern "C" void cucpyfromGPU (void *dst, void *src, size_t sz);
extern "C" void * cucalloc (size_t sz);
extern "C" void cucpytoGpu (void *dst, void *src, size_t sz);
extern "C" void cucpyInGpu (void *dst, void *src, size_t sz);
extern "C" void cufree (void *ptr);
extern "C" void cumemset (void *ptr, int value, size_t sz);
extern "C" void cuassign (real*ptr, real value);
extern "C" void cuassignneu0ac (real *dev_data, int *prevwords, size_t nrow, size_t mb, real v);
extern "C" void cuassigncolumn (real *ptr, int icol, real value, size_t nrows);
extern "C" void cuassignmatvalue (real *ptr, size_t nrows, size_t ncols, real value);
extern "C" void initcuHandle ();
extern "C" void initcuRandstates ();
extern "C" void matXmat (bool transA, bool transB, size_t m, size_t n, size_t k, real *dataA, size_t lda, real *dataB, size_t ldb, real *dataC, size_t ldc, real alpha = 1.0, real beta = 0.0);
extern "C" void neu1addneu0words (int *prevwords, real *ac1, real *layer0, size_t layer0_size, size_t layer1_size, size_t mb);
// extern "C" void cusoftmax(real *us, size_t nrows, size_t mb);
extern "C" void cusoftmax(real *us, size_t nrows, size_t mb, real *lognorms);
extern "C" void assignsubmat (size_t i0, size_t j0, size_t nr, size_t nc, size_t width, real *src, real *tgt);
extern "C" void multiplyacsigmoid (real *er, real *ac, int nrows, int ncols);
extern "C" void multiplyactanh (real *er, real *ac, int nrows, int ncols);
extern "C" void multiplyacrelu (real *er, real *ac, int nrow, int ncol, float ratio);
extern "C" void cudotMultiply (real *us, real *other, int nrow, int ncol);
extern "C" void cucalHiddenacGRU (real *us, real *x, real *h, real *z, int nrow, int ncol);
extern "C" void cumultiplyScalar (real *us, float v, real nrow, int ncol);
extern "C" void cuaddScalar (real *us, float v, real nrow, int ncol);
extern "C" void addmatrix (real *us, real *other, size_t nelem);
extern "C" void addProductMat (real *us, real *other1, real *other2, size_t nelem);
extern "C" void adddotMultiplyMat (real *us, real *peelhole, real *c, size_t nr, size_t nc);
extern "C" void addOneMinusMatrix (real *us, real *other, size_t nelem);
extern "C" void subtractmatrix (real *us, real *other, size_t nelem);
extern "C" void cuaddgrad (real *us, real *other, float alpha, float l2reg, size_t nr, size_t nc);
extern "C" void cuaddadagrad (real *us, real *dU, real *accsdU, float alpha, float l2reg,  size_t nr, size_t nc);
extern "C" void cuaddsquaregrad (real *us, real *other, float gamma, float beta, size_t nr, size_t nc);
extern "C" void cuaddpeepholegrad (real *us, real *other, float alpha, float l2reg, size_t nr, size_t nc);
extern "C" void updatelayer0_wordweight (real *layer0, real *er, int *words, int nrows, int ncols, int mb, real alpha, real beta=0.0);
extern "C" void addgrad_word_v1 (real *layer0_word, real *thisgrad, int *prevwords, size_t nr, size_t nc, size_t mb);
extern "C" void addsubmatrix_v1 (real *us, real *other, size_t i0, size_t j0, size_t nr, size_t nc, size_t colwidth_us, size_t colwidth_other, real alpha = 1.0, real beta = 0.0);
extern "C" void assignsubmatrix_v1 (real *us, real *other, size_t i0, size_t j0, size_t nr, size_t nc, size_t colwidth_us, size_t colwidth_other);
extern "C" void cufetchwordprobs (real *dev_data, size_t nrow, size_t ncol, int *dev_curwords, real *dev_wordprobs, int fulldict_size);
extern "C" void cucalerronoutputlayer (real *us, real *ac, int *words, size_t rowwidth, size_t colwidth);
extern "C" void cucalerronoutputlayer_vr (real *us, real *ac, int *words, size_t rowwidth, size_t colwidth, real *lognorms, real vrpenalty);
extern "C" void GPUsynchronizewithCPU ();
extern "C" void cugetsubmatrix (real *us, real *other, size_t i0, size_t j0, size_t nr, size_t nc, size_t rowwidth_other, size_t colwidth_other);
extern "C" void cuforwardWordlayer (float *weights, float *srcac, float *tgtac, int *curclass, int *classinfo, int l1, int l2, int mbsize);
extern "C" void cusoftmaxWordlayer (float *ac, int *curclass, int *classinfo, int lN, int mbsize);
extern "C" void cucalerronWordlayer (float *er, float *ac, int *curclass, int *curwords, int *classinfo, int lN, int mbsize);
extern "C" void cubperWordlayer (float *weights, float *srcer, float *tgter, int *curclass, int *classinfo, int l1, int l2, int mbsize, float alpha, float beta);
extern "C" void cubpupdateWordlayer (float *ac, float *er, float *weights, int *curclass, int *classinfo, int l1, int l2, int mbsize, float alpha, float beta);


// cu helper function
extern "C" size_t numdevices ();
extern "C" void dumpdeviceInfo ();
extern "C" void setDeviceid (size_t i);
extern "C" void sample_v1 (float *data, int nr, int nc, int *samples, float *randv);

// NCE training
extern "C" void cuaddgrad_NCE (float *us, float *gradwgt, int *targetsample, int ntargetsample, int *ncesample, int nncesample, float alpha, int nrow, int ncol);
extern "C" void cumulScalar (float *us, int nrow, int ncol, float gradient_cutoff);
extern "C" void cucalerronOutputLayer (float *er, float *ac, float *log_noise, int *curwords, int *targetsample, int *ncesample, int *ncesamplecnt, int *mbid2arrid, int ntargetsample, int nncesample, int nrow, int mbsize);
extern "C" void cucalerronOutputLayer_oldversion (float *er, float *ac, float *er_mask, float *log_noise, int *targetsample, int *ncesample, int *ncesamplecnt, int *mbid2arrid, int ntargetsample, int nncesample, int nrow, int mbsize);
extern "C" void cucopyOutputWgtsforNCE(float *srcwgt, float *dstwgt, int *sample, int nsample, int nrow, int spos);
extern "C" void cucalnorm2 (float *dev_data, int num, int minibatch, float *norm2);

#endif

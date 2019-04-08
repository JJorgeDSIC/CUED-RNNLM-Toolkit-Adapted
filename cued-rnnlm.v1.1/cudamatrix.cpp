#include "cudamatrix.h"

// C = alpha * A * B + beta * C
void cumatrixXmatrix (matrix *A, matrix *B, matrix *C, bool transA, bool transB, real alpha /* = 1.0 */, real beta /* = 0.0 */, int Cbias  /* =0 */)
{
    const size_t m = transA ? A->cols() : A->rows();
    const size_t n = transB ? B->rows() : B->cols();
    const size_t k = transA ? A->rows() : A->cols();
    const size_t lda = A->rows();
    const size_t ldb = B->rows();
    const size_t ldc = C->rows();
    matXmat (transA, transB,  m, n, k, A->getdevdataptr(), lda, B->getdevdataptr(), ldb, C->getdevdataptr() + Cbias, ldc, alpha, beta);
    // C->fetch();
}

// A: layer0    B: neu0_ac      C: neu1_ac
void cumatrixXmatrix_fw (matrix *A, matrix *B, matrix *C, bool transA, bool transB, real alpha /* = 1.0 */, real beta /* = 0.0 */, int Bbias  /* =0 */, int Cbias, int chunksize)
{
    const size_t m = transA ? A->cols() : A->rows();
    const size_t n = transB ? B->rows() : B->cols() / chunksize;
    const size_t k = transA ? A->rows() : A->cols();
    const size_t lda = A->rows();
    const size_t ldb = B->rows();
    const size_t ldc = C->rows();
    matXmat (transA, transB,  m, n, k, A->getdevdataptr(), lda, B->getdevdataptr()+Bbias, ldb, C->getdevdataptr() + Cbias, ldc, alpha, beta);
    // C->fetch();
}

void bperWordlayer (matrix *layers, matrix *srcer, matrix *tgter, int *curclass, int *classinfo, float alpha, float beta)
{
    assert (srcer->cols() == tgter->cols());
    int l1 = tgter->rows();
    int l2 = srcer->rows();
    int mbsize = tgter->cols();
    cubperWordlayer (layers->getdevdataptr(), srcer->getdevdataptr(), tgter->getdevdataptr(), curclass, classinfo, l1, l2, mbsize, alpha, beta);
}

void bpupdateWordlayer (matrix *ac, matrix *er, matrix *layers, int *curclass, int *classinfo, float alpha, float beta)
{
    assert (ac->cols() == er->cols());
    int l1 = ac->rows();
    int l2 = er->rows();
    int mbsize = ac->cols();
    cubpupdateWordlayer (ac->getdevdataptr(), er->getdevdataptr(), layers->getdevdataptr(), curclass, classinfo, l1, l2, mbsize, alpha, beta);
}

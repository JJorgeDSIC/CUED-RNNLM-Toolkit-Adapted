#ifndef _CUDAMATRIX_H__
#define _CUDAMATRIX_H__
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "cudaops.cu.h"
#include <string.h>
#include <math.h>
#include "helper.h"
#include "head.h"

// current code should guranttee after call the function, GPU and CPU has the same value in the corresponding postion.
class matrix
{
private:
    real* host_data;
    real* dev_data;
    size_t nrows;
    size_t ncols;
    size_t size;
public:
    matrix ():host_data(NULL), dev_data(NULL), nrows(0), ncols(0)
    {}
    matrix (size_t nr, size_t nc, bool allocgpu=true)
    {
        nrows = nr;
        ncols = nc;
        size = sizeof(real) * ncols * nrows;
        host_data = (real *) malloc (size);
        if(allocgpu)
        {
            dev_data = (real *) cucalloc (size);
        }
        else
        {
            dev_data = NULL;
        }
    }
    ~matrix ()
    {
        if (dev_data)
        {
            cufree (dev_data);
            dev_data = NULL;
        }
        if (host_data)
        {
            free (host_data);
            host_data = NULL;
        }
    }
    size_t Sizeof ()
    {
        return (nrows * ncols * sizeof(real));
    }
    size_t nelem ()
    {
        return (nrows * ncols);
    }
    // copy all data from CPU to GPU
    void assign ()
    {
        cucpytoGpu (dev_data, host_data, Sizeof());
    }

    // assign value on both CPU and GPU for elem[i,j]
    void assignvalue(size_t i,size_t j, real v)
    {
        assigndevvalue (i,j, v);
        // assignhostvalue (i,j, v);
    }
    // assign value on GPU
    void assigndevvalue (size_t i, size_t j, real v)
    {
        cuassign (&dev_data[i+j*nrows], v);
    }
    // assign value in one column of matrix on GPU
    void assigndevcolumnvalue (size_t j, real v)
    {
        cuassigncolumn (dev_data, j, v, nrows);
    }
    // asign value on CPU
    void assignhostvalue (size_t i, size_t j, real v)
    {
        host_data[i+j*nrows] = v;
    }
    real addhostvalue (size_t i, size_t j, real v)
    {
        return (host_data[i+j*nrows] += v);
    }

    void assignneu0ac (int *prevwords, size_t mb, real v)
    {
        cuassignneu0ac (dev_data, prevwords, nrows, mb, v);
    }

    // copy all data from GPU to CPU
    void fetch ()
    {
        cucpyfromGPU (host_data, dev_data, Sizeof());
    }

    real fetchvalue (size_t i, size_t j)
    {
        assert ((i >= 0) && (i < nrows) && (j >= 0) && (j < ncols));
        // verify the value in CPU and GPU is the same
        real value = host_data[i+j*nrows];      // value on CPU
        real devv;                              // value on GPU
        cucpyfromGPU (&devv, &dev_data[i+j*nrows], sizeof(real));
#if 0
        if (fabs(devv -value) > 1e-7)
        {
            printf ("ERROR: fetchvalue: value on GPU(%f) and CPU(%f) are not consistent!\n", devv, value);
            exit (0);
        }
#endif
        return devv;
    }
    real fetchhostvalue (size_t i, size_t j)
    {
        return host_data[i+j*nrows];
    }


    // get the word probabilities from top layer after forward, used for testNet and testNbest
    void fetchwordprobs (int *dev_curwords, size_t mb, real *wordprobs, int fulldict_size)
    {
        assert (mb == ncols);
        real *dev_wordprobs = (real *) cucalloc (sizeof(real)*ncols);

        cufetchwordprobs (dev_data, nrows, ncols, dev_curwords, dev_wordprobs, fulldict_size);
        // copy data from GPU to CPU
        cucpyfromGPU (wordprobs, dev_wordprobs, sizeof(real)*ncols);
        cufree (dev_wordprobs);
    }

    // ensure the value in GPU and CPU is the same.
    void checkCPUandGPUmem ()
    {
        real devv, hostv;
        for (int i=0; i<nrows; i++)
        {
            for (int j=0; j<ncols; j++)
            {
                cucpyfromGPU (&devv, &dev_data[i+j*nrows], sizeof(real));
                hostv = host_data[i+j*nrows];
                // if (abs(devv-hostv) > 1e-7)
                if (devv != hostv)
                {
                    printf ("ERROR: fetchvalue: value on GPU and CPU are not consistent at (%d,%d) elem. %f v.s. %f!\n", i, j, devv, hostv);
                    exit (0);
                }
            }
        }
    }

    void setnrows (size_t nr)
    {
        nrows = nr;
    }
    void setncols (size_t nc)
    {
        ncols = nc;
    }
    size_t rows ()
    {
        return nrows;
    }
    size_t cols ()
    {
        return ncols;
    }
    void freemem ()
    {
        free (host_data);
        cufree (dev_data);
        ncols = 0;
        nrows = 0;
        size = 0;
    }
    real& operator() (int i, int j) const
    {
        assert ((i >= 0) && (i < nrows) && (j >= 0) && (j < ncols));
        return host_data[i + j*nrows];
    }
    const real& operator() (int i, int j)
    {
        assert ((i >= 0) && (i < nrows) && (j >= 0) && (j < ncols));
        return host_data[i + j*nrows];
    }
    real* getdevdataptr ()
    {
        return dev_data;
    }
    real *getdevdataptr (int i, int j)
    {
        return &dev_data[i+j*nrows];
    }
    real* gethostdataptr ()
    {
        return host_data;
    }
    real *gethostdataptr(int i, int j)
    {
        return &host_data[i+j*nrows];
    }
    void assign (size_t i0, size_t j0, size_t nr, size_t nc, matrix* &other)
    {
        if (other != NULL)
        {
            if (other->Sizeof() != 0)
            {
                other->freemem();
            }
        }
        other = new matrix (nr, nc);
        // copy to GPU side first, then copy back to CPU side
        assignsubmat (i0, j0, nr, nc, nrows, host_data, other->getdevdataptr());
        other->fetch();
    }


    void getdevsubmatrix (matrix *other, size_t i0, size_t j0, size_t nr, size_t nc)
    {
        cugetsubmatrix (dev_data, other->getdevdataptr(), i0, j0, nr, nc, other->rows(), other->cols());
        setncols (nc);
        setnrows (nr);
    }
    // assign matrix from element from another matrix (other), both GPU and CPU
    // TODO: write GPU to GPU version, make all process free of CPU
    void assign (matrix *other)
    {
        assert (rows() == other->rows());
        assert (cols() == other->cols());
#if 0
        other->fetch();
        cucpytoGpu (dev_data, other->gethostdataptr(), Sizeof());
        fetch();
#else
        cucpyInGpu (dev_data, other->getdevdataptr(), Sizeof());
#endif
    }

    void hostassign (matrix *other)
    {
        assert (rows() == other->rows());
        assert (cols() == other->cols());
        memcpy (host_data, other->gethostdataptr(), Sizeof());
    }

    // assign the submatrix ([i0, j0] to [i0+nr-1, j0+nc-1]) using value from another matrx (other), both GPU and CPU
    void assignsubmatrix (matrix *other, int i0, int j0, int nr, int nc)
    {
        assert (nr == other->rows());
        assert (nc == other->cols());
        assignsubmatrix_v1 (dev_data, other->getdevdataptr(), i0, j0, nr, nc, rows(), other->rows());
        fetch();
    }

    // initialize all element (both GPU and CPU) in matrx with v
    void initmatrix (int v = 0)
    {
        cumemset (dev_data, v, Sizeof());
        // memset (host_data, v, Sizeof());
    }

    void hostinitmatrix (int v = 0)
    {
        memset (host_data, v, Sizeof());
    }

    void assignmatvalue (real v)
    {
        cuassignmatvalue (dev_data, nrows, ncols, v);
    }
    // sigmoid on all elements seperately
    void sigmoid ()
    {
        cusigmoid (dev_data, nrows, ncols);
        // fetch();
    }

    void tanh ()
    {
        cutanh (dev_data, nrows, ncols);
    }

    void sigmoid_forchunk (int chunkiter, int chunksize)
    {
        int mbsize = ncols / chunksize;
        cusigmoid (dev_data+chunkiter*mbsize, nrows, mbsize);
        // fetch();
    }

    void relu (float ratio)
    {
        curelu (dev_data, nrows, ncols, ratio);
    }

    void relu_forchunk (float ratio, int chunkiter, int chunksize)
    {
        int mbsize = ncols / chunksize;
        curelu (dev_data+chunkiter*mbsize, nrows, mbsize, ratio);
    }

    void gendropoutmask (float dropoutrate)
    {
        cugendropoutmask (dev_data, dropoutrate, nrows, ncols);
    }

    void genEmbeddropoutmask (float dropoutrate)
    {
        cugenEmbeddropoutmask (dev_data, dropoutrate, nrows, ncols);
    }

    void genvardropoutmask (int mbidx, float dropoutrate)
    {
        cugenvardropoutmask (dev_data, mbidx, dropoutrate, nrows, ncols);
    }

    void dropout (matrix *dropoutmask, float dropoutrate, bool evalmode)
    {
        cudropout (dev_data, dropoutmask->getdevdataptr(), dropoutrate, evalmode, nrows, ncols);
    }

    void hostrelu (float ratio)
    {
        assert (ncols == 1);
        for (int i=0; i<nrows; i++)
        {
            if (host_data[i] > 0)
            {
                host_data[i] *= ratio;
            }
            else
            {
                host_data[i] = 0;
            }
        }
    }

    void hostsigmoid()
    {
        assert (ncols == 1);
        for (int i=0; i<nrows; i++)
        {
            host_data[i] = 1/(1 + exp(-host_data[i]));
        }
    }

    void hosttanh ()
    {
        assert (ncols == 1);
        for (int i=0; i<nrows; i++)
        {
            if (host_data[i] > 50) host_data[i] = 50;
            host_data[i] = (exp(2*host_data[i])-1)/(exp(2*host_data[i])+1);
        }
    }


    void hostsoftmax()
    {
        int a, maxi;
        float v, norm, maxv = 1e-8;
        assert (ncols == 1);
        maxv = -1e10;
#ifdef SMOOTHBIRNNLM
        for (a=0; a<nrows; a++)
        {
            host_data[a] = host_data[a]*SMOOTHBIRNNLM;
        }
#endif
        for (a=0; a<nrows; a++)
        {
            v = host_data[a];
            if (v > maxv)
            {
                maxv = v;
                maxi = a;
            }
        }
        norm = 0;
        for (a=0; a<nrows; a++)
        {
            v = host_data[a] - maxv;
            host_data[a] = exp(v);
            norm += host_data[a];
        }
        for (a=0; a<nrows; a++)
        {
            v = host_data[a] / norm;
            host_data[a] = v;
        }
    }

    void hostpartsoftmax(int swordid, int ewordid)
    {
        int a, maxi;
        float v, norm, maxv = 1e-8;
        assert (ncols == 1);
        maxv = 1e-10;
        for (a=swordid; a<=ewordid; a++)
        {
            v = host_data[a];
            if (v > maxv)
            {
                maxv = v;
                maxi = a;
            }
        }
        norm = 0;
        for (a=swordid; a<=ewordid; a++)
        {
            v = host_data[a] - maxv;
            host_data[a] = exp(v);
            norm += host_data[a];
        }
        for (a=swordid; a<=ewordid; a++)
        {
            v = host_data[a] / norm;
            host_data[a] = v;
        }
    }
    // softmax function, TODO speed up.
#if 0
    void softmax ()
    {
        cusoftmax (dev_data, nrows, ncols);
        // fetch();
    }
#endif

    void softmax (matrix *lognorms)
    {
        float *dev_lognorms = lognorms->getdevdataptr();
        cusoftmax (dev_data, nrows, ncols, dev_lognorms);
    }


    void gradcutoff (real gradient_cutoff)
    {
        cugradcutoff (dev_data, nrows, ncols, gradient_cutoff);
    }

    // calculate error signal on the output layer (with softmax function)
    // er = (\delta(i,j) - ac), where i is the target, j is the index of nodes in output layer.
    // ac: ac
    // start_array: starts id for each sample (on GPU)
    // cn_array:    number of words need to calculate error in each sample (on GPU)
    void calerronoutputlayer(matrix *ac, int *words)
    {
        cucalerronoutputlayer (dev_data, ac->getdevdataptr(), words, ac->rows(), ac->cols());
    }

    void calerronoutputlayer_vr(matrix *ac, int *words, matrix *lognorms, float vrpenalty)
    {
        cucalerronoutputlayer_vr (dev_data, ac->getdevdataptr(), words, ac->rows(), ac->cols(), lognorms->getdevdataptr(), vrpenalty);
    }

    void hostadddotMultiply (matrix *c, matrix *p)
    {
        float *host_c = c->gethostdataptr ();
        float *host_p = p->gethostdataptr ();
        foreach_coord (i, j, c)
        {
            host_data[i+j*nrows] += host_c[i+j*nrows] * host_p[i+j*nrows];
        }
    }

    void hostadd (matrix *other)
    {
        if (rows() != other->rows() || cols() != other->cols())
        {
            printf ("error: hostadd: the size is different!\n");
            exit (0);
        }
        float *host_other = other->gethostdataptr();
        foreach_coord (i, j, other)
        {
            host_data[i+j*nrows] += host_other[i+j*nrows];
        }
    }

    void hostdotMultiply (matrix *other)
    {
        if (rows() != other->rows() || cols() != other->cols())
        {
            printf ("error: hostdotMultiply: the size is different!\n");
            exit (0);
        }
        float *host_other = other->gethostdataptr();
        foreach_coord (i, j, other)
        {
            host_data[i+j*nrows] *= host_other[i+j*nrows];
        }
    }

    void hostcalHiddenacGRU (matrix *h, matrix *z)
    {
        if (rows() != h->rows() || cols() != h->cols() || rows() != z->rows() || cols() != z->cols())
        {
            printf ("error: hostcalHiddenacGRU: the size is different!\n");
            exit (0);
        }
        float *host_h_ = h->gethostdataptr();
        float *host_z = z->gethostdataptr();
        foreach_coord (i, j, h)
        {
            float zvalue = host_z[i+j*nrows];
            float h_value = host_h_[i+j*nrows];
            float hvalue = host_data[i+j*nrows];
            host_data[i+j*nrows] = hvalue*(1-zvalue) + h_value*zvalue;
        }
    }

    void dotMultiply (matrix *other)
    {
        if (rows() != other->rows() || cols() != other->cols())
        {
            printf ("error: dotmultiply: the size is different!\n");
            exit (0);
        }
        cudotMultiply (dev_data, other->getdevdataptr(), rows(), cols());
    }

    void calHiddenacGRU (matrix *x, matrix *h, matrix *z)
    {
        if (rows() != h->rows() || cols() != h->cols() || rows() != z->rows() || cols() != z->cols())
        {
            printf ("error: calHiddenacGRU: the size is different!\n");
            exit (0);
        }
        cucalHiddenacGRU (dev_data, x->getdevdataptr(), h->getdevdataptr(), z->getdevdataptr(), rows(), cols());

    }

    void multiplyScalar (float v)
    {
        cumultiplyScalar (dev_data, v, rows(), cols());
    }

    void addScalar (float v)
    {
        cuaddScalar (dev_data, v, rows(), cols());
    }

    // used when calculating the gradient in back propogation (through time)
    // er = er * ac * (1-ac)
    void multiplysigmoid (matrix *ac)
    {
        if (rows() != ac->rows() || cols() != ac->cols())
        {
            printf ("ERROR: multisigmoid: the size is different!\n");
            exit (0);
        }
        multiplyacsigmoid (dev_data, ac->getdevdataptr(), rows(), cols());
        // fetch();
    }

    void multiplytanh (matrix *ac)
    {
        if (rows() != ac->rows() || cols() != ac->cols())
        {
            printf ("ERROR: multisigmoid: the size is different!\n");
            exit (0);
        }
        multiplyactanh (dev_data, ac->getdevdataptr(), rows(), cols());
        // fetch();
    }

    void multiplyrelue (matrix *ac, float ratio)
    {
        if (rows() != ac->rows() || cols() != ac->cols())
        {
            printf ("ERROR: multisigmoid: the size is different!\n");
            exit (0);
        }
        multiplyacrelu (dev_data, ac->getdevdataptr(), rows(), cols(), ratio);
        // fetch();
    }
    // update the weight matrix connecting input word and hidden layer
    void updatelayer0_word (matrix *neu1_er, int *words, real alpha, real beta = 0.0)
    {
        updatelayer0_wordweight (dev_data, neu1_er->getdevdataptr(), words, rows(), cols(), neu1_er->cols(), alpha, beta);
    }


    // add with other matrix element by element
    // this += other
    void add (matrix *other)
    {
        assert (rows() == other->rows());
        assert (cols() == other->cols());
        addmatrix (dev_data, other->getdevdataptr(), rows()*cols());
        // fetch();
    }

    void addProduct (matrix *other1, matrix *other2)
    {
        assert (rows() == other1->rows());
        assert (cols() == other1->cols());
        assert (rows() == other2->rows());
        assert (cols() == other2->cols());
        addProductMat (dev_data, other1->getdevdataptr(), other2->getdevdataptr(), rows()*cols());
    }

    void  adddotMultiply (matrix *peelhole, matrix *c)
    {
        assert (rows() == c->rows());
        assert (cols() == c->cols());
        assert (peelhole->rows() == rows());
        assert (peelhole->cols() == 1);
        adddotMultiplyMat (dev_data, peelhole->getdevdataptr(), c->getdevdataptr(), rows(), cols());
    }

    void addOneMinus (matrix *other)
    {
        assert (rows() == other->rows());
        assert (cols() == other->cols());
        addOneMinusMatrix (dev_data, other->getdevdataptr(), rows()*cols());
        // fetch();
    }

    void subtract (matrix *other)
    {
        assert (rows() == other->rows());
        assert (cols() == other->cols());
        subtractmatrix (dev_data, other->getdevdataptr(), rows()*cols());
    }

    void addadagrad (matrix *dU, matrix *accsdU, float alpha, float l2reg)
    {
        assert (rows() == dU->rows());
        assert (cols() == dU->cols());
        assert (rows() == accsdU->rows());
        assert (cols() == accsdU->cols());
        cuaddadagrad (dev_data, dU->getdevdataptr(), accsdU->getdevdataptr(), alpha, l2reg, rows(), cols());
    }

    void addgrad (matrix *other, float alpha, float l2reg)
    {
        assert (rows() == other->rows());
        assert (cols() == other->cols());
        cuaddgrad (dev_data, other->getdevdataptr(), alpha, l2reg, rows(), cols());
    }

    void addsquaregrad (matrix *other, float gamma, float beta)
    {
        assert (rows() == other->rows());
        assert (cols() == other->cols());
        cuaddsquaregrad (dev_data, other->getdevdataptr(), gamma, beta, rows(), cols());
    }

    void addpeepholegrad (matrix *other, float alpha, float l2reg)
    {
        assert (rows() == other->rows());
        assert (cols() == 1);
        cuaddpeepholegrad (dev_data, other->getdevdataptr(), alpha, l2reg, rows(), other->cols());
    }

    void addgrad_word (matrix *gradlayer0_word, int *prevwords, int minibatch)
    {
        assert (rows() == gradlayer0_word->rows());
        assert (cols() == gradlayer0_word->cols());
        addgrad_word_v1 (dev_data, gradlayer0_word->getdevdataptr(), prevwords, rows(), cols(), minibatch);
    }

    // this = alpha * other + beta * this
    void addsubmatrix (matrix *other, int i0, int j0, int nr, int nc, real alpha=1.0, real beta = 1.0)
    {
        assert (nr == other->rows());
        assert (nc == other->cols());
        addsubmatrix_v1 (dev_data, other->getdevdataptr(), i0, j0, nr, nc, rows(), other->rows(), alpha, beta);
        // fetch();
    }

    void random(float min, float max)
    {
        int i, j;
        float v;
        for (i=0; i<nrows; i++)
        {
            for (j=0; j<ncols; j++)
            {
                v = randomv(min, max) + randomv(min,max) + randomv(min, max);
                host_data[i+j*nrows] = v;
            }
        }
        assign();
    }

    void randomidentity (float scale)
    {
        int i, j;
        float v;
        assert (nrows == ncols);
        for (i=0; i<nrows; i++)
        {
            for (j=0; j<ncols; j++)
            {
                if (i==j)
                {
                    v = scale;
                }
                else
                {
                    v = 0;
                }
                host_data[i+j*nrows] = v;
            }
        }
        assign();
    }

    void sample (int *dev_samples, float *dev_randv, int minibatch)
    {
        assert (ncols == minibatch);
        sample_v1 (dev_data, nrows, ncols, dev_samples, dev_randv);
    }

    void forwardWordlayer (matrix *srcac, matrix *tgtac, int *curclass, int *classinfo)
    {
        assert (srcac->rows() == nrows);
        assert (tgtac->rows() == ncols);
        assert (srcac->cols() == tgtac->cols());
        cuforwardWordlayer (dev_data, srcac->getdevdataptr(), tgtac->getdevdataptr(), curclass, classinfo, nrows, ncols, srcac->cols());
    }

    void softmaxWordlayer (int *curclass, int *classinfo)
    {
        cusoftmaxWordlayer (dev_data, curclass, classinfo, nrows, ncols);
    }
    void calerronWordlayer (matrix *ac, int *curclass, int *curwords, int *classinfo)
    {
        cucalerronWordlayer (dev_data, ac->getdevdataptr(), curclass, curwords, classinfo, nrows, ncols);
    }

    void copyOutputWgtsforNCE (matrix *outputlayer, int *dev_targetsample, int ntargetsample, int *dev_ncesample, int nncesample)
    {
        int nrow = outputlayer->rows();
        // layers[num_layer] to layerN_NCE (L1xL2 -> L1xN')
        cucopyOutputWgtsforNCE (outputlayer->getdevdataptr(), dev_data, dev_targetsample, ntargetsample, nrow, 0);
        cucopyOutputWgtsforNCE (outputlayer->getdevdataptr(), dev_data, dev_ncesample, nncesample, nrow, ntargetsample);
    }

    void calerronOutputLayer (matrix *neuN_ac_NCE, float *log_noise, int *curwords, int *targetsample, int *ncesample, int *ncesamplecnt, int *mbid2arrid, int ntargetsample, int nncesample)
    {
        int nrow = neuN_ac_NCE->rows();
        int mbsize = neuN_ac_NCE->cols();
        cucalerronOutputLayer (dev_data, neuN_ac_NCE->getdevdataptr(), log_noise, curwords, targetsample, ncesample, ncesamplecnt, mbid2arrid, ntargetsample, nncesample, nrow, mbsize);
    }

    void calerronOutputLayer_oldversion (matrix *neuN_ac_NCE, matrix *neuN_er_NCE_mask, float *log_noise, int *targetsample, int *ncesample, int *ncesamplecnt, int *mbid2arrid, int ntargetsample, int nncesample)
    {
        int nrow = neuN_ac_NCE->rows();
        int mbsize = neuN_ac_NCE->cols();
        cucalerronOutputLayer_oldversion (dev_data, neuN_ac_NCE->getdevdataptr(), neuN_er_NCE_mask->getdevdataptr(), log_noise, targetsample, ncesample, ncesamplecnt, mbid2arrid, ntargetsample, nncesample, nrow, mbsize);
    }

    void calnorm2 (int minibatch, float *norm2)
    {
        cucalnorm2 (dev_data, nrows*ncols, minibatch, norm2);
    }

    void mulScalar (float gradient_cutoff)
    {
        cumulScalar (dev_data, nrows, ncols, gradient_cutoff);
    }

    void L2norm (float gradient_cutoff, float *devnorm, int minibatch)
    {
        float norm;
        cucalnorm2 (dev_data, nrows*ncols, minibatch, devnorm);
        cucpyfromGPU (&norm, devnorm, sizeof(float));
        // norm = norm  / (nrows*ncols);
        if (norm > gradient_cutoff)
        {
            cumulScalar (dev_data, nrows, ncols, gradient_cutoff/norm);
        }
    }

    void addgrad_NCE (matrix *gradwgt, int *targetsample, int ntargetsample, int *ncesample, int nncesample, float alpha)
    {
        cuaddgrad_NCE (dev_data, gradwgt->getdevdataptr(), targetsample, ntargetsample, ncesample, nncesample, alpha, nrows, ncols);
    }

    void Read (FILE *fptr)
    {
        int i, j;
        float v;
        for (i=0; i<nrows; i++)
        {
            for (j=0; j<ncols; j++)
            {
                fscanf (fptr, "%f ", &v);
                assignhostvalue(i, j, v);
            }
            fscanf (fptr, "\n");
        }
        assign();
    }

    void Write (FILE *fptr)
    {
        fetch ();
        int i, j;
        for (i=0; i<nrows; i++)
        {
            for (j=0; j<ncols; j++)
            {
                fprintf (fptr, "%.8f ", fetchhostvalue(i, j));
            }
            fprintf (fptr, "\n");
        }
    }

    void dump ()
    {
        fetch ();
        int i, j;
        for (i=0; i<nrows; i++)
        {
            for (j=0; j<ncols; j++)
            {
                printf ("%.4f ", fetchhostvalue(i, j));
            }
            printf ("\n");
        }
        printf ("\n");
    }

};



void cumatrixXmatrix (matrix *A, matrix *B, matrix *C, bool transA, bool transB, real alpha = 1.0, real beta = 0.0, int Cbias = 0);

// tgter = alpha*layers*srcer + beta*tgter
void bperWordlayer (matrix *layers, matrix *srcer, matrix *tgter, int *curclass, int *classinfo, float alpha, float beta);
void bpupdateWordlayer (matrix *ac, matrix *er, matrix *layers, int *curclass, int *classinfo, float alpha, float beta);
#endif

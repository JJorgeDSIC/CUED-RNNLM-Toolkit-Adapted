#ifndef _LAYER_H__
#define _LAYER_H__
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "cudaops.cu.h"
#include <string.h>
#include <math.h>
#include "helper.h"
#include "head.h"
#include "cudamatrix.h"
#include <random>
#include <algorithm>
typedef mt19937_64 rng_type;


class layer
{
public:
    size_t nrows;
    size_t ncols;
    size_t size;
    string type;
    matrix *U;
    matrix *dU;
    float  *cunorm2ptr;
    float  norm2;
    float momentum;
    float gamma;        // use for adagrad and rmsprop update
    int   lrtunemode;   // learn rate tune mode; 0: newbob 1: adagrad 2: rmsprop
    float l2reg, gradient_cutoff;
    matrix *accsdU;     // accumulated square gradient
    float dropoutrate;
    bool evalmode;
    int chunkiter, chunksize;
    int minibatch;      // used for L2norm
    vector<matrix *> dropoutmaskMat_vec;        // likely to apply dropout for every layer
    float alpha;        // learning rate
public:
    virtual void forward(matrix *neu0_ac, matrix *neu1_ac)
    {
        printf ("virtual function (forward) called!\n");
    }
    virtual void forward_nosigm(matrix *neu0_ac, matrix *neu1_ac)
    {
        printf ("virtual function (forward_nosigm) called!\n");
    }
    virtual void host_forward (matrix *neu0_ac, matrix *neu1_ac)
    {
        printf ("virtual function (host_forward) called!\n");
    }
    virtual void host_forward_nosigm (matrix *neu0_ac, matrix *neu1_ac)
    {
        printf ("virtual function (host_forward_nosigm) called!\n");
    }
    virtual void backward(matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac)
    {
        printf ("virtual function (backward) called!\n");
    }
    virtual void backward_succ(matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac)
    {
        printf ("virtual function (backward_succ) called!\n");
    }
    virtual void update (float alpha)
    {
        printf ("virtual function (update) called!\n");
    }
    virtual void Read (FILE *fptr)
    {
        printf ("virtual function (Read) called!\n");
    }
    virtual void Write (FILE *fptr)
    {
        printf ("virtual function (Write) called!\n");
    }
    virtual void getWordEmbedding (int *dev_prevwords, matrix *neu_ac)
    {
        printf ("virtual function (getWordEmbedding) called!\n");
    }
    virtual void host_getWordEmbedding (int prevword, matrix *neu_ac)
    {
        printf ("virtual function (host_getWordEmbedding) called!\n");
    }
    virtual void host_resetHiddenac ()
    {
        printf ("virtual function (host_resetHiddenac) called!\n");
    }
    virtual void setTrainCrit (int n)
    {
        printf ("virtual function (setTrainCrit) called!\n");
    }
    virtual void setnodetype (int n)
    {
        printf ("virtual function (setnodetype) called!\n");
    }
    virtual void setreluratio (float v)
    {
        printf ("virtual function (setreluratio) called!\n");
    }
    virtual void resetHiddenAc (int mbidx)
    {
        printf ("virtual function (resetHiddenAc) called!\n");
    }
    virtual void loadHiddenAc ()
    {
        printf ("virtual function (loadHiddenAc) called!\n");
    }
    virtual void saveHiddenAc ()
    {
        printf ("virtual function (saveHiddenAc) called!\n");
    }
    virtual void initHiddenAc ()
    {
        printf ("virtual function (initHiddenAc) called!\n");
    }
    virtual void resetErrVec (int mbidx)
    {
        printf ("virtual function (resetErrVec) called!\n");
    }
    virtual void initHiddenEr ()
    {
        printf ("virtual function (initHiddenEr) called!\n");
    }
    virtual void updateWordEmbedding (int *dev_prevwords, matrix *neu_er)
    {
        printf ("virtual function (updateWordEmbedding) called!\n");
    }
    virtual void calerr (matrix *neu_ac, matrix *neu_er, int *dev_curwords)
    {
        printf ("virtual function (calerr) called!\n");
    }
    virtual void setVRpenalty (float v)
    {
        printf ("virtual function (setVRpenalty) called!\n");
    }
    virtual void ComputeAccMeanVar (int *host_curwords, double &lognorm_mean, double &lognorm_var)
    {
        printf ("virtual function (ComputeAccMeanVar) called!\n");
    }
    virtual void setLognormConst (float v)
    {
        printf ("virtual function (setLognormConst) called!\n");
    }
    virtual void printHiddenAcValue (int i, int j)
    {
        printf ("virtual function (printHiddenAcValue) called!\n");
    }
    virtual void prepareNCEtrain (double *wordprob, vector<string> &outputvec)
    {
        printf ("virtual function (prepareNCEtrain) called!\n");
    }
    virtual void assignTargetSample (int *host_curwords)
    {
        printf ("virtual function (assignTargetSample) called!\n");
    }
    virtual void genNCESample ()
    {
        printf ("virtual function (genNCESample) called!\n");
    }
    virtual void allocNCEMem (int k)
    {
        printf ("virtual function (allocNCEMem) called!\n");
    }
    virtual void copyLSTMhighwayc (layer *layer0,  int chunkiter)
    {
        printf ("virtual function (copyLSTMhighwayc) called!\n");
    }
    virtual void copyLSTMhighwayc_er (layer *layer1, int chunkiter)
    {
        printf ("virtual function (copyLSTMhighwayc_er) called!\n");
    }
    virtual void host_copyLSTMhighwayc (layer *layer0)
    {
        printf ("virtual function (host_copyLSTMhighwayc) called!\n");
    }
    virtual void setLRtunemode (int lr)
    {
        printf ("virtual function (setLRtunemode) called!\n");
    }
    virtual void genvardropoutmask (int mbidx)
    {
        printf ("virtual function (genvardropoutmask) called!\n");
    }
    virtual void copyvardropoutmask ()
    {
        printf ("virtual function (copyvardropoutmask) called!\n");
    }
    virtual void ReadFeaFile (string str)
    {
        printf ("virtual function (ReadFeaFile) called!\n");
    }
    virtual void assignFeaMat ()
    {
        printf ("virtual function (assignFeaMat) called!\n");
    }
    virtual void host_assignFeaVec (int feaid)
    {
        printf ("virtual function (host_assignFeaVec) called!\n");
    }
    virtual void fillInputFeature ()
    {
        printf ("virtual function (fillInputFeature) called!\n");
    }
    virtual bool usefeainput ()
    {
        printf ("virtual function (usefeainput) called!\n");
    }
    virtual void updateFeaMat (int mbidx)
    {
        printf ("virtual function (updateFeaMat) called!\n");
    }
    virtual void setFeaIndices (int *ptr)
    {
        printf ("virtual function (setFeaIndices) called!\n");
    }
    virtual int getnumfea ()
    {
        printf ("virtual function (getnumfea) called!\n");
    }
    virtual int getdimfea ()
    {
        printf ("virtual function (getdimfea) called!\n");
    }
    virtual void allocFeaMem ()
    {
        printf ("virtual function (allocFeaMem) called!\n");
    }
    virtual void setSuccAc (matrix *ac)
    {
        printf ("virtual function (setSuccAc) called!\n");
    }
    virtual void setSuccEr (matrix *er)
    {
        printf ("virtual function (setSuccEr) called!\n");
    }
    bool isDropout ();
    void setMomentum (float v)          {momentum = v;}
    void setL2Regularization (float v)  {l2reg = v;}
    void setClipping (float v)          {gradient_cutoff = v;}
    void printGrad();
    float fetchweightvalue (int i, int j);
    float fetchgradweightvalue (int i, int j);
    void setEvalmode (bool flag) { evalmode = flag;}
    void setDropoutRate (float v) { dropoutrate = v;}
    void setChunkIter (int iter)  { chunkiter = iter;}
    void allocDropoutMask ();
    void setLearnRate (float v) { alpha = v;}
    int  rows ()        { return nrows; }
    int  cols ()        { return ncols; }
    layer (int nr, int nc, int mbsize, int cksize);
    ~layer ();
};

class inputlayer : public layer
{
private:
    ////// variable for topic feature input
    int dim_fea, num_fea;
    string feafile;
    matrix *feamatrix;
    matrix *U_fea, *dU_fea;
    matrix *ac_fea;
    vector<matrix *> ac_fea_vec;
    int *mbfeaindices, *feaindices;
    //////

    int *dev_prevwords;
    vector<matrix *> dropoutHiddenmaskMat_vec;
public:
    virtual void backward (matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac);
    virtual void Read (FILE *fptr);
    virtual void Write (FILE *fptr);

    inputlayer (int nr, int nc, int mbsize, int cksize, int dim_fea);
    ~inputlayer ();

    virtual void ReadFeaFile (string str);
    virtual void assignFeaMat ();
    virtual void host_assignFeaVec (int feaid);
    virtual void fillInputFeature ();
    virtual bool usefeainput ();
    virtual void updateFeaMat (int mbidx);
    virtual void setFeaIndices (int *ptr);
    virtual void getWordEmbedding (int *dev_prevwords, matrix *neu_ac);
    virtual void host_getWordEmbedding (int prevword, matrix *neu_ac);
    virtual void updateWordEmbedding (int *dev_prevwords, matrix *neu_er);
    virtual void update (float alpha);
    virtual void printHiddenAcValue (int i, int j)
    {
        printf ("hidden[%d, %d]=%f\n", i, j, U->fetchvalue(i, j));
    }
    virtual void setLRtunemode (int lr);
    virtual void allocFeaMem ();
    virtual int getnumfea () { return num_fea; }
    virtual int getdimfea () { return dim_fea; }
};


class outputlayer : public layer
{
private:
    int  traincrit;         // CE, VR or NCE
    // for CE training
    float vrpenalty;
    // for NCE training
    matrix *gradlayerN_NCE;
    matrix *gradlayerN_NCE_succ;
    matrix *neuN_er_NCE, *W_NCE, *dW_NCE;
    matrix *logwordnoise;
    vector<matrix *> lognormvec;
    vector<matrix *> neuN_ac_NCE_vec;
    vector<matrix *> layerN_NCE_vec;
    vector<matrix *> layerN_succ_NCE_vec;
    vector<matrix *> targetsample_vec, ncesample_vec;
    int **mbid2arrid, **dev_mbid2arrid, **ncesamplecnt, **dev_ncesamplecnt;
    int **targetsample, **ncesample, **dev_targetsample, **dev_ncesample;
    // int **mbid2arrid_vec, **ncesamplecnt_vec, **dev_mbid2arrid_vec, **dev_ncesamplecnt_vec;
    int *ntargetsample, *nncesample, *outputlayersize_NCE, *host_curwords;
    //  double *unigram, *accprob, *logunigram;
    vector<double> unigram, accprobvec, logunigram;
    float log_num_noise;
    int k,          // number of noise sample
        N,          // k+minibatch
        outOOSindex;
    rng_type* rngs;
    uniform_real_distribution<double> uniform_real_dist;
    // for VR and NCE training
    float lognorm;

    matrix *neu0_ac_succ, *neu0_er_succ, *U_succ, *dU_succ;
public:
    virtual void forward(matrix *neu0_ac, matrix *neu1_ac);
    virtual void host_forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void calerr (matrix *neu_ac, matrix *neu_er, int *dev_curwords);
    virtual void backward(matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac);
    virtual void update (float alpha);
    virtual void Read (FILE *fptr);
    virtual void Write (FILE *fptr);

    outputlayer (int nr, int nc, int minibatch, int chunksize);
    // construction function for NCE training
    outputlayer (int nr, int nc, int minibatch, int chunksize, int k);
    ~outputlayer ();
    // NCE training
    virtual void prepareNCEtrain (double *wordprob, vector<string> &outputvec);
    virtual void assignTargetSample (int *host_curwords);
    virtual void genNCESample ();
    virtual void allocNCEMem (int k);

    virtual void setTrainCrit (int n)   { traincrit = n; }
    virtual void setVRpenalty (float v) { vrpenalty = v; }
    virtual void ComputeAccMeanVar (int *host_curwords, double &lognorm_mean, double &lognorm_var);
    virtual void setLognormConst (float v) {lognorm = v;}
    virtual void setLRtunemode (int lr);
    virtual void genvardropoutmask (int mbidx);
    virtual void copyvardropoutmask ();
    virtual void setSuccAc (matrix *ac);
    virtual void setSuccEr (matrix *er);
    };


class recurrentlayer : public layer
{
private:
    matrix *accsdW;
    matrix *W, *dW, *hidden_er, *hidden_ac, *hidden_ac_last;
    vector<matrix *> hidden_ac_vec;
    int  nodetype;          // sigmoid or relu
    float reluratio;
public:
    virtual void forward(matrix *neu0_ac, matrix *neu1_ac);
    virtual void host_forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void backward(matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac);
    virtual void host_resetHiddenac ();
    virtual void update (float alpha);
    virtual void Read (FILE *fptr);
    virtual void Write (FILE *fptr);

    recurrentlayer (int nr, int nc, int minibatch, int chunksize);
    ~recurrentlayer ();

    virtual void setnodetype (int n) { nodetype = n; }
    virtual void setreluratio (float v) { reluratio = v; }
    virtual void resetHiddenAc (int mbidx);
    virtual void loadHiddenAc ();
    virtual void saveHiddenAc ();
    virtual void initHiddenAc ();
    virtual void resetErrVec (int mbidx);
    virtual void initHiddenEr ();
    virtual void printHiddenAcValue (int i, int j)
    {
        printf ("hidden[%d, %d]=%f\n", i, j, hidden_ac->fetchvalue(i, j));
    }
    virtual void setLRtunemode (int lr);
};

class feedforwardlayer : public layer
{
private:
    int  nodetype;          // sigmoid or relu
    float reluratio;
public:
    virtual void forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void forward_nosigm (matrix *neu0_ac, matrix *neu1_ac);
    virtual void host_forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void host_forward_nosigm (matrix *neu0_ac, matrix *neu1_ac);
    virtual void backward(matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac);
    virtual void backward_succ(matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac);
    virtual void update (float alpha);
    virtual void Read (FILE *fptr);
    virtual void Write (FILE *fptr);

    feedforwardlayer (int nr, int nc, int mbsize, int cksize);

    virtual void setnodetype (int n) { nodetype = n; }
    virtual void setreluratio (float v)  { reluratio = v; }
    virtual void setLRtunemode (int lr);
};

class linearlayer : public layer
{
public:
    virtual void forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void host_forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void backward(matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac);
    virtual void update (float alpha);
    virtual void Read (FILE *fptr);
    virtual void Write (FILE *fptr);
    virtual void setLRtunemode (int lr);

    linearlayer (int nr, int nc, int mbsize, int cksize);
};

class grulayer : public layer
{
protected:
    matrix *hidden_er, *hidden_ac, *hidden_ac_last;
    matrix *Wr, *Wz, *Wh, *Ur, *Uz, *Uh;
    matrix *dWr, *dWz, *dWh, *dUr, *dUz, *dUh;
    matrix *accsdWr, *accsdWz, *accsdWh, *accsdUr, *accsdUz, *accsdUh;
    matrix *r, *z, *c, *h_;
    vector<matrix *> r_vec, z_vec, c_vec, h_vec, hidden_ac_vec;
    matrix *dr, *dz, *dc, *dh_;
public:
    virtual void forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void host_forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void backward (matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac);
    virtual void host_resetHiddenac ();
    virtual void update (float alpha);
    virtual void resetErrVec (int mbidx);
    virtual void resetHiddenAc (int mbidx);
    virtual void loadHiddenAc ();
    virtual void initHiddenAc ();
    virtual void saveHiddenAc ();
    virtual void initHiddenEr ();
    virtual void Read (FILE *fptr);
    virtual void Write (FILE *fptr);
    ~grulayer ();
    grulayer (int nr, int nc, int minibatch, int chunksize);
    virtual void printHiddenAcValue (int i, int j)
    {
        printf ("Uz[0,0]=%f\n", i, j, dUz->fetchvalue(i,j));
    }
    virtual void setLRtunemode (int lr);
};

class gruhighwaylayer : public grulayer
{
private:
    matrix *g, *v, *dg, *dv, *hidden_er_highway;
    matrix *Uhw, *Whw, *dUhw, *dWhw;
    matrix *accsdUhw, *accsdWhw;
    vector <matrix *> g_vec, v_vec;
public:
    gruhighwaylayer (int nr, int nc, int minibatch, int chunksize);
    ~gruhighwaylayer ();
    virtual void forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void host_forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void backward (matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac);

    // virtual void host_resetHiddenac ();
    virtual void update (float alpha);
    // virtual void resetErrVec (int mbidx);
    // virtual void resetHiddenAc (int mbidx);
    // virtual void loadHiddenAc ();
    // virtual void initHiddenAc ();
    // virtual void saveHiddenAc ();
    // virtual void initHiddenEr ();
    virtual void Read (FILE *fptr);
    virtual void Write (FILE *fptr);
    virtual void setLRtunemode (int lr);
};

class lstmlayer : public layer
{
public:
    matrix *newc;
    vector<matrix *> c_vec;
    vector<matrix *> dropoutHiddenmaskMat_vec;
protected:
    matrix *hidden_er, *hidden_ac, *c_er, *hidden_ac_last, *c_last;
    matrix *Uz, *Ui, *Uf, *Uo, *Wz, *Wi, *Wf, *Wo;
    matrix *dUz, *dUi, *dUf, *dUo, *dWz, *dWi, *dWf, *dWo;
    matrix *accsdUz, *accsdUi, *accsdUf, *accsdUo, *accsdWz, *accsdWi, *accsdWf, *accsdWo;
    matrix *dPi_1col, *dPf_1col, *dPo_1col;
    matrix *Pi, *Pf, *Po, *dPi, *dPf, *dPo;
    matrix *accsdPi, *accsdPf, *accsdPo;
    matrix *z, *i, *f, *c, *zi, *fc, *o;
    vector<matrix *> z_vec, i_vec, f_vec, newc_vec, o_vec, hidden_ac_vec;
    matrix *dz, *di, *df, *dnewc, *do_;
public:
    virtual void forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void host_forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void backward (matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac);
    virtual void host_resetHiddenac ();
    virtual void update (float alpha);
    virtual void resetErrVec (int mbidx);
    virtual void resetHiddenAc (int mbidx);
    virtual void loadHiddenAc ();
    virtual void initHiddenAc ();
    virtual void saveHiddenAc ();
    virtual void initHiddenEr ();
    virtual void Read (FILE *fptr);
    virtual void Write (FILE *fptr);
    lstmlayer (int nr, int nc, int minibatch, int chunksize);
    ~lstmlayer ();
    virtual void printHiddenAcValue (int i, int j)
    {
        printf ("Uz[0,0]=%f\n", i, j, dUz->fetchvalue(i,j));
    }
    virtual void copyLSTMhighwayc_er (layer *layer1, int chunkiter);
    virtual void setLRtunemode (int lr);
    virtual void genvardropoutmask (int mbidx);
    virtual void copyvardropoutmask ();
};


class lstmhighwaylayer : public lstmlayer
{
public:
    matrix *c_hw, *c_er_hw;
private:
    matrix *s, *sc, *ds, *dsc;
    matrix *Uhw, *Phw, *Rhw;
    matrix *dUhw, *dPhw, *dRhw;
    matrix *accsdUhw, *accsdPhw, *accsdRhw;
    matrix *dPhw_1col, *dRhw_1col;
    vector<matrix *> c_hw_vec, s_vec;
public:
    lstmhighwaylayer (int nr, int nc, int minibatch, int chunksize);
    ~lstmhighwaylayer ();
    virtual void forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void host_forward (matrix *neu0_ac, matrix *neu1_ac);
    virtual void backward (matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac);

    // virtual void host_resetHiddenac ();
    virtual void update (float alpha);
    // virtual void resetErrVec (int mbidx);
    // virtual void resetHiddenAc (int mbidx);
    // virtual void initHiddenAc ();
    // virtual void saveHiddenAc ();
    // virtual void initHiddenEr ();
    virtual void Read (FILE *fptr);
    virtual void Write (FILE *fptr);
    virtual void copyLSTMhighwayc (layer *layer0,  int chunkiter);
    virtual void host_copyLSTMhighwayc (layer *layer0);
    virtual void setLRtunemode (int lr);
};

#endif

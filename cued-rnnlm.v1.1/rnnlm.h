#include "head.h"
#include "layer.h"
#include "cudamatrix.h"
#include "fileops.h"
#include "DataType.h"
#include <omp.h>
#include <random>
#include <algorithm>
typedef mt19937_64 rng_type;

class RNNLM
{
protected:
    string inmodelfile, outmodelfile, trainfile, validfile,
           testfile, inputwlist, outputwlist, nglmstfile,
           sampletextfile, feafile, nceunigramfile, uglmfile;
    vector<int> layersizes,  succlayersizes;
    map<string, int> inputmap, outputmap;
    vector<string>  inputvec, outputvec, ooswordsvec, layertypes;
    vector<float>   ooswordsprob;
    vector<vector<matrix *> > neu_ac_chunk;
    vector<layer *> layers;
    vector<matrix *> neu_ac, neu_er;
    auto_timer timer;

    vector<layer *> layer1_succ, layers_succ;
    vector<matrix *> neu_er1_succ, neu_ac1_succ, neu_er_succ, neu_ac_succ;
    vector<vector<matrix *> > neu_ac_succ_chunk, neu_ac1_succ_chunk;

    float logp, llogp, gradient_cutoff, l2reg, alpha, momentum, min_improvement,
          nwordspersec, lognorm, vrpenalty, lognorm_output, log_num_noise,
          lognormconst, lambda, version, diaginit, dropoutrate,
          lmscale, ip;
    double trnlognorm_mean, lognorm_mean, lognorm_var;
    int rand_seed, deviceid, minibatch, chunksize, debug, iter,
        lrtunemode, traincritmode,
        inputlayersize, outputlayersize, num_layer, wordcn, trainwordcnt,
         validwordcnt, independent, inStartindex, inOOSindex, cachesize,
        outEndindex, outOOSindex, k, counter, mbcntiter,
        mbcnt, fullvocsize, prevword, curword, num_oosword,
        nsample, nthread, succwindowlength,
        num_sulayer, succmergelayer;
    int *host_prevwords, *host_curwords,
        *dev_prevwords, *dev_curwords,
        *mbfeaindices;
    int  *succwords;
    int **dev_succwords, **host_succwords;
    bool alpha_divide, binformat, flag_usegpu, flag_nceunigram;
    auto_timer timer_sampler, timer_forward, timer_output, timer_backprop, timer_hidden;

public:
    RNNLM(string inmodelfile_1, string outmodelfile_1, string inputwlist_1, string outputwlist_1, vector<string> &layersizes_1, int deviceid_1, int mbsize, int cksize, int rand_seed_1, bool binformat_1);
    RNNLM(string inmodelfile_1, string inputwlist_1, string outputwlist_1, int fvocsize, bool bformat, int debuglevel, int mbsize=1, int cksize=1, int succwindowlength_1=0, int dev=0);

    ~RNNLM();

    bool train (string trainfile, string validfile, float learnrate, int deviceid, int csize, int fullvocsize, int independent, int debug);

    bool calppl (string testfilename, float lambda, string nglmfile);

    bool calnbest (string testfilename, float lambda, string nglmfile);

    bool sample (string textfile, string unigramfile, int n);

    void prepareChunk (ReadFileBuf &databuf);
    void cuChunkForward ();
    float host_forward (int prevword, int curword);
    void cuChunkBackward ();
    void cuChunkUpdate ();
    void CollectEntropy();
    void ComputeLognormMeanVar ();
    bool AdjustLearnRate ();
    void InitVariables ();
    void LoadRNNLM(string modelname);
    void LoadTextRNNLM_new(string modelname);
    void WriteRNNLM(string modelname);
    void WriteTextRNNLM_new(string modelname);
    void ReadWordlist (string inputlist, string outputlist);
    void WriteWordlist (string inputlist, string outputlist);
    void printTrainInfo ();
    void printPPLInfo ();
    void printSampleInfo ();
    void printTrainProgress ();

    void init();

    void SelectDevice ();
    void allocSuccwords ();
    void setLRtunemode (string lrtune);
    void setTrainCritmode (string traincrit);
    void setNumncesample (int n)
    {
        k = n;
        // NCE training
        if (traincritmode == 2)
        {
            layers[num_layer-1]->allocNCEMem (k);
        }
    }
    void setMinimprovement (float v) {min_improvement = v;}
    void setLognormConst (float v);
    void setNthread (int n)         {nthread = n;}
    void setMomentum (float momentum);
    void setChunksize (int n)       {chunksize = n;}
    void setLmscale (float v)       {lmscale = v;}
    void setRandseed (int n);
    void setIp (float v)            {ip = v;}
    void setVRpenalty (float v);
    void setFullVocsize (int n);
    void setClipping (float v);
    void setL2Regularization (float v);
    void setDropoutRate (float v);
    void setDiagInit (float v)      {diaginit = v;}
    void setIndependentmode (int v);
    void updateInputWordWgts(int *inwords);
    void allocRNNMem (bool flag_alloclayers=true);
    void allocWordMem ();
    void HandleSentStart_fw (int l, int *prevwords);
    void updateFeatureVector (int *prevwords);
    void genVarDropoutMask (int l, int *prevword);
    void computeErrOnOutputLayer (int chunkiter);
    void host_HandleSentEnd_fw ();
    void HandleSentEnd_bw (int l, int *curwords);
    void initHiddenEr ();
    void initHiddenAc ();
    void loadPreHiddenAc ();
    void saveLastHiddenAc ();
    void setLearnRate (float v);
    void ReadUnigramFile (string unigramfile);

    // functions for using additional feature file in input layer
    void ReadFeaFile (string str)
    {
        feafile = str;
        layers[0]->ReadFeaFile (feafile);
    }

    void setSuccVars (int windowlength, vector<int> sulayersizes, int mergelayer)
    {
        succwindowlength = windowlength;
        succlayersizes = sulayersizes;
        succmergelayer = mergelayer;
    }

    void parseModelTopo (vector<string> &modelstructure);
    };


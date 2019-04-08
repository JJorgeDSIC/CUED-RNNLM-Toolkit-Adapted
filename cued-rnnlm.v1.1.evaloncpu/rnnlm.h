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
    vector<int> layersizes;
    map<string, int> inputmap, outputmap;
    vector<string>  inputvec, outputvec, ooswordsvec, layertypes;
    vector<float>   ooswordsprob;
    vector<vector<matrix *> > neu_ac_chunk;
    vector<layer *> layers;
    vector<matrix *> neu_ac, neu_er;
    auto_timer timer;

    // used for cuedrnnlm v1.1
    int succwindowlength, num_sulayer, succmergelayer;
    int *succwords;
    vector<int> succlayersizes;
    vector<layer *> layer1_succ, layers_succ;
    vector<matrix *> neu_ac_succ, neu_ac1_succ;

    float logp, llogp, gradient_cutoff, l2reg, alpha, momentum, min_improvement,
          nwordspersec, lognorm, vrpenalty, lognorm_output, log_num_noise,
          lognormconst, lambda, version, diaginit, dropoutrate,
          lmscale, ip, smoothvalue;
    double trnlognorm_mean, lognorm_mean, lognorm_var;
    int rand_seed, deviceid, minibatch, chunksize, debug, iter, lrtunemode, traincritmode,
        inputlayersize, outputlayersize, num_layer, wordcn, trainwordcnt,
         validwordcnt, independent, inStartindex, inOOSindex, cachesize,
        outEndindex, outOOSindex, k, counter, mbcntiter,
        mbcnt, fullvocsize, prevword, curword, num_oosword, nsample, nthread;
    int *host_prevwords, *host_curwords,
        *mbfeaindices;
    bool alpha_divide, binformat, flag_nceunigram, linearintplt;
    auto_timer timer_sampler, timer_forward, timer_output, timer_backprop, timer_hidden;

public:
    RNNLM(string inmodelfile_1, string inputwlist_1, string outputwlist_1, int fvocsize, bool bformat, int debuglevel, int mbsize=1, int cksize=1, int dev=0);

    ~RNNLM();

    bool calppl (string testfilename, float lambda, string nglmfile);

    bool calnbest (string testfilename, float lambda, string nglmfile);

    float host_forward (int prevword, int curword);
    void InitVariables ();
    void LoadRNNLM(string modelname);
    void LoadTextRNNLM_new(string modelname);
    void ReadWordlist (string inputlist, string outputlist);
    void printPPLInfo ();

    void init();

    void setNthread (int n)         {nthread = n;}
    void setLmscale (float v)       {lmscale = v;}
    void setIp (float v)            {ip = v;}
    void setSmoothValue (float v)
    {
        smoothvalue = v;
        layers[num_layer-1]->setSmoothValue (smoothvalue);
    }
    void setLinearIntplt (bool flag){linearintplt = flag;}
    void setFullVocsize (int n);
    void setIndependentmode (int v);
    void allocRNNMem (bool flag_alloclayers=true);
    void allocWordMem ();
    void host_HandleSentEnd_fw ();
    void initHiddenAc ();

    // functions for using additional feature file in input layer
    void ReadFeaFile (string str)
    {
        feafile = str;
        layers[0]->ReadFeaFile (feafile);
    }

    };


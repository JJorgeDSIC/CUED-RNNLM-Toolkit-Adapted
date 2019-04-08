#include "rnnlm.h"

#ifndef linux
#define exp10(x)	exp((x)*M_LN10)
#endif

#if 0
            {
                matrix *a = new matrix (100,100);
                matrix *b = new matrix (100, 100);
                matrix *c = new matrix (100, 100);
                foreach_coord (i,j,a)
                {
                    a->assignvalue(i,j,1.0);
                    b->assignvalue(i,j,1.0);
                    c->assignvalue(i,j,1.0);
                }
                printf ("debugging point 1\n");
                cumatrixXmatrix (a,b,c, false, false);
                printf ("debugging point 2\n");
            }
#endif
bool isRNN (string type)
{
    if (type == "recurrent" || type == "gru" || type == "lstm" || type == "gru-highway" || type == "lstm-highway")
        return true;
    else
        return false;
}
bool isGRU (string type)
{
    if (type == "gru" || type == "gru-highway")
        return true;
    else
        return false;
}
bool isLSTM (string type)
{
    if (type == "lstm" || type == "lstm-highway")
        return true;
    else
        return false;

}
bool isLSTMhighway (string type)
{
    if (type == "lstm-highway")
        return true;
    else
        return false;
}

void RNNLM::init()
{
    gradient_cutoff = 5;            // default no gradient cutoff
    llogp           = -1e8;
    min_improvement = 1.001;
    lognormconst    = 0;
    lognorm_mean    = 0;
    lognorm_var     = 0;
    lognorm_output  = -100.0;
    lrtunemode      = 0;                 // default use newbob
    alpha_divide    = false;
    flag_nceunigram = false;
    alpha           = 0.8;
    lambda          = 0.5;
    version         = 1.1;
    iter            = 0;
    num_layer       = 0;
    wordcn          = 0;
    trainwordcnt    = 0;
    validwordcnt    = 0;
    counter         = 0;
    num_oosword     = 0;
    dropoutrate     = 0;
    diaginit        = 0;
    host_prevwords  = NULL;
    host_curwords   = NULL;
    dev_prevwords   = NULL;
    dev_curwords    = NULL;
    host_succwords  = NULL;
    dev_succwords   = NULL;
    flag_usegpu     = true;
    mbfeaindices    = NULL;
	nthread		    = 0;
    lmscale         = 12.0;
    ip              = 0;        // insertion penalty
    num_sulayer     = 0;
    succwindowlength= 0;
}

RNNLM::~RNNLM()
{
    int i, j;
    if (dev_prevwords)      cufree(dev_prevwords);
    if (dev_curwords)       cufree(dev_curwords);
    if (dev_succwords)
    {
        for (int i=0; i<succwindowlength; i++)
        {
            cufree(dev_succwords[i]);
        }
    }
    cufree(dev_succwords);
    if (host_succwords)
    {
        for (int i=0; i<succwindowlength; i++)
        {
            free(host_succwords[i]);
        }
    }
    free (host_succwords);
    if (host_prevwords)     free(host_prevwords);
    if (host_curwords)      free(host_curwords);
    for (i=0; i<num_layer; i++)
    {
        delete layers[i];
    }

    for (i=1; i<num_layer; i++)
    {
        for (j=0; j<chunksize; j++)
        {
            delete neu_ac_chunk[i][j];
        }
        delete neu_er[i];
    }
    if (mbfeaindices) {free(mbfeaindices); mbfeaindices=NULL;}
}


RNNLM::RNNLM(string inmodelfile_1, string inputwlist_1, string outputwlist_1, int fvocsize, bool bformat, int debuglevel, int mbsize/*=1*/, int cksize/*=1*/, int succwindowlength_1/*=0*/, int dev/*=0*/):inmodelfile(inmodelfile_1), inputwlist(inputwlist_1), outputwlist(outputwlist_1), binformat(bformat),  debug(debuglevel)
{
    int i;
    init ();
    minibatch = mbsize;
    chunksize = cksize;
    succwindowlength = succwindowlength_1;
    deviceid = dev;
    flag_usegpu = true;
    SelectDevice ();
    LoadRNNLM (inmodelfile);
    // this will be useful for sampling
    allocWordMem();
    setFullVocsize (fvocsize);
}


RNNLM::RNNLM(string inmodelfile_1, string outmodelfile_1, string inputwlist_1, string outputwlist_1, vector<string> &modelstructure, int deviceid_1, int mbsize, int cksize, int rand_seed_1, bool binformat_1) : inmodelfile(inmodelfile_1), outmodelfile(outmodelfile_1), inputwlist(inputwlist_1), outputwlist(outputwlist_1), deviceid(deviceid_1), minibatch(mbsize), chunksize(cksize), rand_seed(rand_seed_1), binformat(binformat_1)
{
    int i;
#ifdef RAND48
    srand48(rand_seed);
#else
    srand(rand_seed);
#endif
    init ();
    // select proper gpu device
    SelectDevice ();
    if (isEmpty(inmodelfile))       // train from scratch
    {
        parseModelTopo (modelstructure);
        allocRNNMem ();
        ReadWordlist (inputwlist, outputwlist);
    }
    else
    {
        LoadRNNLM (inmodelfile);
    }
    allocWordMem();
}

// allocate memory for RNNLM model
void RNNLM::allocRNNMem (bool flag_alloclayers/*=true*/)
{
    int i, j, nr, nc, dim_fea=0;
    num_layer = layersizes.size() - 1;
    if (num_layer < 2)
    {
        printf ("ERROR: the number of layers (%d) should be greater than 2\n", num_layer);
    }
    inputlayersize = layersizes[0];
    outputlayersize = layersizes[num_layer];
    layers.resize(num_layer);
    neu_ac.resize(num_layer+1);
    neu_er.resize(num_layer+1);
    if (flag_alloclayers)
    {
        for (i=0; i<num_layer; i++)
        {
            nr = layersizes[i];
            nc = layersizes[i+1];
            if (layertypes[i][0] == 'I' || layertypes[i][0] == 'i') // input layer
            {
                assert (i == 0);
                layers[i] = new inputlayer (nr, nc,  minibatch, chunksize, dim_fea);
                layertypes[i] = "input";
            }
            else if (layertypes[i][0] == 'L' || layertypes[i][0] == 'l') // linear layer
            {
                layers[i] = new linearlayer (nr, nc, minibatch, chunksize);
                layertypes[i] = "linear";
            }
            else if (layertypes[i][0] == 'F' || layertypes[i][0] == 'f') // feedforward layer
            {
                layers[i] = new feedforwardlayer (nr, nc, minibatch, chunksize);
                if (layertypes[i].length() == 1 || layertypes[i][1] == '0')
                    // sigmoid node
                {
                    layers[i]->setnodetype (0);
                }
                else if (layertypes[i][1] == '1') // relu
                {
                    layers[i]->setnodetype (1);
                }
                layertypes[i] = "feedforward";
            }
            else if (layertypes[i][0] == 'r') // simple rnn with sigmoid
            {
                layers[i] = new recurrentlayer (nr, nc, minibatch, chunksize);
                layers[i]->setnodetype (0);
                layertypes[i] = "recurrent";
            }
            else if (layertypes[i][0] == 'R')       // simple rnn with relu
            {
                layers[i] = new recurrentlayer (nr, nc, minibatch, chunksize);
                layers[i]->setnodetype (1);
                layers[i]->setreluratio (RELURATIO);
                layertypes[i] = "recurrent";
            }
            else if (layertypes[i][0] == 'G' || layertypes[i][0] == 'g') // GRU
            {
                layers[i] = new grulayer (nr, nc, minibatch, chunksize);
                layertypes[i] = "gru";
            }
            else if (layertypes[i][0] == 'X' || layertypes[i][0] == 'x') // GRU-highway
            {
                layers[i] = new gruhighwaylayer (nr, nc, minibatch, chunksize);
                layertypes[i] = "gru-highway";
            }
            else if (layertypes[i][0] == 'M' || layertypes[i][0] == 'm') // LSTM
            {
                layers[i] = new lstmlayer (nr, nc, minibatch, chunksize);
                layertypes[i] = "lstm";
            }
            else if (layertypes[i][0] == 'Y' || layertypes[i][0] == 'y') // LSTM-highway
            {
                // last layer much be lstm with same width
                layers[i] = new lstmhighwaylayer (nr, nc, minibatch, chunksize);
                layertypes[i] = "lstm-highway";
            }
            else if (i == num_layer-1)
            {
                layers[i] = new outputlayer (nr, nc, minibatch, chunksize);
                layers[i]->setTrainCrit (traincritmode);
                layertypes[i] = "output";
            }
        }
    }

    neu_ac_chunk.resize (layersizes.size());
    for (i=1; i<layersizes.size(); i++)  // no need to allocate for input layer
    {
        int nrow = layersizes[i];
        neu_ac_chunk[i].resize(chunksize);
        for (j=0; j<chunksize; j++)
        {
            neu_ac_chunk[i][j] = new matrix (nrow, minibatch, flag_usegpu);
        }
        neu_er[i] = new matrix (nrow, minibatch, flag_usegpu);
        neu_ac[i] = new matrix (nrow, minibatch, flag_usegpu);
    }
}

// allocate memory used for input words, target words
void RNNLM::allocWordMem ()
{
    host_prevwords = (int *)calloc(minibatch*chunksize, sizeof(int));
    host_curwords = (int *)calloc (minibatch*chunksize, sizeof(int));
    dev_prevwords = (int *)cucalloc (minibatch * chunksize * sizeof(int));
    dev_curwords = (int *)cucalloc (minibatch * chunksize * sizeof(int));
    memset(host_prevwords, 0, sizeof(int)*minibatch*chunksize);
    memset(host_curwords, 0, sizeof(int)*minibatch*chunksize);
    cumemset(dev_prevwords, 0, sizeof(int)*minibatch*chunksize);
    cumemset(dev_curwords, 0, sizeof(int)*minibatch*chunksize);

}

void RNNLM::printTrainInfo ()
{
    string str;
    printf ("train   txt:       %s\n", trainfile.c_str());
    printf ("valid   txt:       %s\n", validfile.c_str());
    printf ("input  list:       %s\n", inputwlist.c_str());
    printf ("output list:       %s\n", outputwlist.c_str());
    printf ("num   layer:       %d\n", num_layer);
    for (int i=0; i<=num_layer; i++)
    {
        printf ("#layer[%d]  :       %-10d type:   %-6s\n", i, layersizes[i], layertypes[i].c_str());
    }
    if (layers[0]->usefeainput())
    {
    printf ("feature file:      %s\n", feafile.c_str());
    printf ("dim  feature:      %d\n", layers[0]->getdimfea());
    }
    printf ("fullvoc size:      %d\n", fullvocsize);
    printf ("device   id:       %d\n", deviceid);
    printf ("nthread    :       %d\n", nthread);
    printf ("minibatch  :       %d\n", minibatch);
    printf ("chunksize  :       %d\n", chunksize);
    printf ("learn  rate:       %f\n", alpha);
    printf ("momentum   :       %f\n", momentum);
    printf ("cache  size:       %d\n", cachesize);
    printf ("min  improv:       %f\n", min_improvement);
    printf ("independent:       %d\n", independent);
    printf ("debug level:       %d\n", debug);
    printf ("random seed:       %d\n", rand_seed);
    printf ("write model:       %s\n", outmodelfile.c_str());
    printf ("binary     :       %d\n", binformat);
    if (dropoutrate > 0 && dropoutrate < 1)
    {
        printf ("dropout rate:      %f\n", dropoutrate);
    }
    printf ("clipping   :       %f\n", gradient_cutoff);
    printf ("l2reg      :       %f\n", l2reg);
    if (lrtunemode == 0)        str = "newbob";
    else if(lrtunemode == 1)    str = "adagrad";
    else if(lrtunemode == 2)    str = "rmsprop";
    else                        str = "unknown";
    printf ("learn tune :       %s\n", str.c_str());
    if (traincritmode== 0)      str = "ce";
    else if (traincritmode==1)  str = "vr";
    else if (traincritmode==2)  str = "nce";
    else                        str = "unknown";
    printf ("train crit :       %s\n", str.c_str());
    if (traincritmode == 1)
    {
        printf ("vr penalty :       %f\n", vrpenalty);
    }
    else if (traincritmode == 2)
    {
        printf ("ncesample  :       %d\n", k);
        if (flag_nceunigram)
        {
            printf ("nceunigramfile:    %s\n", nceunigramfile.c_str());
        }
    }
    printf ("lognormconst :     %f\n", lognormconst);
    printf ("#succeeding words: %d\n", succwindowlength);
    if (succwindowlength > 0)
    {
        printf ("\tFor succeeding word part:\n");
        for (int i=0; i<succlayersizes.size(); i++)
        {
            printf ("#layer[%d]  :       %-10d\n", i, succlayersizes[i]);
        }
        printf ("merge layer  :     %d\n", succmergelayer);
    }
}


void RNNLM::printPPLInfo ()
{
    string str;
    printf ("model file :       %s\n", inmodelfile.c_str());
    printf ("input  list:       %s\n", inputwlist.c_str());
    printf ("output list:       %s\n", outputwlist.c_str());
    printf ("num   layer:       %d\n", num_layer);
    for (int i=0; i<=num_layer; i++)
    {
        printf ("#layer[%d]  :       %-10d type:   %-6s\n", i, layersizes[i], layertypes[i].c_str());
    }
    if (layers[0]->usefeainput())
    {
    printf ("feature file:      %s\n", feafile.c_str());
    printf ("dim  feature:      %d\n", layers[0]->getdimfea());
    }
    printf ("independent:       %d\n", independent);
    printf ("test file  :       %s\n", testfile.c_str());
    printf ("nglm file  :       %s\n", nglmstfile.c_str());
    printf ("lambda (rnn):      %f\n", lambda);
    printf ("fullvocsize:       %d\n", fullvocsize);
    printf ("debug level:       %d\n", debug);
    printf ("nthread    :       %d\n", nthread);
    printf ("lognormconst :     %f\n", lognormconst);
    printf ("#succeeding words: %d\n", succwindowlength);
    if (succwindowlength > 0)
    {
        printf ("\tFor succeeding word part:\n");
        for (int i=0; i<succlayersizes.size(); i++)
        {
            printf ("#layer[%d]  :       %-10d\n", i, succlayersizes[i]);
        }
        printf ("merge layer  :     %d\n", succmergelayer);
    }
}

void RNNLM::printSampleInfo ()
{
    printf ("model file :       %s\n", inmodelfile.c_str());
    printf ("input  list:       %s\n", inputwlist.c_str());
    printf ("output list:       %s\n", outputwlist.c_str());
    printf ("num   layer:       %d\n", num_layer);
    for (int i=0; i<=num_layer; i++)
    {
        printf ("#layer[%d]  :       %d\n", i, layersizes[i]);
    }
    printf ("independent:       %d\n", independent);
    printf ("device   id:       %d\n", deviceid);
    printf ("minibatch  :       %d\n", minibatch);
    printf ("text file  :       %s\n", sampletextfile.c_str());
    printf ("uglm file  :       %s\n", uglmfile.c_str());
    printf ("nsample    :       %d\n", nsample);
    printf ("fullvocsize:       %d\n", fullvocsize);
    printf ("debug level:       %d\n", debug);
    printf ("rand seed  :       %d\n", rand_seed);
}

bool RNNLM::train (string trainfilename, string validfilename, float learnrate, int deviceid, int csize, int fvocsize, int independent, int debuglevel)
{
    int randint, i, j;
    trainfile = trainfilename;
    validfile = validfilename;
    debug = debuglevel;
    alpha = learnrate / (minibatch*chunksize);
    setLearnRate (alpha);
    cachesize = csize;
    int localmbiter;
    float finetunealpha = alpha;
    setFullVocsize (fvocsize);

    if (debug > 1)
    {
        printTrainInfo();
    }
    ReadFileBuf trainbuf(trainfile, inputmap, outputmap, minibatch, cachesize, layers[0]->getnumfea() );
    randint = trainbuf.getRandint();
    ReadFileBuf validbuf(validfile, inputmap, outputmap, minibatch, 0, layers[0]->getnumfea() );     // don't cache the valid data
    if (cachesize == 0)     trainbuf.FillBuffer();
    validbuf.FillBuffer();
    trainwordcnt = trainbuf.getWordcnt();
    validwordcnt = validbuf.getWordcnt();

    if (debug > 0)
    {
        printf ("Read train txt file: %s, %d lines, %d words totally\n", trainfile.c_str(), trainbuf.getLinecnt(), trainwordcnt);
        printf ("Read valid txt file: %s, %d lines, %d words totally\n", validfile.c_str(), validbuf.getLinecnt(), validwordcnt);
        printf ("Start RNNLM training...initial learning rate per sample: %f\n", alpha);
    }

    if (traincritmode == 2)     // NCE training
    {
        layers[num_layer-1]->prepareNCEtrain (trainbuf.getUnigram(), outputvec);
    }

    while (1)
    {
        int chunkcnt = 0;
        timer.start();
        InitVariables();
        initHiddenAc ();
        if (cachesize > 0)
        {
            trainbuf.Init();
        }
        // Train Stage
        mbcntiter = 0;
        mbcnt = trainbuf.getMBcnt();

        // using feature vector in the input layer
        if (layers[0]->usefeainput())
        {
            layers[0]->setFeaIndices (trainbuf.getfeaptr());
        }

        while (1)
        {
            chunkcnt ++;
            if (mbcntiter >= mbcnt)          break;

            // fill in the chunk with prevword and curwords
            prepareChunk (trainbuf);

            cuChunkForward ();

            cuChunkBackward ();

            cuChunkUpdate ();

            // collect entropy
            CollectEntropy ();

            printTrainProgress ();
        }
        if (traincritmode == 2)
        {
            printf ("%cIter %3d\tAlpha: %.8f\t Train Entropy: %s\t ", 13, iter, alpha, "NULL");
        }
        else
        {
            printf ("%cIter %3d\tAlpha: %.8f\t Train Entropy: %.4f\t ", 13, iter, alpha, -logp/log10(2)/wordcn);
        }
        nwordspersec = wordcn / (timer.stop());

        // Valid Stage
        chunkcnt = 0;
        InitVariables ();
        initHiddenAc ();
        mbcntiter = 0;
        mbcnt = validbuf.getMBcnt();
        // for NCE training, use CE for validation
        for (int l=0; l<num_layer; l++)
        {
            layers[l]->setEvalmode (true);
        }
        // using feature vector in the input layer
        if (layers[0]->usefeainput())
        {
            layers[0]->setFeaIndices (validbuf.getfeaptr());
        }

        while (1)
        {
            chunkcnt ++;
            if (mbcntiter >= mbcnt)          break;

            // fill in the chunk with prevword and curwords
            prepareChunk (validbuf);

            cuChunkForward ();

            // collect entropy
            CollectEntropy ();

            ComputeLognormMeanVar ();
        }

        for (int l=0; l<num_layer; l++)
        {
            layers[l]->setEvalmode (false);
        }

        // Print Statistics
        timer.add();
        printf ("Valid Entropy: %.4f\tPPL: %.3f  Words/sec: %.3f\tTime: %.2fs\n", -logp/log10(2)/wordcn, exp10(-logp/wordcn), nwordspersec, timer.getacctime());
        if (traincritmode != 0)         // VR or NCE training
        {
            lognorm_var  = (lognorm_var - lognorm_mean*lognorm_mean/wordcn)/(wordcn-1);
            lognorm_mean = lognorm_mean/wordcn;
            lognorm_output = lognorm_mean;
            printf ("Mean: %.3f Var: %.3f\n", lognorm_mean, lognorm_var);
            lognorm_mean = 0;
            lognorm_var = 0;
        }
        // Learn rate tune
        if (AdjustLearnRate () )
        {
            break;
        }

        if (debug > 2)
        {
            char filename[1024];
            if (binformat)
            {
                sprintf (filename, "%s.bin.iter%d", outmodelfile.c_str(), iter);
            }
            else
            {
                sprintf (filename, "%s.iter%d", outmodelfile.c_str(), iter);
            }
            WriteRNNLM (filename);
        }
        iter ++;
    }

    WriteRNNLM (outmodelfile);

    // delete the train and valid index file
    trainbuf.DeleteIndexfile();
    validbuf.DeleteIndexfile();

    return SUCCESS;
}


void RNNLM::initHiddenAc ()
{
    for (int l=0; l<num_layer; l++)
    {
        if (isRNN(layertypes[l]))
        {
            layers[l]->initHiddenAc ();
        }
    }
}
void RNNLM::prepareChunk (ReadFileBuf &databuf)
{
    int localmbiter;
    memset (host_prevwords, 0, sizeof(int)*minibatch*chunksize);
    memset (host_curwords,  0, sizeof(int)*minibatch*chunksize);
    for (int chunkiter=0; chunkiter<chunksize; chunkiter++)
    {
        counter ++;
        if (mbcntiter >= mbcnt)
        {
            // break;
            for (int i=0; i<minibatch; i++)
            {
                host_prevwords[chunkiter*minibatch+i] = INVALID_INT;
                host_curwords[chunkiter*minibatch+i] = INVALID_INT;
            }
            // for succeeding words in the input layer
            for (int widx=0; widx<succwindowlength; widx++)
            {
                for (int i=0; i<minibatch; i++)
                {
                    host_succwords[widx][chunkiter*minibatch+i] = INVALID_INT;
                }
            }
            mbcntiter ++;
        }
        else
        {
            if (cachesize == 0)
            {
                localmbiter = mbcntiter;
            }
            else
            {
                if (mbcntiter%cachesize==0)
                {
                    databuf.FillBuffer();
                }
                localmbiter = mbcntiter % cachesize;
            }
            mbcntiter ++;
            databuf.GetData(localmbiter, host_prevwords+chunkiter*minibatch, host_curwords+chunkiter*minibatch);

            // for succeeding words in the input layer
            for (int widx=0; widx<succwindowlength; widx++)
            {
                if (widx > 0)
                {
                    databuf.GetInputData (localmbiter, localmbiter+2+widx, host_succwords[widx]+chunkiter*minibatch, host_succwords[widx-1]+chunkiter*minibatch);
                }
                else
                {
                    databuf.GetInputData (localmbiter, localmbiter+2+widx, host_succwords[widx]+chunkiter*minibatch, 0);
                }
            }
        }
    }
    cucpytoGpu(dev_prevwords, host_prevwords, sizeof(int)*minibatch*chunksize);
    cucpytoGpu(dev_curwords,  host_curwords,  sizeof(int)*minibatch*chunksize);


    for (int i=0; i<succwindowlength; i++)
    {
        cucpytoGpu (dev_succwords[i], host_succwords[i], sizeof(int)*minibatch*chunksize);
    }
}

void RNNLM::printTrainProgress ()
{
    if (debug > 1 && counter % 100 == 0)
    {
        GPUsynchronizewithCPU();
        nwordspersec = wordcn / (timer.stop());
        if (traincritmode == 2)
        {
            printf("%cIter %3d\tAlpha: %.8f\t Train Entropy: %s\t Progress: %.2f%%\t Words/sec: %.3f", 13, iter, alpha, "NULL", wordcn*100.0/trainwordcnt, nwordspersec);
        }
        else
        {
            printf("%cIter %3d\tAlpha: %.8f\t Train Entropy: %.4f\t Progress: %.2f%%\t Words/sec: %.3f", 13, iter, alpha, -logp/log10(2)/wordcn, wordcn*100.0/trainwordcnt, nwordspersec);
        }
        fflush(stdout);
    }
}

void RNNLM::genVarDropoutMask (int l, int *prevwords)
{
    int i;
    // copy the previous dropout mask first;
    // if new sentence, then generate new mask for the mbidx
    layers[l]->copyvardropoutmask ();
    for (i=0; i<minibatch; i++)
    {
        if (prevwords[i] == inStartindex)
        {
            layers[l]->genvardropoutmask (i);
        }
    }
}


void RNNLM::HandleSentStart_fw (int l, int *prevwords)
{
    int mbidx;
    for (mbidx=0; mbidx<minibatch; mbidx++)
    {
        if (prevwords[mbidx] == inStartindex)
        {
            layers[l]->resetHiddenAc (mbidx);
        }
    }
}

void RNNLM::updateFeatureVector (int *prevwords)
{
    int mbidx;
    for (mbidx=0; mbidx<minibatch; mbidx++)
    {
        if (prevwords[mbidx] == inStartindex)
        {
            // use feature vector
            layers[0]->updateFeaMat (mbidx);
        }
    }
}

void RNNLM::host_HandleSentEnd_fw ()
{
    if (! independent)
    {
        return;
    }
    for (int l=0; l<num_layer; l++)
    {
        if (isRNN (layertypes[l]))
        {
            layers[l]->host_resetHiddenac();
        }
    }
}

void RNNLM::HandleSentEnd_bw (int l, int *curwords)
{
    int i;
    for (i=0; i<minibatch; i++)
    {
        if (curwords[i] == outEndindex)
        {
            layers[l]->resetErrVec (i);
        }
    }
}

void RNNLM::initHiddenEr()
{
    int l;
    for (l=0; l<num_layer; l++)
    {
        if (isRNN(layertypes[l]))
        {
            layers[l]->initHiddenEr ();
        }
    }
}

void RNNLM::computeErrOnOutputLayer(int chunkiter)
{
    neu_ac[num_layer] = neu_ac_chunk[num_layer][chunkiter];
    layers[num_layer-1]->calerr (neu_ac[num_layer], neu_er[num_layer], dev_curwords+chunkiter*minibatch);
}

void RNNLM::loadPreHiddenAc ()
{
    for (int l=1; l<num_layer; l++)
    {
        if (isRNN(layertypes[l]))
        {
            layers[l]->loadHiddenAc();
        }
    }
}

void RNNLM::saveLastHiddenAc ()
{
    for (int l=1; l<num_layer; l++)
    {
        if (isRNN(layertypes[l]))
        {
            layers[l]->saveHiddenAc();
        }
    }
}

void RNNLM::cuChunkForward ()
{
    int l=1, chunkiter;
    num_sulayer = succlayersizes.size() - 1;
    GPUsynchronizewithCPU ();
    // if the feature input layer is used
    if (layers[0]->usefeainput())
    {
        layers[0]->assignFeaMat();
    }
    // load AC from last chunk
    loadPreHiddenAc ();
    for (chunkiter=0; chunkiter<chunksize; chunkiter++)
    {
        // the first layer is always word embedding layer
        for (l=0; l<num_layer; l++)
        {
            layers[l]->setChunkIter (chunkiter);
        }
        neu_ac[1] = neu_ac_chunk[1][chunkiter];
        // not update feature matrix for the first minibatch
        if (layers[0]->usefeainput() && (mbcntiter!=chunksize || chunkiter!=0))
        {
            updateFeatureVector (host_prevwords+chunkiter*minibatch);
        }
        layers[0]->getWordEmbedding (dev_prevwords+chunkiter*minibatch, neu_ac[1]);

        // for succeeding words in the input layer, hard-coding now
        if (succwindowlength > 0)
        {
            for (l=2; l<num_sulayer; l++)
            {
                layers_succ[l]->setChunkIter (chunkiter);
            }

            neu_ac_succ[2] = neu_ac_succ_chunk[2][chunkiter];
            neu_ac_succ[2]->initmatrix();
            for (int i=0; i<succwindowlength; i++)
            {
                layer1_succ[i]->setChunkIter (chunkiter);
                neu_ac1_succ[i] = neu_ac1_succ_chunk[i][chunkiter];
                layers[0]->getWordEmbedding (dev_succwords[i]+chunkiter*minibatch, neu_ac1_succ[i]);
                layer1_succ[i]->forward_nosigm (neu_ac1_succ[i], neu_ac_succ[2]);
            }
            neu_ac_succ[2]->sigmoid();

            for (l=2; l<num_sulayer; l++)
            {
                neu_ac_succ[l] = neu_ac_succ_chunk[l][chunkiter];
                neu_ac_succ[l+1] = neu_ac_succ_chunk[l+1][chunkiter];
                neu_ac_succ[l+1]->initmatrix();
                layers_succ[l]->forward (neu_ac_succ[l], neu_ac_succ[l+1]);
            }
        }

        if (traincritmode == 2)         // NCE training
        {
            layers[num_layer-1]->assignTargetSample (host_curwords+chunkiter*minibatch);
        }
        // forward for the later layers
        for (l=1; l<num_layer; l++)
        {
            if (isRNN(layertypes[l]) && independent)
            {
                // set hidden_ac to intial vector if there is a sent start
                HandleSentStart_fw (l, host_prevwords+chunkiter*minibatch);
            }
#ifdef VARIATIONALDROPOUT
            if (layers[l]->isDropout() && l != 0)
            {
                genVarDropoutMask (l, host_prevwords+chunkiter*minibatch);
            }
#endif
            neu_ac[l] = neu_ac_chunk[l][chunkiter];
            neu_ac[l+1] = neu_ac_chunk[l+1][chunkiter];
            neu_ac[l+1]->initmatrix();

            // merged the ac here
            if (succwindowlength > 0 && l+1 == succmergelayer)
            {
                neu_ac[l+1]->assign (neu_ac_succ[num_sulayer]);
            }

            layers[l]->forward (neu_ac[l], neu_ac[l+1]);

            // if next layer is LSTM-highway layer
            if (l != num_layer-1 && isLSTMhighway(layertypes[l+1]))
            {
                layers[l+1]->copyLSTMhighwayc (layers[l], chunkiter);
            }
        }
    }
    // save AC for next chunk
    saveLastHiddenAc ();
}

void RNNLM::cuChunkBackward ()
{
    int l, chunkiter;
    GPUsynchronizewithCPU ();
    // set hidden_er to 0 before BPTT for each chunk
    initHiddenEr ();
    for (chunkiter=chunksize-1; chunkiter>=0; chunkiter--)
    {
        for (l=num_layer-1; l>=0; l--)
        {
            layers[l]->setChunkIter (chunkiter);
        }
        // compute error in the output layer
        computeErrOnOutputLayer (chunkiter);

        // neu_ac_succ[2] = neu_ac_succ_chunk[2][chunkiter];
        // layers[num_layer-1]->setSuccAc (neu_ac_succ[2]);
        // layers[num_layer-1]->setSuccEr (neu_er_succ[2]);

        // BPTT from output to input layer
        for (l=num_layer-1; l>0; l--)
        {
            if (isRNN(layertypes[l]) && independent)
            {
                // set hidden_er to 0 if there is a sent end
                HandleSentEnd_bw (l, host_curwords+chunkiter*minibatch);
            }
            neu_ac[l+1] = neu_ac_chunk[l+1][chunkiter];
            neu_ac[l] = neu_ac_chunk[l][chunkiter];
            layers[l]->backward(neu_er[l+1], neu_ac[l+1], neu_er[l], neu_ac[l]);
            if (isLSTMhighway (layertypes[l]))
            {
                layers[l-1]->copyLSTMhighwayc_er (layers[l], chunkiter);
            }
        }
        // compute gradient in input layer (word embedding)
        layers[0]->updateWordEmbedding (dev_prevwords+chunkiter*minibatch, neu_er[1]);


        // for succeeding words
        if (succwindowlength > 0)
        {
            for (l=num_sulayer-1; l>=2; l--)
            {
                layers_succ[l]->setChunkIter (chunkiter);
            }
            neu_er_succ[num_sulayer]->assign (neu_er[succmergelayer]);
            for (l=num_sulayer-1; l>=2; l--)
            {
                neu_ac_succ[l+1] = neu_ac_succ_chunk[l+1][chunkiter];
                neu_ac_succ[l]   = neu_ac_succ_chunk[l][chunkiter];
                layers_succ[l]->backward (neu_er_succ[l+1], neu_ac_succ[l+1], neu_er_succ[l], neu_ac_succ[l]);
            }
            // for succeeding words in the input layer
            neu_er_succ[2]->multiplysigmoid(neu_ac_succ[2]);
            for (int i=0; i<succwindowlength; i++)
            {
                layer1_succ[i]->setChunkIter (chunkiter);
                neu_ac1_succ[i] = neu_ac1_succ_chunk[i][chunkiter];
                layer1_succ[i]->backward_succ (neu_er_succ[2], neu_ac_succ[2], neu_er1_succ[i], neu_ac1_succ[i]);
                layers[0]->updateWordEmbedding (dev_succwords[i]+chunkiter*minibatch, neu_er1_succ[i]);
            }
        }
    }
}

void RNNLM::cuChunkUpdate ()
{
    int l;
    for (l=0; l<num_layer; l++)
    {
        layers[l]->update(alpha);
    }

    if (succwindowlength > 0)
    {
        for (int i=0; i<succwindowlength; i++)
        {
            layer1_succ[i]->update (alpha);
        }

        for (l=2; l<num_sulayer; l++)
        {
            layers_succ[l]->update (alpha);
        }
    }
}

void RNNLM::InitVariables ()
{
    int i, j;
    counter = 0;
    logp = 0;
    wordcn = 0;
}

bool RNNLM::AdjustLearnRate ()
{
    if (iter == 0)
    {
        llogp = -1e8;
    }
    if (lrtunemode == 0)        // newbob
    {
        if (logp*min_improvement < llogp)
        {
            if (alpha_divide)   {return true;}
            else                {alpha_divide = true;}
            // alpha_divide = true;
        }
        if (alpha_divide)   alpha /= 2;
    }
    else if (lrtunemode == 1)   // adagrad
    {
        alpha = alpha;
        if (logp*min_improvement < llogp)
        {
            return true;
        }
    }
    else if (lrtunemode == 2)   // rmsprop
    {
        alpha = alpha*0.99;
        if (logp*min_improvement < llogp && iter > 10)
        {
            return true;
        }
    }
    else
    {
        printf ("ERROR: unrecognized learning rate tune algorithm!\n");
    }
    llogp = logp;
    setLearnRate (alpha);
    return false;
}

void RNNLM::ComputeLognormMeanVar ()
{
    if (traincritmode == 0) return;
    // for VR and NCE training
    layers[num_layer-1]->ComputeAccMeanVar (host_curwords, lognorm_mean, lognorm_var);
}

void RNNLM::CollectEntropy ()
{
    for (int chunkiter=0; chunkiter<chunksize; chunkiter++)
    {
        neu_ac[num_layer] = neu_ac_chunk[num_layer][chunkiter];
        neu_ac_chunk[num_layer][chunkiter]->fetch ();
        for (int mb=0; mb<minibatch; mb ++)
        {
            int word = host_curwords[mb+chunkiter*minibatch];
            if (word != INVALID_INT)
            {
                float word_prob = neu_ac_chunk[num_layer][chunkiter]->fetchhostvalue (word, mb);
                if (isinf (log10(word_prob))|| word_prob == 0)
                {
                    logp += log10(1e-10);
                    wordcn ++;
#if 0
                    if (debug > 2)
                    {
                        printf("\nNumerical error %dth sample in %dth minibatch, word_prob=%f\n", mb, counter, neu_ac[num_layer]->fetchhostvalue (word, mb));
                    }
#endif
                }
                else
                {
                    logp += log10(word_prob);
                    wordcn ++;
                }
            }
        }
    }
}

void RNNLM::SelectDevice ()
{
    if (debug > 0)
    {
        dumpdeviceInfo ();
    }
    int deviceNum = numdevices();
    if (deviceid < 0)
    {
        printf ("ERROR: gpu device id (%d) should be greater than 0\n", deviceid);
    }
    if (deviceid > deviceNum-1)
    {
        printf ("ERROR: deviceid(%d) should be less than deviceNum(%d)\n", deviceid, deviceNum);
        exit(0);
    }
    setDeviceid (deviceid);
    if (debug > 1)
    {
        printf ("Device %d will be chosen for training\n", deviceid);
    }
    // init cublas
    initcuHandle ();
    // init curand
    initcuRandstates ();
}

void RNNLM::LoadRNNLM(string modelname)
{
    LoadTextRNNLM_new (modelname);
    ReadWordlist (inputwlist, outputwlist);
}

void RNNLM::WriteRNNLM(string modelname)
{
    // write to input and output word list
    string inputlist = outmodelfile + ".input.wlist.index";
    string outputlist = outmodelfile + ".output.wlist.index";
    WriteWordlist (inputlist, outputlist);
    WriteTextRNNLM_new (modelname);
}

void RNNLM::LoadTextRNNLM_new (string modelname)
{
    int i, a, b, dim_fea;
    int err;
    float v;
    char word[1024];
    FILE *fptr = NULL;
    // read model file
    fptr = fopen (modelname.c_str(), "r");
    if (fptr == NULL)
    {
        printf ("ERROR: Failed to read RNNLM model file(%s)\n", modelname.c_str());
        exit (0);
    }
    err = fscanf (fptr, "cuedrnnlm v%f\n", &v);
    if (v != version)
    {
        printf ("Error: the version of rnnlm model(v%.1f) is not consistent with binary supported(v%.1f)\n", v, version);
        exit (0);
    }
    err = fscanf (fptr, "train file: %s\n", word);     trainfile = word;
    err = fscanf (fptr, "valid file: %s\n", word);     validfile = word;
    err = fscanf (fptr, "number of iteration: %d\n", &iter);
    iter ++;
    err = fscanf (fptr, "#train words: %d\n", &trainwordcnt);
    err = fscanf (fptr, "#valid words: %d\n", &validwordcnt);
    err = fscanf (fptr, "#layer: %d\n", &num_layer);
    layersizes.resize(num_layer+1);
    layertypes.resize(num_layer+1);
    for (i=0; i<layersizes.size(); i++)
    {
        err = fscanf (fptr, "layer %d size: %d type: %s\n", &b, &a, word);
        assert(b==i);
        layersizes[i] = a;
        layertypes[i] = word;
    }
    err = fscanf (fptr, "fullvoc size: %d\n", &fullvocsize);

    err = fscanf (fptr, "independent mode: %d\n", &independent);
    err = fscanf (fptr, "train crit mode: %d\n",  &traincritmode);
    err = fscanf (fptr, "log norm: %f\n", &lognorm_output);
    err = fscanf (fptr, "dim feature: %d\n", &dim_fea);

    err = fscanf (fptr, "num of succeeding words: %d\n", &succwindowlength);
    if (succwindowlength > 0)
    {
        err = fscanf (fptr, "num of succlayer: %d\n", &num_sulayer);
        succlayersizes.resize (num_sulayer+1);
        for (i=0; i<=num_sulayer; i++)
        {
            err = fscanf (fptr, "succlayer %d size: %d\n", &i, &a);
            succlayersizes[i] = a;
        }
        err = fscanf (fptr, "merging layer: %d\n", &succmergelayer);
    }

    lognormconst = lognorm_output;
    allocRNNMem (false);
    for (i=0; i<num_layer; i++)
    {
        err = fscanf (fptr, "layer %d -> %d type: %s\n", &a, &b, word);
        string type = layertypes[i];
        assert (word == type);
        int nr = layersizes[i];
        int nc = layersizes[i+1];
        if (type == "input")
        {
            assert (i == 0);
            layers[i] = new inputlayer (nr, nc, minibatch, chunksize, dim_fea);
        }
        else if (type == "output")
        {
            assert (i == num_layer-1);
            layers[i] = new outputlayer (nr, nc, minibatch, chunksize);
        }
        else if (type == "feedforward")
        {
            layers[i] = new feedforwardlayer (nr, nc, minibatch, chunksize);
        }
        else if (type == "recurrent")
        {
            layers[i] = new recurrentlayer (nr, nc, minibatch, chunksize);
        }
        else if (type == "linear")
        {
            layers[i] = new linearlayer (nr, nc, minibatch, chunksize);
        }
        else if (type == "lstm")
        {
            layers[i] = new lstmlayer (nr, nc, minibatch, chunksize);
        }
        else if (type == "gru")
        {
            layers[i] = new grulayer (nr, nc, minibatch, chunksize);
        }
        else if (type == "gru-highway")
        {
            layers[i] = new gruhighwaylayer (nr, nc, minibatch, chunksize);
        }
        else if (type == "lstm-highway")
        {
            layers[i] = new lstmhighwaylayer (nr, nc, minibatch, chunksize);
        }
        else
        {
            printf ("Error: unknown layer type: %s\n", type.c_str());
        }
        layers[i]->Read(fptr);
    }

    if (succwindowlength > 0)
    {
        layer1_succ.resize (succwindowlength);
        for (int i=0; i<succwindowlength; i++)
        {
            int nr = layersizes[1];
            int nc = layersizes[2];
            layer1_succ[i] = new feedforwardlayer (nr, nc, minibatch, chunksize);
            err = fscanf (fptr, "layer for succeeding word: %d type: %s\n", &a, word);
            layer1_succ[i]->Read (fptr);
        }
        layers_succ.resize (num_sulayer);
        for (int i=2; i<num_sulayer; i++)
        {
            err = fscanf (fptr, "sulayer %d -> %d type: %s\n", &a, &b, word);
            int nr = succlayersizes[i];
            int nc = succlayersizes[i+1];
            if (i == num_sulayer-1)
            {
                layers_succ[i] = new linearlayer (nr, nc, minibatch, chunksize);
            }
            else
            {
                layers_succ[i] = new feedforwardlayer (nr, nc, minibatch, chunksize);
            }
            layers_succ[i]->Read(fptr);
        }
    }

    err = fscanf (fptr, "%d", &a);
    if (a != CHECKNUM)
    {
        printf ("ERROR: failed to read the check number(%d) when reading model\n", CHECKNUM);
        exit (0);
    }
    if (debug > 1)
    {
        printf ("Successfully loaded model: %s\n", modelname.c_str());
    }
    fclose (fptr);
}

void RNNLM::WriteTextRNNLM_new (string modelname)
{
    int i, a, b;
    float v;
    FILE *fptr = NULL;
    // write to RNNLM model
    fptr = fopen (modelname.c_str(), "w");
    if (fptr == NULL)
    {
        printf ("ERROR: Failed to create RNNLM model file(%s)\n", modelname.c_str());
        exit (0);
    }
    fprintf (fptr, "cuedrnnlm v%.1f\n", version);
    fprintf (fptr, "train file: %s\n", trainfile.c_str());
    fprintf (fptr, "valid file: %s\n", validfile.c_str());
    fprintf (fptr, "number of iteration: %d\n", iter);
    fprintf (fptr, "#train words: %d\n", trainwordcnt);
    fprintf (fptr, "#valid words: %d\n", validwordcnt);
    fprintf (fptr, "#layer: %d\n", num_layer);
    for (i=0; i<layersizes.size(); i++)
    {
        fprintf (fptr, "\tlayer %d size: %d type: %s\n", i, layersizes[i], layertypes[i].c_str());
    }
    fprintf (fptr, "fullvoc size: %d\n", fullvocsize);
    fprintf (fptr, "independent mode: %d\n", independent);
    fprintf (fptr, "train crit mode: %d\n", traincritmode);
    fprintf (fptr, "log norm: %f\n", lognorm_output);
    int dim_fea = layers[0]->getdimfea ();
    fprintf (fptr, "dim feature: %d\n", dim_fea);
    fprintf (fptr, "num of succeeding words: %d\n", succwindowlength);
    if (succwindowlength > 0)
    {
        fprintf (fptr, "num of succlayer: %d\n", num_sulayer);
        for (i=0; i<succlayersizes.size(); i++)
        {
            fprintf (fptr, "succlayer %d size: %d\n", i, succlayersizes[i]);
        }
        fprintf (fptr, "merging layer: %d\n", succmergelayer);
    }

    for (int i=0; i<num_layer; i++)
    {
        fprintf (fptr, "layer %d -> %d type: %s\n", i, i+1, layertypes[i].c_str());
        layers[i]->Write (fptr);
    }

    if (succwindowlength > 0)
    {
        for (i=0; i<succwindowlength; i++)
        {
            fprintf (fptr, "layer for succeeding word: %d type: %s\n", i, layer1_succ[i]->type.c_str());
            layer1_succ[i]->Write (fptr);
        }
        for (i=2; i<num_sulayer; i++)
        {
            fprintf (fptr, "sulayer %d -> %d type: %s\n", i, i+1, layers_succ[i]->type.c_str());
            layers_succ[i]->Write (fptr);
        }
    }

    a = CHECKNUM;
    fprintf (fptr, "%d", a);
    fclose (fptr);
}

// read intput and output word list
void RNNLM::ReadWordlist (string inputlist, string outputlist)
{
    //index 0 for <s> and </s> in input and output layer
    //last node for <OOS>
    int i, a, b;
    float v;
    char word[1024];
    FILE *finlst, *foutlst;
    finlst = fopen (inputlist.c_str(), "r");
    foutlst = fopen (outputlist.c_str(), "r");
    if (finlst == NULL || foutlst == NULL)
    {
        printf ("ERROR: Failed to open input (%s) or output list file(%s)\n", inputlist.c_str(), outputlist.c_str());
        exit (0);
    }
    inputmap.insert(make_pair(string("<s>"), 0));
    outputmap.insert(make_pair(string("</s>"), 0));
    inputvec.clear();
    outputvec.clear();
    inputvec.push_back("<s>");
    outputvec.push_back("</s>");
    int index = 1;
    while (!feof(finlst))
    {
        if(fscanf (finlst, "%d%s", &i, word) == 2)
        {
            if (inputmap.find(word) == inputmap.end())
            {
                inputmap[word] = index;
                inputvec.push_back(word);
                index ++;
               
            }
        }
    }
    if (inputmap.find("<OOS>") == inputmap.end())
    {
        inputmap.insert(make_pair(string("<OOS>"), index));
        inputvec.push_back("<OOS>");
    }
    else
    {
        assert (inputmap["<OOS>"] == inputvec.size()-1);
    }

    index = 1;
    int clsid, prevclsid = 0;
    while (!feof(foutlst))
    {
        if (fscanf(foutlst, "%d%s", &i, word) == 2)
        {
            if (outputmap.find(word) == outputmap.end())
            {
                outputmap[word] = index;
                outputvec.push_back(word);
                index ++;
               
            }
        }
    }
    if (outputmap.find("<OOS>") == outputmap.end())
    {
        outputmap.insert(make_pair(string("<OOS>"), index));
        outputvec.push_back("<OOS>");
    }
    else
    {
        assert (outputmap["<OOS>"] == outputvec.size()-1);
    }
    assert (inputvec.size() == layersizes[0]);
    assert (outputvec.size() == layersizes[num_layer]);
    inStartindex = 0;
    outEndindex  = 0;
    inOOSindex   = inputvec.size() - 1;
    outOOSindex  = outputvec.size() - 1;
    assert (outOOSindex == outputmap["<OOS>"]);
    assert (inOOSindex == inputmap["<OOS>"]);
    fclose (finlst);
    fclose (foutlst);
}
// write to intput and output word list
void RNNLM::WriteWordlist (string inputlist, string outputlist)
{
    int i, a, b;
    float v;
    FILE *finlst, *foutlst;
    finlst = fopen (inputlist.c_str(), "w");
    foutlst = fopen (outputlist.c_str(), "w");
    if (finlst == NULL || foutlst == NULL)
    {
        printf ("ERROR: Failed to create input (%s) or output list file(%s)\n", inputlist.c_str(), outputlist.c_str());
        exit (0);
    }
    for (i=0; i<inputvec.size(); i++)
    {
        fprintf (finlst, "%d\t%s\n", i, inputvec[i].c_str());
    }
    for (i=0; i<outputvec.size(); i++)
    {
        fprintf (foutlst, "%d\t%s\n", i, outputvec[i].c_str());
    }
    assert (inputvec.size() == layersizes[0]);
    assert (outputvec.size() == layersizes[num_layer]);
    fclose (finlst);
    fclose (foutlst);
}

void RNNLM::setTrainCritmode (string traincrit)
{
    if (traincrit == "ce")          traincritmode = 0;
    else if (traincrit == "vr")     traincritmode = 1;
    else if (traincrit == "nce")    traincritmode = 2;
    else    {printf ("ERROR: unrecognized train criterion mode: %s!\n", traincrit.c_str()); exit(0);}
    layers[num_layer-1]->setTrainCrit (traincritmode);
}

void RNNLM::allocSuccwords ()
{
    int i, j;
    if (succwindowlength == 0)  return;

    // Allocate memory
    assert (layersizes[0] == succlayersizes[0]);
    assert (layersizes[1] == succlayersizes[1]);
    int nrows, ncols;
    nrows = layersizes[1];
    ncols = succlayersizes[2];
    num_sulayer = succlayersizes.size() - 1;

    if (isEmpty (inmodelfile))
    // if it is a loaded model, no need to allocate mem for layers
    {
        layer1_succ.resize (succwindowlength);
        for (i=0; i<succwindowlength; i++)
        {
            layer1_succ[i] = new feedforwardlayer (nrows, ncols, minibatch, chunksize);
            layer1_succ[i]->setnodetype (0);
        }

        layers_succ.resize (num_sulayer);
        for (i=2; i<num_sulayer; i++)
        {
            nrows = succlayersizes[i];
            ncols = succlayersizes[i+1];
            // the merging layer is linear layer
            if (i == num_sulayer-1)
            {
                layers_succ[i] = new linearlayer (nrows, ncols, minibatch, chunksize);
            }
            else
            {
                layers_succ[i] = new feedforwardlayer (nrows, ncols, minibatch, chunksize);
                layers_succ[i]->setnodetype(0);
            }
        }
    }

    neu_ac_succ_chunk.resize (succlayersizes.size());
    neu_er_succ.resize(layersizes.size());
    neu_ac_succ.resize(layersizes.size());

    for (i=2; i<succlayersizes.size(); i++)
    {
        nrows = succlayersizes[i];
        neu_ac_succ_chunk[i].resize(chunksize);
        for (j=0; j<chunksize; j++)
        {
            neu_ac_succ_chunk[i][j] = new matrix (nrows, minibatch, flag_usegpu);
        }
        neu_er_succ[i] = new matrix (nrows, minibatch, flag_usegpu);
        neu_ac_succ[i] = new matrix (nrows, minibatch, flag_usegpu);
    }

    nrows = layersizes[1];
    neu_ac1_succ_chunk.resize(succwindowlength);
    neu_er1_succ.resize (succwindowlength);
    neu_ac1_succ.resize (succwindowlength);
    for (i=0; i<succwindowlength; i++)
    {
        neu_ac1_succ_chunk[i].resize(chunksize);
        for (j=0; j<chunksize; j++)
        {
            neu_ac1_succ_chunk[i][j] = new matrix (nrows, minibatch, flag_usegpu);
        }
        neu_er1_succ[i] = new matrix (nrows, minibatch, flag_usegpu);
        neu_ac1_succ[i] = new matrix (nrows, minibatch, flag_usegpu);
    }

    // allocate the memory for succeeding words
    host_succwords = (int **)calloc (succwindowlength, sizeof(int *));
    dev_succwords  = (int **)calloc (succwindowlength, sizeof(int *));
    for (int i=0; i<succwindowlength; i++)
    {
        host_succwords[i] = (int *)calloc (minibatch*chunksize, sizeof(int));
        dev_succwords[i]  = (int *)cucalloc (minibatch*chunksize*sizeof(int));
    }
    succwords = (int *)calloc(succwindowlength, sizeof(int));
}

void RNNLM::setLRtunemode (string lrtune)
{
    if      (lrtune == "newbob")    lrtunemode = 0;
    else if (lrtune == "adagrad")   lrtunemode = 1;
    else if (lrtune == "rmsprop")   lrtunemode = 2;
    else
    {
        printf ("ERROR: unrecognized learn rate tune mode: %s!\n", lrtune.c_str()); exit(0);
    }
    for (int l=0; l<num_layer; l++)
    {
        layers[l]->setLRtunemode (lrtunemode);
    }
}

void RNNLM::setIndependentmode (int v)
{
    independent = v;
}

void RNNLM::setFullVocsize (int n)
{
    if (n == 0)
    {
        fullvocsize = layersizes[0];
    }
    else
    {
        fullvocsize = n;
    }
}

void RNNLM::parseModelTopo (vector<string> &modelstructure)
{
    int i, j;
    char str[1024];
    char ch;
    layersizes.resize (modelstructure.size());
    layertypes.resize (modelstructure.size());
    for (i=0; i<modelstructure.size(); i++)
    {
        j = 0;
        ch = modelstructure[i][j];
        while (ch >= '0' && ch <= '9')
        {
            str[j] = ch;
            j ++;
            ch = modelstructure[i][j];
        }
        str[j] = '\0';
        layersizes[i] = atoi (str);
        if (i > 0)
        {
            if (j < modelstructure[i].length())
            {
                layertypes[i-1] = modelstructure[i].substr(j);
            }
        }
    }
    layertypes[i-1] = "NULL";
}



bool RNNLM::calppl (string testfilename, float intpltwght, string nglmfile)
{
    int i, j, wordcn, nwordoov, cnt;
    vector<string> linevec;
    FILEPTR fileptr;
    float prob_rnn, prob_ng, prob_int, logp_rnn,
          logp_ng, logp_int, ppl_rnn, ppl_ng,
          ppl_int;
    bool flag_intplt = false, flag_oov = false;
    FILE *fptr=NULL, *fptr_nglm=NULL;
    auto_timer timer;
    timer.start();
    string word;
    testfile = testfilename;
    nglmstfile = nglmfile;
    lambda = intpltwght;
    if (debug > 1)
    {
        printPPLInfo ();
    }

    if (!nglmfile.empty())
    {
        fptr_nglm = fopen (nglmfile.c_str(), "r");
        if (fptr_nglm == NULL)
        {
            printf ("ERROR: Failed to open ng stream file: %s\n", nglmfile.c_str());
            exit (0);
        }
        flag_intplt = true;
    }
    fileptr.open(testfile);

    wordcn = 0;
    nwordoov = 0;
    logp_int = 0;
    logp_rnn = 0;
    logp_ng = 0;
    if (debug > 1)
    {
        if (flag_intplt)
        {
            printf ("\nId\tP_rnn\t\tP_ng\t\tP_int\t\tWord\n");
        }
        else
        {
            printf ("\nId\tP_rnn\t\tWord\n");
        }
    }
    while (!fileptr.eof())
    {
        if (layers[0]->getdimfea() > 0)
        {
            int feaid = fileptr.readint ();
            assert (feaid >= 0 && feaid < layers[0]->getnumfea());
            layers[0]->host_assignFeaVec (feaid);
        }
        fileptr.readline(linevec, cnt);
        if (linevec.size() > 0)
        {
            if (linevec[cnt-1] != "</s>")
            {
                linevec.push_back("</s>");
                cnt ++;
            }
            assert (cnt == linevec.size());
            if (linevec[0] == "<s>")    i = 1;
            else                        i = 0;
            prevword = inStartindex;
            host_HandleSentEnd_fw ();

            for (i; i<cnt; i++)
            {
                word = linevec[i];
                if (outputmap.find(word) == outputmap.end())
                {
                    curword = outOOSindex;
                }
                else
                {
                    curword = outputmap[word];
                }

                // for succeeding words
                string sword;
                for (int widx=0; widx<succwindowlength; widx++)
                {
                    int index = i+1+widx;
                    if (index >= cnt)
                    {
                        succwords[widx] = INPUT_PAD_INT;
                    }
                    else
                    {
                        sword = linevec[i+1+widx];
                        if (inputmap.find(word) == inputmap.end())
                        {
                            succwords[widx] = inOOSindex;
                        }
                        else
                        {
                            succwords[widx] = inputmap[sword];
                        }
                    }
                }

                prob_rnn = host_forward (prevword, curword);
                if (curword == outOOSindex)     prob_rnn /= (fullvocsize-layersizes[num_layer]+1);

                if (flag_intplt)
                {
                    if (fscanf (fptr_nglm, "%f\n", &prob_ng) != 1)
                    {
                        printf ("ERROR: Failed to read ngram prob from ng stream file!\n");
                        exit (0);
                    }
                    if (fabs(prob_ng) < 1e-9)   {flag_oov = true;}
                    else                        flag_oov = false;
                }
                prob_int = lambda*prob_rnn + (1-lambda)*prob_ng;
                if (inputmap.find(word) == inputmap.end())
                {
                    prevword = inOOSindex;
                }
                else
                {
                    prevword = inputmap[word];
                }
                if (!flag_oov)
                {
                    logp_rnn += log10(prob_rnn);
                    logp_ng  += log10(prob_ng);
                    logp_int += log10(prob_int);
                }
                else
                {
                    nwordoov ++;
                }
                wordcn ++;
                if (debug > 1)
                {
                    if (flag_intplt)
                        printf ("%d\t%.10f\t%.10f\t%.10f\t%s", curword, prob_rnn, prob_ng, prob_int, word.c_str());
                    else
                        printf ("%d\t%.10f\t%s", curword, prob_rnn, word.c_str());
                    if (curword == outOOSindex)
                    {
                        if (flag_oov)   printf ("<OOV>");
                        else            printf ("<OOS>");
                    }
                    printf ("\n");
                }
                if (debug > 2)
                {
                    if (wordcn % 10000 == 0)
                    {
                        float nwordspersec = wordcn / (timer.stop());
                        printf ("eval speed  %.4f Words/sec\n", nwordspersec);
                    }
                }
            }
        }
    }
    if (debug > 2)
    {
        float nwordspersec = wordcn / (timer.stop());
        printf ("eval speed  %.4f Words/sec\n", nwordspersec);
    }
    ppl_rnn = exp10(-logp_rnn/(wordcn-nwordoov));
    ppl_ng  = exp10(-logp_ng/(wordcn-nwordoov));
    ppl_int = exp10(-logp_int/(wordcn-nwordoov));
    if (flag_intplt)
    {
        printf ("Total word: %d\tOOV word: %d\n", wordcn, nwordoov);
        printf ("N-Gram log probability: %.3f\n", logp_ng);
        printf ("RNNLM  log probability: %.3f\n", logp_rnn);
        printf ("Intplt log probability: %.3f\n\n", logp_int);
        printf ("N-Gram PPL : %.3f\n", ppl_ng);
        printf ("RNNLM  PPL : %.3f\n", ppl_rnn);
        printf ("Intplt PPL : %.3f\n", ppl_int);
    }
    else
    {
        printf ("Total word: %d\tOOV word: %d\n", wordcn, nwordoov);
        printf ("Average logp: %f\n", logp_rnn/log10(2)/wordcn);
        printf ("RNNLM  log probability: %.3f\n", logp_rnn);
        printf ("RNNLM  PPL : %.3f\n", ppl_rnn);
    }
    fileptr.close();

    if (fptr_nglm)
    {
        fclose(fptr_nglm);
    }
    return SUCCESS;
}

bool RNNLM::calnbest (string testfilename, float intpltwght, string nglmfile)
{
    int i, j, wordcn, cnt, nbestid, prevnbestid=-1, sentcnt=0, nword;
    vector<string> linevec, maxlinevec;
    FILEPTR fileptr;
    float prob_rnn, prob_ng, prob_int, logp_rnn,
          logp_ng, logp_int, ppl_rnn, ppl_ng,
          ppl_int, sentlogp, acscore, lmscore, score, maxscore;
    bool flag_intplt = false;
    FILE *fptr=NULL, *fptr_nglm=NULL;
    auto_timer timer;
    timer.start();
    string word;
    testfile = testfilename;
    nglmstfile = nglmfile;
    lambda = intpltwght;
    if (debug > 1)
    {
        printPPLInfo ();
    }
    if (!nglmfile.empty())
    {
        fptr_nglm = fopen (nglmfile.c_str(), "r");
        if (fptr_nglm == NULL)
        {
            printf ("ERROR: Failed to open ng stream file: %s\n", nglmfile.c_str());
            exit (0);
        }
        flag_intplt = true;
    }
    fileptr.open(testfile);

    wordcn = 0;
    logp_int = 0;
    logp_rnn = 0;
    logp_ng = 0;
    while (!fileptr.eof())
    {
        if (layers[0]->getdimfea() > 0)
        {
            int feaid = fileptr.readint ();
            assert (feaid >= 0 && feaid < layers[0]->getnumfea());
            layers[0]->host_assignFeaVec (feaid);
        }
        fileptr.readline(linevec, cnt);
        if (linevec.size() > 0)
        {
            if (linevec[cnt-1] != "</s>")
            {
                linevec.push_back("</s>");
                cnt ++;
            }
            assert (cnt == linevec.size());
            // the first two iterms for linevec are: <s> nbestid
            // nbid acscore  lmscore  nword <s> ... </s>
            // 0    2750.14 -6.03843    2   <s> HERE YEAH </s>
            // erase the first <s> and last</s>
            vector<string>::iterator it= linevec.begin();
            linevec.erase(it);
            cnt --;
            // it = linevec.end();
            // it --;
            // linevec.erase(it);
            nbestid = string2int(linevec[0]);

            if (nbestid != prevnbestid)
            {
                if (prevnbestid != -1)
                {
                    for (i=4; i<maxlinevec.size(); i++)
                    {
                        word = maxlinevec[i];
                        if (word != "<s>" && word != "</s>")
                        {
                            printf (" %s", word.c_str());
                        }
                    }
                    printf ("\n");
                }
                maxscore = -1e10;
            }

            acscore = string2float(linevec[1]);
            lmscore = string2float(linevec[2]);
            nword   = string2int(linevec[3]);
            if (linevec[4] == "<s>")    i = 5;
            else                        i = 4;
            sentlogp = 0;
            prevword = inStartindex;
            host_HandleSentEnd_fw ();
            for (i; i<cnt; i++)
            {
                word = linevec[i];
                if (outputmap.find(word) == outputmap.end())
                {
                    curword = outOOSindex;
                }
                else
                {
                    curword = outputmap[word];
                }

                // for succeeding words
                string sword;
                for (int widx=0; widx<succwindowlength; widx++)
                {
                    int index = i+1+widx;
                    if (index >= cnt)
                    {
                        succwords[widx] = INPUT_PAD_INT;
                    }
                    else
                    {
                        sword = linevec[i+1+widx];
                        if (inputmap.find(word) == inputmap.end())
                        {
                            succwords[widx] = inOOSindex;
                        }
                        else
                        {
                            succwords[widx] = inputmap[sword];
                        }
                    }
                }

                prob_rnn = host_forward (prevword, curword);
                if (curword == outOOSindex)     prob_rnn /= (fullvocsize-layersizes[num_layer]+1);

                if (flag_intplt)
                {
                    if (fscanf (fptr_nglm, "%f\n", &prob_ng) != 1)
                    {
                        printf ("ERROR: Failed to read ngram prob from ng stream file!\n");
                        exit (0);
                    }
                }
                prob_int = lambda*prob_rnn + (1-lambda)*prob_ng;
                if (inputmap.find(word) == inputmap.end())
                {
                    prevword = inOOSindex;
                }
                else
                {
                    prevword = inputmap[word];
                }
                logp_rnn += log10(prob_rnn);
                logp_ng  += log10(prob_ng);
                logp_int += log10(prob_int);
                sentlogp += log10(prob_int);
                wordcn ++;
                if (debug == 1)
                {
                    printf ("%f ", log10(prob_int));
                }
                if (debug > 1)
                {
                    if (flag_intplt)
                        printf ("%d\t%.10f\t%.10f\t%.10f\t%s", curword, prob_rnn, prob_ng, prob_int, word.c_str());
                    else
                        printf ("%d\t%.10f\t%s", curword, prob_rnn, word.c_str());
                    if (curword == outOOSindex)
                    {
                        printf ("<OOS>");
                    }
                    printf ("\n");
                }
                if (debug > 1)
                {
                    if (wordcn % 10000 == 0)
                    {
                        float nwordspersec = wordcn / (timer.stop());
                        printf ("eval speed  %.4f Words/sec\n", nwordspersec);
                    }
                }
            }
            sentcnt ++;
            if (debug == 1)
            {
                printf ("sent=%f %d\n", sentlogp, sentcnt);
            }
#if 0
            if (debug == 0)
            {
                printf ("%f\n", sentlogp);
            }
#endif
            if (debug == 0)
            {
                score = acscore + sentlogp*lmscale + ip*(nword+1);
                if (score > maxscore)
                {
                    maxscore = score;
                    maxlinevec = linevec;
                }
                for (i=4; i<cnt; i++)
                {
                    word = linevec[i];
                    // printf (" %s", word.c_str());
                }
                // printf ("\n");
            }
            fflush(stdout);
            prevnbestid = nbestid;
        }
    }
    for (i=4; i<maxlinevec.size(); i++)
    {
        word = maxlinevec[i];
        if (word != "<s>" && word != "</s>")
        {
            printf (" %s", word.c_str());
        }
    }
    printf ("\n");
    if (debug > 1)
    {
        float nwordspersec = wordcn / (timer.stop());
        printf ("eval speed  %.4f Words/sec\n", nwordspersec);
    }
    ppl_rnn = exp10(-logp_rnn/(wordcn));
    ppl_ng  = exp10(-logp_ng/(wordcn));
    ppl_int = exp10(-logp_int/(wordcn));
    if (debug > 1)
    {
        if (flag_intplt)
        {
            printf ("Total word: %d\n", wordcn);
            printf ("N-Gram log probability: %.3f\n", logp_ng);
            printf ("RNNLM  log probability: %.3f\n", logp_rnn);
            printf ("Intplt log probability: %.3f\n\n", logp_int);
            printf ("N-Gram PPL : %.3f\n", ppl_ng);
            printf ("RNNLM  PPL : %.3f\n", ppl_rnn);
            printf ("Intplt PPL : %.3f\n", ppl_int);
        }
        else
        {
            printf ("Total word: %d\n", wordcn);
            printf ("Average logp: %f\n", logp_rnn/log10(2)/wordcn);
            printf ("RNNLM  log probability: %.3f\n", logp_rnn);
            printf ("RNNLM  PPL : %.3f\n", ppl_rnn);
        }
    }
    fileptr.close();

    if (fptr_nglm)
    {
        fclose(fptr_nglm);
    }
    return SUCCESS;
}

bool RNNLM::sample (string textfile, string unigramfile, int n)
{
    int i, mb, word, sentlen;
    float g, f;
    string wordstr;
    ReadUnigramFile (unigramfile);
    sampletextfile = textfile;
    uglmfile = unigramfile;
    nsample = n;
    FILE **fptr_txts;
    fptr_txts = (FILE **) malloc (sizeof(FILE *) * minibatch);
    char txtfilestr[1024][1024];
    srand (rand_seed);
    if (debug > 1)
    {
        printSampleInfo ();
    }
    for (int i=0; i<minibatch; i++)
    {
        sprintf (txtfilestr[i], "%s.mb%d", textfile.c_str(), i);
        fptr_txts[i] = fopen (txtfilestr[i], "w");
    }
    for (i=0; i<minibatch; i++)
    {
        host_prevwords[i] = inStartindex;
        host_curwords[i] = outEndindex;
    }
    cucpytoGpu (dev_prevwords, host_prevwords, minibatch*sizeof(int));
    cucpytoGpu (dev_curwords, host_curwords, minibatch*sizeof(int));
    wordcn = 0;
    int nvalidmbsize = minibatch;

    int *dev_samples = (int *)cucalloc (minibatch*sizeof(int));
    int *host_sample = (int *)calloc(minibatch, sizeof(int));
    float *dev_randv = (float *)cucalloc(minibatch*sizeof(float));
    float *host_randv = (float *)calloc(minibatch, sizeof(float));

    initHiddenAc ();
    // while (wordcn < nsample)
    while (nvalidmbsize > 0)
    {
        cuChunkForward ();
        // using GPU for sampling
        for (i=0; i<minibatch; i++)
        {
            host_randv[i] = randomv(0, 1);
        }
        cucpytoGpu(dev_randv, host_randv, minibatch*sizeof(float));

        neu_ac[num_layer]->sample(dev_samples, dev_randv, minibatch);

        cucpyfromGPU (host_sample, dev_samples, minibatch*sizeof(int));
        for (int mb=0; mb<minibatch; mb++)
        {
            if (host_prevwords[mb] == INVALID_INT || host_curwords[mb] == INVALID_INT)
            {
                continue;
            }
            word = host_sample[mb];
            if (word == outOOSindex)
            {
                f = randomv(0, 1);
                g = 0;
                i = 0;
                while ((g<f) && (i < num_oosword))
                {
                    g += ooswordsprob[i];
                    i ++;
                }
                word = i - 1;
                wordstr = ooswordsvec[word].c_str();
                if (inputmap.find(wordstr) == inputmap.end())
                {
                    host_prevwords[mb] = inOOSindex;
                }
                else
                {
                    host_prevwords[mb] = inputmap[wordstr];
                }
                fprintf (fptr_txts[mb], " %s", wordstr.c_str());
            }
            else if (word != outEndindex)
            {
                wordcn ++;

                wordstr = outputvec[word].c_str();
                if (inputmap.find(wordstr) == inputmap.end())
                {
                    host_prevwords[mb] = inOOSindex;
                }
                else
                {
                    host_prevwords[mb] = inputmap[wordstr];
                }
                fprintf (fptr_txts[mb], " %s", wordstr.c_str());
            }
            else
            {
                host_prevwords[mb] = inStartindex;
                wordstr = "\n";
                if (wordcn > nsample)
                {
                    host_prevwords[mb] = INVALID_INT;
                    host_curwords[mb] = INVALID_INT;
                    nvalidmbsize --;
                }
                fprintf (fptr_txts[mb], "%s", wordstr.c_str());
            }
        }
        cucpytoGpu (dev_prevwords, host_prevwords, minibatch*sizeof(int));
    }

    char *line = (char *)malloc(MAX_WORD_LINE * sizeof(char));
    int max_words_line = MAX_WORD_LINE;
    FILE *fptr = fopen (textfile.c_str(), "w");
    for (i=0; i<minibatch; i++)
    {
        fclose (fptr_txts[i]);
        fptr_txts[i] = NULL;
        fptr_txts[i] = fopen(txtfilestr[i], "r");
        if (fptr_txts[i] == NULL)
        {
            printf ("failed to open text file: %s\n", txtfilestr[i]);
            exit (0);
        }
        while (!feof(fptr_txts[i]))
        {
            sentlen = getline (line, max_words_line, fptr_txts[i]);
            if (sentlen > 1)
            {
                fprintf (fptr, "%s", line);
            }
        }
        fclose (fptr_txts[i]);
        if (remove(txtfilestr[i]) == 0)
        {
            printf ("Removed %s.\n", txtfilestr[i]);
        }
        else
        {
            printf ("ERROR: failed to remove %s\n", txtfilestr[i]);
        }
        fflush(stdout);
        fflush(fptr);
    }
    fclose (fptr);
    if (line)  free (line);
    return SUCCESS;
}

void RNNLM::ReadUnigramFile (string unigramfile)
{
    FILE *fin = fopen (unigramfile.c_str(), "r");
    int err;
    char line[MAX_STRING];
    char line2[MAX_STRING];
    float prob = 0.0, logprob = 0.0;
    float accprob = 0.0;
    char word[MAX_STRING];
    ooswordsvec.clear();
    ooswordsprob.clear();
    err = fscanf (fin, "%s\n", line);  /*  \data\  */
    err = fscanf (fin, "%s%s\n", line, line2);
    err = fscanf (fin, "%sgram:\n", line);
    int i = 0;
    while (! feof(fin))
    {
        if (fscanf (fin, "%f%s", &logprob, word) == 2)
        {
            prob = pow(10, logprob);
            if ((outputmap.find(word) == outputmap.end()) && (word[0] != '!'))
            {
                accprob += prob;
                ooswordsvec.push_back(string(word));
                ooswordsprob.push_back(prob);
                i ++;
            }
        }
        else
        {
            err = fscanf (fin, "%s", word);
        }
    }
    num_oosword = ooswordsprob.size();
    fclose (fin);

    for (i=0; i<num_oosword; i++)
    {
        ooswordsprob[i] = ooswordsprob[i] / accprob;
    }
}

float RNNLM::host_forward (int prevword, int curword)
{
   int i;

   if (succwindowlength > 0)
   {
       neu_ac_succ[2]->hostinitmatrix();
       for (i=0; i<succwindowlength; i++)
       {
           if (succwords[i] != INPUT_PAD_INT)
           {
               layers[0]->host_getWordEmbedding (succwords[i], neu_ac1_succ[i]);
               layer1_succ[i]->host_forward_nosigm (neu_ac1_succ[i], neu_ac_succ[2]);
           }
       }

       neu_ac_succ[2]->hostsigmoid ();

       for (i=2; i<num_sulayer; i++)
       {
           neu_ac_succ[i+1]->hostinitmatrix();
           layers_succ[i]->host_forward (neu_ac_succ[i], neu_ac_succ[i+1]);
       }
   }


   layers[0]->host_getWordEmbedding (prevword, neu_ac[1]);
   for (i=1; i<num_layer; i++)
   {

       neu_ac[i+1]->hostinitmatrix();
       if (succwindowlength > 0 && i+1 == succmergelayer)
       {
           neu_ac[i+1]->hostassign(neu_ac_succ[num_sulayer]);
       }
       layers[i]->host_forward (neu_ac[i], neu_ac[i+1]);
       // if next layer is LSTM-highway layer
       if (i != num_layer-1 && isLSTMhighway (layertypes[i+1]))
       {
           layers[i+1]->host_copyLSTMhighwayc (layers[i]);
       }
   }
   return neu_ac[num_layer]->fetchhostvalue(curword, 0);
}

void RNNLM::setMomentum (float v)
{
    momentum = v;
    for (int l=0; l<num_layer; l++)
    {
        layers[l]->setMomentum (momentum);
    }
}

void RNNLM::setClipping (float v)
{
    gradient_cutoff = v;
    for (int l=0; l<num_layer; l++)
    {
        layers[l]->setClipping (gradient_cutoff);
    }
}

void RNNLM::setL2Regularization (float v)
{
    l2reg = v;
    for (int l=0; l<num_layer; l++)
    {
        layers[l]->setL2Regularization (l2reg);
    }
}

void RNNLM::setVRpenalty (float v)
{
    vrpenalty = v;
    layers[num_layer-1]->setVRpenalty(v);
}

void RNNLM::setLognormConst (float v)
{
    layers[num_layer-1]->setLognormConst (v);
    lognormconst = v;
}


void RNNLM::setDropoutRate (float v)
{
    dropoutrate = v;
    for (int l=0; l<num_layer; l++)
    {
        layers[l]->setDropoutRate (v);
    }
    if (dropoutrate > 0)
    {
        for (int l=0; l<num_layer; l++)
        {
            layers[l]->allocDropoutMask ();
        }
    }
}

void RNNLM::setLearnRate (float v)
{
    for (int l=0; l<num_layer; l++)
    {
        layers[l]->setLearnRate (v);
    }
}

void RNNLM::setRandseed (int n)
{
    rand_seed = n;
#ifdef RAND48
    srand48(rand_seed);
#else
    srand(rand_seed);
#endif
}

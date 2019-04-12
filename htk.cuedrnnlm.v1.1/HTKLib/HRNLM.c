/* ----------------------------------------------------------- */
/*                                                             */
/*                          ___  */
/*                       |_| | |_/   SPEECH                    */
/*                       | | | | \   RECOGNITION               */
/*                       =========   SOFTWARE                  */
/*                                                             */
/*                                                             */
/* ----------------------------------------------------------- */
/* developed at:                                               */
/*                                                             */
/*      Speech Vision and Robotics group                       */
/*      Cambridge University Engineering Department            */
/*      http:// svr-www.eng.cam.ac.uk/                         */
/*                                                             */
/*      Entropic Cambridge Research Laboratory                 */
/*      (now part of Microsoft)                                */
/*                                                             */
/* ----------------------------------------------------------- */
/*         Copyright: Microsoft Corporation                    */
/*          1995-2000 Redmond, Washington USA                  */
/*                    http:// www.microsoft.com                */
/*                                                             */
/*          2001-2002 Cambridge University                     */
/*                    Engineering Department                   */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*     File: HRNLM.c  recurrent neural network language model
 *     handling    */
/* ----------------------------------------------------------- */

char *hrnlm_version = "!HVER!RHNLM:   3.4 [CUED 06/30/13]";
char *hrnlm_vc_id = "$Id: HRNLM.c,v 1.1.1.1 2013/06/30 09:54:57 yw293 Exp $";

#include <math.h>
#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HRNLM.h"

/* --------------------------- Trace Flags ------------------------- */

#define T_TIO 1  /* Progress tracing whilst performing IO */
#define T_TOP 256
#define T_PRO 128   /*  print all probabilities  */

#define HASHTABLESIZE 10000
#define MAXLINELEN      4096
/* #define FASTEXP */

static int trace=128;

static union{
    double d;
    struct{
        int j,i;
        } n;
} d2i;
#define M_LN2		0.69314718055994530942	/* log_e 2 */
#define M_LN10		2.30258509299404568402	/* log_e 10 */
#define EXP_A (1048576/M_LN2)
#define EXP_C 60801
#define FAST_EXP(y) (d2i.n.i = EXP_A*(y)+(1072693248-EXP_C),d2i.d)



/* ---------------- Configuration Parameters --------------------- */

static ConfParam *cParm[MAXGLOBS];
static int nParm = 0;            /* total num params */
static Boolean readClassAllocation = TRUE;
static char* oosNodeWord="<OOS>";
static Boolean inputContainOOS = TRUE;
static Boolean outputContainOOS= TRUE;
static int oosNodeClass=-1;         /*  default : derived implicitly from output list */
static Boolean calcProbCache = TRUE;
static LogFloat outLayerLogNorm = -1;

/* ---------------------- Global Variables ----------------------- */

static MemHeap rnnInfoStack;       /* Local stack to for MLP info */
static Boolean rawMITFormat = TRUE;

/*-----------------------------------------------------------------------------
 *  const data definition
 *-----------------------------------------------------------------------------*/
static unsigned int PRIMES[]={108641969, 116049371, 125925907, 133333309, 145678979, 175308587, 197530793, 234567803, 251851741, 264197411, 330864029, 399999781,
407407183, 459258997, 479012069, 545678687, 560493491, 607407037, 629629243, 656789717, 716048933, 718518067, 725925469, 733332871, 753085943, 755555077,
782715551, 790122953, 812345159, 814814293, 893826581, 923456189, 940740127, 953085797, 985184539, 990122807};
static unsigned int PRIMES_SIZE=sizeof(PRIMES)/sizeof(PRIMES[0]);




/* --------------------------- helper function ---------------------- */

static void goToDelimiter(int delim, FILE *fi)
{
    int ch=0;

    while (ch!=delim) {
        ch=fgetc(fi);
        if (feof(fi)) {
	    exit(1);
        }
    }
}

static void readWord(char *word, FILE *fin)
{
    int a=0, ch;

    while (!feof(fin)) {
	ch=fgetc(fin);

	if (ch==13) continue;

	if ((ch==' ') || (ch=='\t') || (ch=='\n')) {
    	    if (a>0) {
                if (ch=='\n') ungetc(ch, fin);
                break;
            }

            if (ch=='\n') {
                strcpy(word, (char *)"</s>");
                return;
            }
            else continue;
        }

        word[a]=ch;
        a++;

        if (a>=MAX_STRING) {
            /*printf("Too long word found!\n");
             * * truncate too long words */
            a--;
        }
    }
    word[a]=0;
}

static real rnnrandom(real min, real max)
{
    return rand()/(real)RAND_MAX*(max-min)+min;
}

static void saveWeights(RNNLM* rnnlm)
{
    int a,b;

    for (a=0; a<rnnlm->layer0_size; a++) {
        rnnlm->neu0b[a].ac=rnnlm->neu0[a].ac;
        rnnlm->neu0b[a].er=rnnlm->neu0[a].er;
    }

    for (a=0; a<rnnlm->layer1_size; a++) {
        rnnlm->neu1b[a].ac=rnnlm->neu1[a].ac;
        rnnlm->neu1b[a].er=rnnlm->neu1[a].er;
    }

    for (a=0; a<rnnlm->layerc_size; a++) {
        rnnlm->neucb[a].ac=rnnlm->neuc[a].ac;
        rnnlm->neucb[a].er=rnnlm->neuc[a].er;
    }

    for (a=0; a<rnnlm->layer2_size; a++) {
        rnnlm->neu2b[a].ac=rnnlm->neu2[a].ac;
        rnnlm->neu2b[a].er=rnnlm->neu2[a].er;
    }

    for (b=0; b<rnnlm->layer1_size; b++)
        for (a=0; a<rnnlm->layer0_size; a++) {
            rnnlm->syn0b[a+b*rnnlm->layer0_size].weight=
                rnnlm->syn0[a+b*rnnlm->layer0_size].weight;
    }

    if (rnnlm->layerc_size>0)
    {
        for (b=0; b<rnnlm->layerc_size; b++)
            for (a=0; a<rnnlm->layer1_size; a++) {
            rnnlm->syn1b[a+b*rnnlm->layer1_size].weight=rnnlm->syn1[a+b*rnnlm->layer1_size].weight;
        }

        for (b=0; b<rnnlm->layer2_size; b++)
            for (a=0; a<rnnlm->layerc_size; a++) {
                rnnlm->syncb[a+b*rnnlm->layerc_size].weight=rnnlm->sync[a+b*rnnlm->layerc_size].weight;
            }
    }
    else {
        for (b=0; b<rnnlm->layer2_size; b++)
            for (a=0; a<rnnlm->layer1_size; a++) {
                rnnlm->syn1b[a+b*rnnlm->layer1_size].weight=rnnlm->syn1[a+b*rnnlm->layer1_size].weight;
        }
    }

    /*for (a=0; a<direct_size; a++) syn_db[a].weight=syn_d[a].weight; */
}


static void MakeVocabularyClass(RNNLM* rnnlm)
{
    int i;
    double df=0;
    double dd=0;
    int a=0;
    int b=0;
    int cl=0;
    int* ptmp = NULL;

    if (rnnlm->old_classes) {  	/* old classes */
        for (i=0; i<rnnlm->vocab_size; i++) b+=rnnlm->vocab[i].cn;
        for (i=0; i<rnnlm->vocab_size; i++) {
            df+=rnnlm->vocab[i].cn/(double)b;
            if (df>1) df=1;
            if (df>(a+1)/(double)rnnlm->class_size)
            {
                rnnlm->vocab[i].class_index=a;
                if (a<rnnlm->class_size-1) a++;
            }
            else
            {
                rnnlm->vocab[i].class_index=a;
            }
        }
    }
    else {			/* new classes */
        if ( readClassAllocation == FALSE ) {
            for (i=0; i<rnnlm->vocab_size; i++) b+=rnnlm->vocab[i].cn;
            for (i=0; i<rnnlm->vocab_size; i++) dd+=sqrt(rnnlm->vocab[i].cn/(double)b);
            for (i=0; i<rnnlm->vocab_size; i++) {
                df+=sqrt(rnnlm->vocab[i].cn/(double)b)/dd;
                if (df>1) df=1;
                if (df>(a+1)/(double)rnnlm->class_size) {
                    rnnlm->vocab[i].class_index=a;
                    if (a<rnnlm->class_size-1) a++;
                }
                else
                {
                    rnnlm->vocab[i].class_index=a;
                }
            }
        }
    }

    /*allocate auxiliary class variables (for faster search when normalizing probability at output layer) */

    rnnlm->class_words=New(rnnlm->x, sizeof(int**)*rnnlm->class_size);
    rnnlm->class_cn=New(rnnlm->x, sizeof(int)*rnnlm->class_size);
    rnnlm->class_max_cn=New(rnnlm->x, sizeof(int)*rnnlm->class_size);

    for (i=0; i<rnnlm->class_size; i++) {
        rnnlm->class_cn[i]=0;
        rnnlm->class_max_cn[i]=10;
        rnnlm->class_words[i]=(int *)New(rnnlm->x, rnnlm->class_max_cn[i]*sizeof(int));
    }

    for (i=0; i<rnnlm->vocab_size; i++) {
        cl=rnnlm->vocab[i].class_index;
        rnnlm->class_words[cl][rnnlm->class_cn[cl]]=i;
        rnnlm->class_cn[cl]++;
        if (rnnlm->class_cn[cl]+2>=rnnlm->class_max_cn[cl]) {
            rnnlm->class_max_cn[cl]+=10;
            ptmp=(int*)New(rnnlm->x, rnnlm->class_max_cn[cl]*sizeof(int));
            for (a=0; a<rnnlm->class_cn[cl] ; a++)
                ptmp[a]=rnnlm->class_words[cl][a];
            rnnlm->class_words[cl]=ptmp;
        }
    }
}

#if 0
static void initFNet(RNNLM* rnnlm)
{
    int a, b;
    long long aa;

    /*-----------------------------------------------------------------------------
     *  neuron allocation
     *-----------------------------------------------------------------------------*/
    if (rnnlm->layer0_size )
    {
        rnnlm->neu0=(struct neuron *)New(rnnlm->x,  sizeof(struct neuron)*rnnlm->layer0_size);
        rnnlm->neu0b=(struct neuron *)New(rnnlm->x,  sizeof(struct neuron)*rnnlm->layer0_size);
    }
    if (rnnlm->layer1_size )
    {
        rnnlm->neu1=(struct neuron *)New(rnnlm->x,  sizeof(struct neuron)*rnnlm->layer1_size);
        rnnlm->neu1b=(struct neuron *)New(rnnlm->x,  sizeof(struct neuron)*rnnlm->layer1_size);
        rnnlm->neu1b2=(struct neuron *)New(rnnlm->x,  sizeof(struct neuron)*rnnlm->layer1_size);
    }
    if (rnnlm->layerc_size )
    {
        rnnlm->neuc=(struct neuron *)New(rnnlm->x,  sizeof(struct neuron)*rnnlm->layerc_size);
        rnnlm->neucb=(struct neuron *)New(rnnlm->x,  sizeof(struct neuron)*rnnlm->layerc_size);
    }
    if (rnnlm->layer2_size )
    {
        rnnlm->neu2=(struct neuron *)New(rnnlm->x,  sizeof(struct neuron)*rnnlm->layer2_size);
        rnnlm->neu2b=(struct neuron *)New(rnnlm->x,  sizeof(struct neuron)*rnnlm->layer2_size);
    }



    /*-----------------------------------------------------------------------------
     *  prob table calculation
     *-----------------------------------------------------------------------------*/

    if ( calcProbCache == TRUE ) {
        rnnlm->outP_arr = New(rnnlm->x, sizeof(float)*rnnlm->out_num_word );
    }

    /*-----------------------------------------------------------------------------
     *  synapse allocation
     *-----------------------------------------------------------------------------*/

    if (rnnlm->layer0_size * rnnlm->layer1_size ) {
        rnnlm->syn0=(struct synapse *)New(rnnlm->x, rnnlm->layer0_size*rnnlm->layer1_size*sizeof(struct synapse));
        rnnlm->syn0b=(struct synapse *)New(rnnlm->x, rnnlm->layer0_size*rnnlm->layer1_size*sizeof(struct synapse));
    }

    if (rnnlm->layerc_size==0)
    {
	    rnnlm->syn1=(struct synapse *)New(rnnlm->x, rnnlm->layer2_size * rnnlm->layer1_size*sizeof(struct synapse));
        rnnlm->syn1b=(struct synapse *)New(rnnlm->x, rnnlm->layer2_size * rnnlm->layer1_size*sizeof(struct synapse));
    }
    else
    {
	    rnnlm->syn1=(struct synapse *)New(rnnlm->x, rnnlm->layer1_size*rnnlm->layerc_size*sizeof(struct synapse));
	    rnnlm->sync=(struct synapse *)New(rnnlm->x, rnnlm->layer2_size*rnnlm->layerc_size*sizeof(struct synapse));
        rnnlm->syn1b=(struct synapse *)New(rnnlm->x, rnnlm->layer1_size*rnnlm->layerc_size*sizeof(struct synapse));
	    rnnlm->syncb=(struct synapse *)New(rnnlm->x, rnnlm->layer2_size*rnnlm->layerc_size*sizeof(struct synapse));
    }

    if (rnnlm->syn1==NULL) {
	    printf("Memory allocation failed\n");
	    exit(1);
    }

    if (rnnlm->layerc_size>0)
        if (rnnlm->sync==NULL) {
	    printf("Memory allocation failed\n");
	    exit(1);
    }

    if ( rnnlm->direct_size ) {

        rnnlm->syn_d=(direct_t *)New(rnnlm->x, rnnlm->direct_size*sizeof(direct_t));

        if (rnnlm->syn_d==NULL) {
            printf("Memory allocation for direct connections failed (requested %lld bytes)\n", (long long)rnnlm->direct_size *sizeof(direct_t));
            exit(1);
        }

    }

    /*-----------------------------------------------------------------------------
     *  Initialisation
     *-----------------------------------------------------------------------------*/
    for (a=0; a<rnnlm->layer0_size; a++) {
        rnnlm->neu0[a].ac=0;
        rnnlm->neu0[a].er=0;
    }

    for (a=0; a<rnnlm->layer1_size; a++) {
        rnnlm->neu1[a].ac=0;
        rnnlm->neu1[a].er=0;
    }

    for (a=0; a<rnnlm->layerc_size; a++) {
        rnnlm->neuc[a].ac=0;
        rnnlm->neuc[a].er=0;
    }

    for (a=0; a<rnnlm->layer2_size; a++) {
        rnnlm->neu2[a].ac=0;
        rnnlm->neu2[a].er=0;
    }

    for (b=0; b<rnnlm->layer1_size; b++)
        for (a=0; a<rnnlm->layer0_size; a++) {
        rnnlm->syn0[a+b*rnnlm->layer0_size].weight=rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1);
    }

    if (rnnlm->layerc_size>0) {
        for (b=0; b<rnnlm->layerc_size; b++)
            for (a=0; a<rnnlm->layer1_size; a++) {
                rnnlm->syn1[a+b*rnnlm->layer1_size].weight=rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1);
            }

        for (b=0; b<rnnlm->layer2_size; b++)
            for (a=0; a<rnnlm->layerc_size; a++) {
                rnnlm->sync[a+b*rnnlm->layerc_size].weight=
                    rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1);
            }
    }
    else
    {
        for (b=0; b<rnnlm->layer2_size; b++)
            for (a=0; a<rnnlm->layer1_size; a++)
            {
                rnnlm->syn1[a+b*rnnlm->layer1_size].weight=rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1);
            }
    }

    for (aa=0; aa<rnnlm->direct_size; aa++) rnnlm->syn_d[aa]=0;

    if (rnnlm->bptt>0) {
        if (trace & T_TOP)
	        printf("WARNING: use RNNLM in HTK does not support training at this point");
    }

    saveWeights(rnnlm);
    /* read topic matrix file */
    if (rnnlm->layer0_topic_size > 0)
       {
          ReadTopicMatrixFile (rnnlm);
       }
}
#endif

static void initNet(RNNLM* rnnlm)
{
    int a, b;
    long long aa;

    /*-----------------------------------------------------------------------------
     *  neuron allocation
     *-----------------------------------------------------------------------------*/
    if (rnnlm->layer0_size )
    {
        rnnlm->neu0=(struct neuron *)New(rnnlm->x,  sizeof(struct neuron)*rnnlm->layer0_size);
        rnnlm->neu0b=(struct neuron *)New(rnnlm->x,  sizeof(struct neuron)*rnnlm->layer0_size);
    }
    if (rnnlm->layer1_size )
    {
        rnnlm->neu1=(struct neuron *)New(rnnlm->x,  sizeof(struct neuron)*rnnlm->layer1_size);
        rnnlm->neu1b=(struct neuron *)New(rnnlm->x,  sizeof(struct neuron)*rnnlm->layer1_size);
        rnnlm->neu1b2=(struct neuron *)New(rnnlm->x,  sizeof(struct neuron)*rnnlm->layer1_size);
    }
    if (rnnlm->layerc_size )
    {
        rnnlm->neuc=(struct neuron *)New(rnnlm->x,  sizeof(struct neuron)*rnnlm->layerc_size);
        rnnlm->neucb=(struct neuron *)New(rnnlm->x,  sizeof(struct neuron)*rnnlm->layerc_size);
    }
    if (rnnlm->layer2_size )
    {
        rnnlm->neu2=(struct neuron *)New(rnnlm->x,  sizeof(struct neuron)*rnnlm->layer2_size);
        rnnlm->neu2b=(struct neuron *)New(rnnlm->x,  sizeof(struct neuron)*rnnlm->layer2_size);
    }



    /*-----------------------------------------------------------------------------
     *  prob table calculation
     *-----------------------------------------------------------------------------*/

    if ( calcProbCache == TRUE ) {
        rnnlm->outP_arr = New(rnnlm->x, sizeof(float)*rnnlm->out_num_word );
    }

    /*-----------------------------------------------------------------------------
     *  synapse allocation
     *-----------------------------------------------------------------------------*/

    if (rnnlm->layer0_size * rnnlm->layer1_size ) {
        rnnlm->syn0=(struct synapse *)New(rnnlm->x, rnnlm->layer0_size*rnnlm->layer1_size*sizeof(struct synapse));
        rnnlm->syn0b=(struct synapse *)New(rnnlm->x, rnnlm->layer0_size*rnnlm->layer1_size*sizeof(struct synapse));
    }

    if (rnnlm->layer0_topic_size > 0)
       {
          rnnlm->syn0_topic = (struct synapse *) New (rnnlm->x, rnnlm->layer0_topic_size*rnnlm->layer1_size*sizeof(struct synapse));
          rnnlm->syn_d_topic = (struct synapse *) New (rnnlm->x, rnnlm->layer0_topic_size*rnnlm->layer2_size*sizeof(struct synapse));
          rnnlm->neu0_topic = (struct neuron *) New (rnnlm->x, sizeof(struct neuron)*rnnlm->layer0_topic_size);
       }

    if (rnnlm->layerc_size==0)
    {
	    rnnlm->syn1=(struct synapse *)New(rnnlm->x, rnnlm->layer2_size * rnnlm->layer1_size*sizeof(struct synapse));
        rnnlm->syn1b=(struct synapse *)New(rnnlm->x, rnnlm->layer2_size * rnnlm->layer1_size*sizeof(struct synapse));
    }
    else
    {
	    rnnlm->syn1=(struct synapse *)New(rnnlm->x, rnnlm->layer1_size*rnnlm->layerc_size*sizeof(struct synapse));
	    rnnlm->sync=(struct synapse *)New(rnnlm->x, rnnlm->layer2_size*rnnlm->layerc_size*sizeof(struct synapse));
        rnnlm->syn1b=(struct synapse *)New(rnnlm->x, rnnlm->layer1_size*rnnlm->layerc_size*sizeof(struct synapse));
	    rnnlm->syncb=(struct synapse *)New(rnnlm->x, rnnlm->layer2_size*rnnlm->layerc_size*sizeof(struct synapse));
    }

    if (rnnlm->syn1==NULL) {
	    printf("Memory allocation failed\n");
	    exit(1);
    }

    if (rnnlm->layerc_size>0)
        if (rnnlm->sync==NULL) {
	    printf("Memory allocation failed\n");
	    exit(1);
    }

    if ( rnnlm->direct_size ) {

        rnnlm->syn_d=(direct_t *)New(rnnlm->x, rnnlm->direct_size*sizeof(direct_t));

        if (rnnlm->syn_d==NULL) {
            printf("Memory allocation for direct connections failed (requested %lld bytes)\n", (long long)rnnlm->direct_size *sizeof(direct_t));
            exit(1);
        }

    }

    /*-----------------------------------------------------------------------------
     *  Initialisation
     *-----------------------------------------------------------------------------*/
    for (a=0; a<rnnlm->layer0_size; a++) {
        rnnlm->neu0[a].ac=0;
        rnnlm->neu0[a].er=0;
    }

    for (a=0; a<rnnlm->layer1_size; a++) {
        rnnlm->neu1[a].ac=0;
        rnnlm->neu1[a].er=0;
    }

    for (a=0; a<rnnlm->layerc_size; a++) {
        rnnlm->neuc[a].ac=0;
        rnnlm->neuc[a].er=0;
    }

    for (a=0; a<rnnlm->layer2_size; a++) {
        rnnlm->neu2[a].ac=0;
        rnnlm->neu2[a].er=0;
    }

    for (b=0; b<rnnlm->layer1_size; b++)
        for (a=0; a<rnnlm->layer0_size; a++) {
        rnnlm->syn0[a+b*rnnlm->layer0_size].weight=rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1);
    }

    if (rnnlm->layerc_size>0) {
        for (b=0; b<rnnlm->layerc_size; b++)
            for (a=0; a<rnnlm->layer1_size; a++) {
                rnnlm->syn1[a+b*rnnlm->layer1_size].weight=rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1);
            }

        for (b=0; b<rnnlm->layer2_size; b++)
            for (a=0; a<rnnlm->layerc_size; a++) {
                rnnlm->sync[a+b*rnnlm->layerc_size].weight=
                    rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1);
            }
    }
    else
    {
        for (b=0; b<rnnlm->layer2_size; b++)
            for (a=0; a<rnnlm->layer1_size; a++)
            {
                rnnlm->syn1[a+b*rnnlm->layer1_size].weight=rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1);
            }
    }

    if (rnnlm->layer0_topic_size)
       {
          for (b=0; b<rnnlm->layer2_size; b++)
             {
                for (a=0; a<rnnlm->layer0_topic_size; a++)
                   {
                      rnnlm->syn_d_topic[a+rnnlm->layer0_topic_size*b].weight=rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1);
                   }
             }
          for (b=0; b<rnnlm->layer1_size; b++)
             {
                for (a=0; a<rnnlm->layer0_topic_size; a++)
                   {
                      rnnlm->syn0_topic[a+rnnlm->layer0_topic_size*b].weight=rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1)+rnnrandom(-0.1, 0.1);

                   }
             }
       }

    for (aa=0; aa<rnnlm->direct_size; aa++) rnnlm->syn_d[aa]=0;

    if (rnnlm->bptt>0) {
        if (trace & T_TOP)
	        printf("WARNING: use RNNLM in HTK does not support training at this point");
    }

    saveWeights(rnnlm);



    /*-----------------------------------------------------------------------------
     *  handle classes
     *-----------------------------------------------------------------------------*/
    if ( rnnlm->setvocab  == TRUE ) {
        MakeVocabularyClass(rnnlm);
    }


}

/* static void copyHiddenLayerToInput(RNNLM* rnnlm) */
void copyHiddenLayerToInput(RNNLM* rnnlm)
{
    int a;

    if (rnnlm->usecuedrnnlm)
    {
        if (fabs(rnnlm->cuedversion - 1.1) < 1e-6)
        {
            for (a=0; a<rnnlm->num_layer; a++)
            {
                CopyHiddenAc (&(rnnlm->_layers[a]));
            }
        }
        else if (rnnlm->cuedversion == 1.0)
        {
            for (a=0; a<rnnlm->num_layer; a++)
            {
                CopyHiddenAc (&(rnnlm->_layers[a]));
            }
        }
        else
        {
            for (a=0; a<rnnlm->layersizes[1]; a++)
            {
                rnnlm->neu0_ac_hist[a] = rnnlm->neu_ac[1][a];
            }
        }
    }
    else
    {
        for (a=0; a<rnnlm->layer1_size; a++) {
            rnnlm->neu0[a+rnnlm->layer0_size-rnnlm->layer1_size].ac=rnnlm->neu1[a].ac;
    }
    }
}

static void netReset(RNNLM* rnnlm)   /* cleans hidden layer activation + bptt history */
{
    int a;

    if (rnnlm->usecuedrnnlm)
    {
        if (fabs(rnnlm->cuedversion - 1.1) < 1e-6)
        {
            for (a=0; a<rnnlm->num_layer; a++)
            {
                Reset(&(rnnlm->_layers[a]));
            }
        }
        else if (rnnlm->cuedversion == 1.0)
        {
            for (a=0; a<rnnlm->num_layer; a++)
            {
                Reset(&(rnnlm->_layers[a]));
            }
        }
        else
        {
            for (a=0; a<rnnlm->layersizes[1]; a++)
            {
                rnnlm->neu0_ac_hist[a] = 0.1;
                rnnlm->neu_ac[1][a] = 0.1;
            }
        }
    }
    else
    {
        for (a=0; a<rnnlm->layer1_size; a++) {
            if (rnnlm->usefrnnlm)
                rnnlm->neu1[a].ac=0.1;
            else
                rnnlm->neu1[a].ac=1.0;
        }
        copyHiddenLayerToInput(rnnlm);
    }

    if (rnnlm->bptt>0) {
        if (trace & T_TOP )
            printf("WARNING: use RNNLM in HTK does not support training at this point");
    }

    for (a=0; a<MAX_NGRAM_ORDER; a++) rnnlm->history[a]=0;
}

static int getWordHash(RNNLM* rnnlm, char *word)
{
    unsigned int hash, a;

    hash=0;
    for (a=0; a<strlen(word); a++) hash=hash*237+word[a];
    hash=hash%rnnlm->vocab_hash_size;

    return hash;
}


int searchRNNVocab(RNNLM* rnnlm, char *word)
{
    int a, b;
    unsigned int hash;

    hash=getWordHash(rnnlm, word);
    b=rnnlm->vocab_hash[hash];

    if (rnnlm->vocab_hash[hash]==-1) return -1;
    if (!strcmp(word, rnnlm->vocab[b].word))
        return b;

    for (a=0; a<rnnlm->vocab_size; a++) {				/* search in vocabulary */
        if (!strcmp(word, rnnlm->vocab[a].word)) {
    	    rnnlm->vocab_hash[hash]=a;
    	    return a;
    	}
    }

    return -1;							/* return OOV if not found */
}


static void matrixXvector_cuedrnnlm (real *src, real *wgt, real *dst, int nr, int nc)
{
    int i, j;
    for (i=0; i<nc; i++)
    {
        for (j=0; j<nr; j++)
        {
            dst[i] += src[j]*wgt[j+i*nr];
        }
    }
    return;
}

void matrixXvector_v2 (float *src, float *wgt, float *dst, int nr, int nc)
{
    int i, j;
    for (i=0; i<nc; i++)
    {
        for (j=0; j<nr; j++)
        {
            dst[i] += src[j]*wgt[j+i*nr];
        }
    }
    return;
}

/*
    neuron *dest, neuron *srcvec, synapse *srcmatrix, int matrix_width, int from, int to, int from2, int to2, int type
    compute matrix vector product:
    matrix: srcmatrix[from ... to ][ from2 ... to2 ] * srcvec[from2 ... to2]    type = 0
    */
static void matrixXvector(struct neuron *dest, struct neuron *srcvec, struct synapse *srcmatrix, int matrix_width, int from, int to, int from2, int to2, int type)
{
    int a, b;
    real val1, val2, val3, val4;
    real val5, val6, val7, val8;


    if (type==0) {		/* forward mode */
        for (b=0; b<(to-from)/8; b++) {
            val1=0;
            val2=0;
            val3=0;
            val4=0;

            val5=0;
            val6=0;
            val7=0;
            val8=0;

            for (a=from2; a<to2; a++) {
                val1 += srcvec[a].ac * srcmatrix[a+(b*8+from+0)*matrix_width].weight;
                val2 += srcvec[a].ac * srcmatrix[a+(b*8+from+1)*matrix_width].weight;
                val3 += srcvec[a].ac * srcmatrix[a+(b*8+from+2)*matrix_width].weight;
                val4 += srcvec[a].ac * srcmatrix[a+(b*8+from+3)*matrix_width].weight;

                val5 += srcvec[a].ac * srcmatrix[a+(b*8+from+4)*matrix_width].weight;
                val6 += srcvec[a].ac * srcmatrix[a+(b*8+from+5)*matrix_width].weight;
                val7 += srcvec[a].ac * srcmatrix[a+(b*8+from+6)*matrix_width].weight;
                val8 += srcvec[a].ac * srcmatrix[a+(b*8+from+7)*matrix_width].weight;
            }
            dest[b*8+from+0].ac += val1;
            dest[b*8+from+1].ac += val2;
            dest[b*8+from+2].ac += val3;
            dest[b*8+from+3].ac += val4;

            dest[b*8+from+4].ac += val5;
            dest[b*8+from+5].ac += val6;
            dest[b*8+from+6].ac += val7;
            dest[b*8+from+7].ac += val8;
        }
        for (b=b*8; b<to-from; b++) {
            for (a=from2; a<to2; a++) {
                dest[b+from].ac += srcvec[a].ac * srcmatrix[a+(b+from)*matrix_width].weight;
            }
        }
    }
    else
    {
        printf("ERROR: in matrixXvector, only forward mode is support");
        exit(1);
    }

}







/* --------------------------- Initialisation ---------------------- */
/* EXPORT-> isProbTableCached */
Boolean isProbTableCached()
{
    return calcProbCache ;
}


/* EXPORT->InitLM: initialise configuration parameters */
void InitRNLM(void)
{
   int i;
   Boolean b;
/*    char buf[MAXSTRLEN]; */
   double f = 0.0;

   Register(hrnlm_version,hrnlm_vc_id);
   nParm = GetConfig("HRNLM", TRUE, cParm, MAXGLOBS);

   /* setup the local memory management */
   CreateHeap(&rnnInfoStack, "rnnInfoStore", MSTAK, 1, 0.0, 100000, ULONG_MAX);

   if (nParm>0){
      if (GetConfInt(cParm,nParm,"TRACE",&i)) trace = i;
      if (GetConfInt(cParm,nParm,"OOSCLASS", &i)) oosNodeClass = i;
      if (GetConfBool(cParm, nParm, "RNNPROBCACHE", &b)) calcProbCache  = b;
      if (GetConfFlt(cParm,nParm,"OUTLAYERLOGNORM",&f)) outLayerLogNorm = f;
   }
}


/* --------------------------- Interface      ---------------------- */
/*  Export->  CreateRNNLM */
RNNLM* CreateRNNLM(MemHeap* x)
{


    RNNLM* model = New(&rnnInfoStack, sizeof(RNNLM) );
    model->x = x;

    model->outP_arr = NULL;

    model->version=10;
    model->cuedversion = 0.1;
    model->lognormconst = -1;
    model->nclass = 0;
    model->binformat = FALSE;
    model->CHECKNUM = 9999999;
    model->filetype=TEXT;

    model->setvocab = FALSE;
    model->usefrnnlm = FALSE;
    model->usecuedrnnlm = FALSE;
    model->usecuedsurnnlm = FALSE;
    model->in_oos_nodeid= 0;
    model->out_oos_nodeid= 0;
    model->lognorm=-1;
    if (outLayerLogNorm >= 0) {
       model->lognorm = outLayerLogNorm;
       fprintf(stdout, "Using constant output layer softmax log-normalization term: %f\n", model->lognorm);
       fflush(stdout);
    }


	model->use_lmprob=0;
	model->lambda=0.75;
	model->gradient_cutoff=15;
	model->dynamic=0;

	model->train_file[0]=0;
	model->valid_file[0]=0;
	model->test_file[0]=0;
	model->rnnlm_file[0]=0;

	model->alpha_set=0;
	model->train_file_set=0;

	model->alpha=0.1;
	model->beta=0.0000001;
	/*beta=0.00000; */
	model->alpha_divide=0;
	model->logp=0;
	model->llogp=-100000000;
	model->iter=0;

	model->min_improvement=1.003;

	model->train_words=0;
	model->train_cur_pos=0;
	model->vocab_max_size=100;
	model->vocab_size=0;
	model->vocab=(struct vocab_word *)New(model->x, model->vocab_max_size*sizeof(struct vocab_word));

	model->layer1_size=30;

	model->direct_size=0;
	model->direct_order=0;

	model->bptt=0;
	model->bptt_block=10;
	model->bptt_history=NULL;
	model->bptt_hidden=NULL;
	model->bptt_syn0=NULL;

	model->gen=0;

	model->independent=0;

	model->neu0=NULL;
	model->neu1=NULL;
	model->neuc=NULL;
	model->neu2=NULL;

	model->syn0=NULL;
	model->syn1=NULL;
	model->sync=NULL;
	model->syn_d=NULL;
	model->syn_db=NULL;
	/*backup */
	model->neu0b=NULL;
	model->neu1b=NULL;
	model->neucb=NULL;
	model->neu2b=NULL;

	model->neu1b2=NULL;

	model->syn0b=NULL;
	model->syn1b=NULL;
	model->syncb=NULL;
	/* */

    model->succeedwords=0;
    model->nsucclayer = 0;

	model->rand_seed=1;

	model->class_size=100;
	model->old_classes=0;

	model->one_iter=0;

	model->debug_mode=1;
	srand(model->rand_seed);

	model->vocab_hash_size=100000000;
	model->vocab_hash=
        (int *)New(model->x, model->vocab_hash_size*sizeof(int));

    return model;
}


void allocMem (RNNLM* rnnlmodel)
{
    int i, nrow, ncol;
    int nlayer = rnnlmodel->num_layer;
    if (nlayer < 2)
    {
        printf ("ERROR: the number of layers (%d) should be greater than 2\n", nlayer);
    }
    rnnlmodel->inputlayersize = rnnlmodel->layersizes[0] + rnnlmodel->layersizes[1];
    rnnlmodel->outputlayersize = rnnlmodel->layersizes[nlayer];
    rnnlmodel->layer0_hist = (real *)New(rnnlmodel->x, sizeof(real)*rnnlmodel->layersizes[1] * rnnlmodel->layersizes[1]);
    rnnlmodel->neu0_ac_hist = (real *)New(rnnlmodel->x, sizeof(real)*rnnlmodel->layersizes[1]);
    rnnlmodel->layers = (real **)New (rnnlmodel->x, sizeof(real *)*nlayer);
    rnnlmodel->neu_ac = (real **)New(rnnlmodel->x, sizeof(real *)*(nlayer+1));
    for (i=0; i<nlayer; i++)
    {
        nrow = rnnlmodel->layersizes[i];
        ncol = rnnlmodel->layersizes[i+1];
        rnnlmodel->layers[i] = (real *)New(rnnlmodel->x, sizeof(real)*nrow*ncol);
    }
    for (i=0; i<nlayer+1; i++)
    {
        nrow = rnnlmodel->layersizes[i];
        rnnlmodel->neu_ac[i] = (real *)New(rnnlmodel->x, sizeof(real)*nrow);
    }
    if (rnnlmodel->dim_fea > 0)
    {
        rnnlmodel->layer0_fea = (real *)New(rnnlmodel->x, sizeof(real)*rnnlmodel->dim_fea*rnnlmodel->layersizes[1]);
        rnnlmodel->neu0_ac_fea = (real *)New(rnnlmodel->x, sizeof(real)*rnnlmodel->dim_fea);
    }
    if (rnnlmodel->nclass > 0)
    {
        rnnlmodel->layerN_class = (real *)New(rnnlmodel->x, sizeof(real)*rnnlmodel->layersizes[nlayer-1]*rnnlmodel->nclass);
        rnnlmodel->neuN_ac_class = (real *)New(rnnlmodel->x, sizeof(real)*rnnlmodel->nclass);
    }

    if ( calcProbCache == TRUE ) {
        rnnlmodel->outP_arr = New(rnnlmodel->x, sizeof(float)*rnnlmodel->out_num_word );
    }
}

void LoadBinaryRNNLM (char *modelfn, RNNLM* rnnlmodel)
{
    return;
}

void LoadTextRNNLM_v1_1 (char *modelfn, RNNLM* rnnlmodel)
{
  
    int i, a, b, nlayer;
    float v;
    char word[1024];
    char type[1024];
    FILE *fptr = NULL;
    fptr = fopen (modelfn, "r");
    if (fptr == NULL)
    {
        printf ("ERROR: Failed to read RNNLM model file(%s)\n", modelfn);
        exit (0);
    }
    fscanf (fptr, "cuedrnnlm v%f\n", &v);
    if (fabs(v-1.1) > 1e-6)
    {
        printf ("Error: the version of cued rnnlm mode should be v1.1\n");
        exit (0);
    }
    fscanf (fptr, "train file: %s\n", word);
    fscanf (fptr, "valid file: %s\n", word);
    fscanf (fptr, "number of iteration: %d\n", &a);
    fscanf (fptr, "#train words: %d\n", &a);
    fscanf (fptr, "#valid words: %d\n", &a);
    fscanf (fptr, "#layer: %d\n", &nlayer);
    rnnlmodel->num_layer = nlayer;
    rnnlmodel->layersizes = (int *)New(rnnlmodel->x, sizeof(int)*(nlayer+1));
    rnnlmodel->_layers = (Layer *)New(rnnlmodel->x, nlayer*sizeof(Layer));
    for (i=0; i<nlayer+1; i++)
    {
        fscanf (fptr, "layer %d size: %d type: %s\n", &b, &a, word);
        assert(b==i);
        rnnlmodel->layersizes[i] = a;
        strcpy(rnnlmodel->layertypes[i], word);
    }
    /* BEGIN: modification */
    rnnlmodel->outputlayersize= rnnlmodel->layersizes[nlayer];
    /* END: modification */

    fscanf (fptr, "fullvoc size: %d\n", &rnnlmodel->fulldict_size);
    
    /* allocRNNMem (false);*/

    fscanf (fptr, "independent mode: %d\n", &rnnlmodel->independent);
    fscanf (fptr, "train crit mode: %d\n",  &rnnlmodel->traincritmode);
    float value;
    fscanf (fptr, "log norm: %f\n", &value);
    rnnlmodel->lognormconst= /*-1.0;*/ value;
    fscanf (fptr, "dim feature: %d\n", &rnnlmodel->dim_fea);
    rnnlmodel->_layers[0].dim_fea = rnnlmodel->dim_fea;
    if (rnnlmodel->dim_fea > 0)
    {
        rnnlmodel->_layers[0].ac_fea = rnnlmodel->neu0_ac_fea;
    }

    fscanf (fptr, "num of succeeding words: %d\n", &rnnlmodel->succeedwords);
    if (rnnlmodel->succeedwords > 0)
    {
        rnnlmodel->usecuedsurnnlm = TRUE;
        fscanf (fptr, "num of succlayer: %d\n", &rnnlmodel->nsucclayer);
        rnnlmodel->succlayersizes = (int *)New(rnnlmodel->x, sizeof(int)*(rnnlmodel->nsucclayer+1));
        for (i=0; i<=rnnlmodel->nsucclayer; i++)
        {
            fscanf (fptr, "succlayer %d size: %d\n", &a, &b);
            rnnlmodel->succlayersizes[a] = b;
        }
        fscanf (fptr, "merging layer: %d\n", &rnnlmodel->succmergelayer);
    }
    else
    {
        rnnlmodel->usecuedsurnnlm = FALSE;
    }
    for (i=0; i<rnnlmodel->num_layer; i++)
    {
        fscanf (fptr, "layer %d -> %d type: %s\n", &a, &b, word);
        strcpy(type, rnnlmodel->layertypes[i]);
        if (strcmp(type, word) != 0)
        {
            printf ("Error, layer type differ: %s vs %s\n", type, word);
            exit (0);
        }
        int nr = rnnlmodel->layersizes[i];
        int nc = rnnlmodel->layersizes[i+1];

        Init (rnnlmodel, &(rnnlmodel->_layers[i]), type, nr, nc);
        Read (&(rnnlmodel->_layers[i]), fptr);

    }

    if (rnnlmodel->succeedwords > 0)
    {
        rnnlmodel->layer1_succ = (Layer *)malloc(rnnlmodel->succeedwords*sizeof(Layer));
        rnnlmodel->succlayers  = (Layer *)malloc(rnnlmodel->nsucclayer*sizeof(Layer));
        rnnlmodel->neu1_ac_succ = (float **) malloc(rnnlmodel->succeedwords*sizeof(float *));
        rnnlmodel->neu_ac_succ = (float **) malloc((rnnlmodel->nsucclayer+1)*sizeof(float *));
        for (i=0; i<rnnlmodel->succeedwords; i++)
        {
            b = fscanf (fptr, "layer for succeeding word: %d type: %s\n", &a, type);
            int nr = rnnlmodel->layersizes[1];
            int nc = rnnlmodel->layersizes[2];
            Init (rnnlmodel, &(rnnlmodel->layer1_succ[i]), "feedforward", nr, nc);
            Read (&(rnnlmodel->layer1_succ[i]), fptr);
        }
        for (i=2; i<rnnlmodel->nsucclayer; i++)
        {
            int n = fscanf (fptr, "sulayer %d -> %d type: %s\n", &a, &b, type);
            int nr = rnnlmodel->succlayersizes[i];
            int nc = rnnlmodel->succlayersizes[i+1];
            if (i == rnnlmodel->nsucclayer-1)
            {
                Init (rnnlmodel, &(rnnlmodel->succlayers[i]), "linear", nr, nc);
            }
            else
            {
                Init (rnnlmodel, &(rnnlmodel->succlayers[i]), "feedforward", nr, nc);
            }
            Read (&(rnnlmodel->succlayers[i]), fptr);
        }
        for (i=0; i<rnnlmodel->succeedwords; i++)
        {
            rnnlmodel->neu1_ac_succ[i] = (float *)malloc(rnnlmodel->layersizes[1]*sizeof(float));
        }
        for (i=2; i<rnnlmodel->nsucclayer+1; i++)
        {
            rnnlmodel->neu_ac_succ[i] = (float *)malloc(rnnlmodel->succlayersizes[i]*sizeof(float));
        }
    }

#if 0
    /* smooth the su-RNNLM prob with a factor of 0.7 */
    if (rnnlmodel->usecuedsurnnlm)
    {
        rnnlmodel->_layers[rnnlmodel->num_layer-1].smfactor = 0.7;
    }
    else
    {
        rnnlmodel->_layers[rnnlmodel->num_layer-1].smfactor = 1.0;
    }
#endif

    fscanf (fptr, "%d", &a);
    if (a != rnnlmodel->CHECKNUM)
    {
        printf ("ERROR: failed to read the check number(%d) when reading model\n", rnnlmodel->CHECKNUM);
        exit (0);
    }
    printf ("Successfully loaded model: %s\n", modelfn);
    fclose (fptr);

    /* assign old variables */
    rnnlmodel->layer1_size = rnnlmodel->layersizes[rnnlmodel->num_layer-1];
    rnnlmodel->layer0_topic_size = rnnlmodel->dim_fea;

    /* allocate memory for neu_ac */
    rnnlmodel->neu_ac = (float **)New(rnnlmodel->x, (nlayer+1)*sizeof(float *));
    for (i=1; i<nlayer+1; i++)
    {
        rnnlmodel->neu_ac[i] = (float *)New(rnnlmodel->x, rnnlmodel->layersizes[i]*sizeof(float));
    }
    if ( calcProbCache == TRUE ) {
      rnnlmodel->outP_arr = New(rnnlmodel->x, sizeof(float)*rnnlmodel->out_num_word );
      /* BEGIN modification */
      if ( rnnlmodel->lognormconst > 0 )
	rnnlmodel->in_llayer_arr= New(rnnlmodel->x, sizeof(float)*rnnlmodel->layersizes[nlayer-1]);
      else
	rnnlmodel->in_llayer_arr= NULL;
      /* END modification */
    }
}

void LoadTextRNNLM_v1_0 (char *modelfn, RNNLM* rnnlmodel)
{
  
    int i, a, b, nlayer;
    float v;
    char word[1024];
    char type[1024];
    FILE *fptr = NULL;
    fptr = fopen (modelfn, "r");
    if (fptr == NULL)
    {
        printf ("ERROR: Failed to read RNNLM model file(%s)\n", modelfn);
        exit (0);
    }
    fscanf (fptr, "cuedrnnlm v%f\n", &v);
    if (v != 1.0)
    {
        printf ("Error: the version of cued rnnlm mode should be v1.0\n");
        exit (0);
    }
    fscanf (fptr, "train file: %s\n", word);
    fscanf (fptr, "valid file: %s\n", word);
    fscanf (fptr, "number of iteration: %d\n", &a);
    fscanf (fptr, "#train words: %d\n", &a);
    fscanf (fptr, "#valid words: %d\n", &a);
    fscanf (fptr, "#layer: %d\n", &nlayer);
    rnnlmodel->num_layer = nlayer;
    rnnlmodel->layersizes = (int *)New(rnnlmodel->x, sizeof(int)*(nlayer+1));
    rnnlmodel->_layers = (Layer *)New(rnnlmodel->x, nlayer*sizeof(Layer));
    for (i=0; i<nlayer+1; i++)
    {
        fscanf (fptr, "layer %d size: %d type: %s\n", &b, &a, word);
        assert(b==i);
        rnnlmodel->layersizes[i] = a;
        strcpy(rnnlmodel->layertypes[i], word);
    }

    fscanf (fptr, "fullvoc size: %d\n", &rnnlmodel->fulldict_size);
    /* allocRNNMem (false);*/

    fscanf (fptr, "independent mode: %d\n", &rnnlmodel->independent);
    fscanf (fptr, "train crit mode: %d\n",  &rnnlmodel->traincritmode);
    fscanf (fptr, "log norm: %f\n", &rnnlmodel->lognormconst);
    fscanf (fptr, "dim feature: %d\n", &rnnlmodel->dim_fea);
    rnnlmodel->_layers[0].dim_fea = rnnlmodel->dim_fea;
    if (rnnlmodel->dim_fea > 0)
    {
        rnnlmodel->_layers[0].ac_fea = rnnlmodel->neu0_ac_fea;
    }

    for (i=0; i<rnnlmodel->num_layer; i++)
    {
        fscanf (fptr, "layer %d -> %d type: %s\n", &a, &b, word);
        strcpy(type, rnnlmodel->layertypes[i]);
        if (strcmp(type, word) != 0)
        {
            printf ("Error, layer type differ: %s vs %s\n", type, word);
            exit (0);
        }
        int nr = rnnlmodel->layersizes[i];
        int nc = rnnlmodel->layersizes[i+1];

        Init (rnnlmodel, &(rnnlmodel->_layers[i]), type, nr, nc);
        Read (&(rnnlmodel->_layers[i]), fptr);
    }

    rnnlmodel->_layers[rnnlmodel->num_layer-1].smfactor = 1.0;

    fscanf (fptr, "%d", &a);
    if (a != rnnlmodel->CHECKNUM)
    {
        printf ("ERROR: failed to read the check number(%d) when reading model\n", rnnlmodel->CHECKNUM);
        exit (0);
    }
    printf ("Successfully loaded model: %s\n", modelfn);
    fclose (fptr);

    /* assign old variables */
    rnnlmodel->layer1_size = rnnlmodel->layersizes[rnnlmodel->num_layer-1];
    rnnlmodel->layer0_topic_size = rnnlmodel->dim_fea;

    /* allocate memory for neu_ac */
    rnnlmodel->neu_ac = (float **)New(rnnlmodel->x, (nlayer+1)*sizeof(float *));
    for (i=1; i<nlayer+1; i++)
    {
        rnnlmodel->neu_ac[i] = (float *)New(rnnlmodel->x, rnnlmodel->layersizes[i]*sizeof(float));
    }
    if ( calcProbCache == TRUE ) {
        rnnlmodel->outP_arr = New(rnnlmodel->x, sizeof(float)*rnnlmodel->out_num_word );
    }
}

void LoadTextRNNLM (char *modelfn, RNNLM* rnnlmodel)
{
  
    int i, a, b, n, nrow, nlayer;
    float v;
    char word[1024];
    FILE *fptr = NULL;
    fptr = fopen (modelfn, "r");
    if (fptr == NULL)
    {
        printf ("ERROR: Failed to read RNNLM model file(%s)\n", modelfn);
        exit (0);
    }
    fscanf (fptr, "cuedrnnlm v%f\n", &v);
    if (fabs(v-rnnlmodel->cuedversion) > 1e-3)
    {
        printf ("Error: the version of rnnlm model(v%.1f) is not consistent with binary supported(v%.1f)\n", v, rnnlmodel->cuedversion);
        exit (0);
    }
    fscanf (fptr, "train file: %s\n", rnnlmodel->train_file);
    fscanf (fptr, "valid file: %s\n", rnnlmodel->valid_file);
    fscanf (fptr, "number of iteration: %d\n", &rnnlmodel->iter);
    fscanf (fptr, "#train words: %d\n", &n);
    fscanf (fptr, "#valid words: %d\n", &n);
    fscanf (fptr, "#layer: %d\n", &nlayer);
    rnnlmodel->num_layer = nlayer;
    rnnlmodel->layersizes = (int *)New(rnnlmodel->x, sizeof(int)*(nlayer+1));
    for (i=0; i<nlayer+1; i++)
    {
        fscanf (fptr, "layer %d size: %d\n", &b, &a);
        assert(b==i);
        rnnlmodel->layersizes[i] = a;
    }
    fscanf (fptr, "feature dimension: %d\n", &rnnlmodel->dim_fea);
    fscanf (fptr, "class layer dimension: %d\n", &rnnlmodel->nclass);
    fscanf (fptr, "fullvoc size: %d\n", &rnnlmodel->fulldict_size);
    printf ("fullvoc size: %d\n", rnnlmodel->fulldict_size);
    allocMem (rnnlmodel);

    fscanf (fptr, "independent mode: %d\n", &rnnlmodel->independent);
    fscanf (fptr, "train crit mode: %d\n",  &rnnlmodel->traincritmode);
    fscanf (fptr, "log norm: %f\n", &v);
    rnnlmodel->lognormconst = v;
    fscanf (fptr, "hidden node type: %d\n", &rnnlmodel->nodetype);
    if (rnnlmodel->nodetype == 0)
    {
        printf ("sigmoid used for hidden node\n");
    }
    else if (rnnlmodel->nodetype == 1)
    {
        printf ("relu used for hidden node\n");
    }
    else
    {
        printf ("unrecognized hidde node type: %d\n", rnnlmodel->nodetype);
        exit (0);
    }
    if (rnnlmodel->nodetype == 1)
    {
        fscanf (fptr, "relu ratio: %f\n", &v);
        rnnlmodel->reluratio = v;
    }
    for (i=0; i<rnnlmodel->num_layer; i++)
    {
        fscanf (fptr, "layer %d -> %d\n", &a, &b);
        assert (a==i);
        assert (b==(i+1));
        for (a=0; a<rnnlmodel->layersizes[i]; a++)
        {
            nrow = rnnlmodel->layersizes[i];
            for (b=0; b<rnnlmodel->layersizes[i+1]; b++)
            {
                fscanf (fptr, "%f", &v);
                rnnlmodel->layers[i][a+b*nrow] = v;
            }
            fscanf (fptr, "\n");
        }
    }
    fscanf (fptr, "recurrent layer 1 -> 1\n");
    for (a=0; a<rnnlmodel->layersizes[1]; a++)
    {
        nrow = rnnlmodel->layersizes[1];
        for (b=0; b<rnnlmodel->layersizes[1]; b++)
        {
            fscanf (fptr, "%f", &v);
            rnnlmodel->layer0_hist[a+nrow*b] = v;
        }
        fscanf (fptr, "\n");
    }
    if (rnnlmodel->dim_fea > 0)
    {
        fscanf (fptr, "feature layer weight\n");
        for (a=0; a<rnnlmodel->dim_fea; a++)
        {
            nrow = rnnlmodel->dim_fea;
            for (b=0; b<rnnlmodel->layersizes[1]; b++)
            {
                fscanf (fptr, "%f", &v);
                rnnlmodel->layer0_fea[a+b*nrow] = v;
            }
            fscanf (fptr, "\n");
        }
    }
    if (rnnlmodel->nclass > 0)
    {
        fscanf (fptr, "class layer weight\n");
        for (a=0; a<rnnlmodel->layersizes[nlayer-1]; a++)
        {
            nrow = rnnlmodel->layersizes[nlayer-1];
            for (b=0; b<rnnlmodel->nclass; b++)
            {
                fscanf (fptr, "%f", &v);
                rnnlmodel->layerN_class[a+b*nrow] = v;
            }
            fscanf (fptr, "\n");
        }
    }
    fscanf (fptr, "hidden layer ac\n");
    for (a=0; a<rnnlmodel->layersizes[1]; a++)
    {
        fscanf (fptr, "%f", &v);
        rnnlmodel->neu0_ac_hist[a] = v;
    }
    fscanf (fptr, "\n");
    fscanf (fptr, "%d", &a);
    if (a != rnnlmodel->CHECKNUM)
    {
        printf ("ERROR: failed to read the check number(%d) when reading model\n", rnnlmodel->CHECKNUM);
        exit (0);
    }
    printf ("Successfully loaded cuedrnnlm model: %s\n", modelfn);
    if (rnnlmodel->lognormconst > 0)
    {
        printf ("RNNLM is to evluate with unnormalized prob, lognorm=%.5f\n", rnnlmodel->lognormconst);
    }
    fclose (fptr);

    /* assign old variables */
    rnnlmodel->layer1_size = rnnlmodel->layersizes[rnnlmodel->num_layer-1];
    rnnlmodel->layer0_topic_size = rnnlmodel->dim_fea;
}

/* EXPORT-> LoadCUEDRNLM v1.1 */
void LoadCUEDRNNLM_v1_1(char* modelfn, RNNLM* rnnlmodel){
    LoadTextRNNLM_v1_1 (modelfn, rnnlmodel);
}

/* EXPORT-> LoadCUEDRNLM v1.0 */
void LoadCUEDRNNLM_v1_0(char* modelfn, RNNLM* rnnlmodel){
    LoadTextRNNLM_v1_0 (modelfn, rnnlmodel);
}

/* EXPORT-> LoadCUEDRNNLM */
void LoadCUEDRNNLM(char* modelfn, RNNLM* rnnlmodel){

    if (rnnlmodel->binformat)
    {
        LoadBinaryRNNLM (modelfn, rnnlmodel);
    }
    else
    {
        LoadTextRNNLM (modelfn, rnnlmodel);
    }
}

/* EXPORT-> LoadFRNNLM */
void LoadFRNNLM(char* modelfn, RNNLM* rnnlmodel){

    FILE* fi = NULL ;
    int ver=0;
    char buf[MAXSTRLEN];
    double d;
    int a = 0, b=0;
    float fl;
    Boolean readVocab=TRUE;
    long long aa;

    fi=fopen(modelfn, "rb");
    if (fi==NULL)
    {
	    printf("ERROR: model file '%s' not found!\n", modelfn);
	    exit(1);
    }

    goToDelimiter(':', fi);
    fscanf(fi, "%d", &ver);
    if ((ver==4) && (rnnlmodel->version==5))
    {
        /* we will solve this later.. */ ;
    }
    else {
        if (ver!=rnnlmodel->version)
        {
            printf("Unknown version of file %s\n", modelfn );
            exit(1);
        }
    }
    /* file format: 1 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->filetype);
    /* training data file: train */
    goToDelimiter(':', fi);
    if (rnnlmodel->train_file_set==0)
	    fscanf(fi, "%s", rnnlmodel->train_file);
    else
        fscanf(fi, "%s", buf);
    /* validation data file: valid */
    goToDelimiter(':', fi);
    fscanf(fi, "%s", rnnlmodel->valid_file);
    /* last probability of validation data: -12628.747010 */
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &rnnlmodel->llogp);
    /* number of finished iterations: 9 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->iter);
    /* current position in training data: 0 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->train_cur_pos);
    /* current probability of training data: -12602.855425 */
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &rnnlmodel->logp);
    /* save after processing # words: 0 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->anti_k);
    /* # of training words: 81350 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->train_words);
    /* input layer size: 3735 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->layer0_size);
    /* hidden layer size: 15 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->layer1_size);
    /* compression layer size: 0 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->layerc_size);
    /* output layer size: 3820 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->layer2_size);

    if (rnnlmodel->version>5) {
	    goToDelimiter(':', fi);
        /* direct connections: 2000000 */
	    fscanf(fi, "%lld", &rnnlmodel->direct_size);
    }
    /*  */
    if (rnnlmodel->version>6) {
        goToDelimiter(':', fi);
        /* direct order: 3 */
        fscanf(fi, "%d", &rnnlmodel->direct_order);
    }
    /* bptt: 5 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->bptt);
    /* bptt block: 10 */
    if (rnnlmodel->version>4) {
        goToDelimiter(':', fi);
        fscanf(fi, "%d", &rnnlmodel->bptt_block);
    } else
        rnnlmodel->bptt_block=10;
    /* vocabulary size: 3720 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->vocab_size);
    /* independent sentences mode: 0 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->independent);
    /* starting learning rate: 0.100000 */
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &d);
    rnnlmodel->starting_alpha=d;
    /* current learning rate: 0.006250 */
    goToDelimiter(':', fi);
    if (rnnlmodel->alpha_set==0) {
	    fscanf(fi, "%lf", &d);
	    rnnlmodel->alpha=d;
    } else
        fscanf(fi, "%lf", &d);
    /* learning rate decrease: 1 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->alpha_divide);
    /* */

    /* flag_specifylist:true  */
    goToDelimiter(':', fi);
    readWord(buf, fi);
    if ( !strcmp(buf, "true")) {
        readVocab = FALSE ;
    }
    for (a=0;a<strlen(buf);a++) buf[a]=tolower(buf[a]);








#if 1
    /*  fulldict_size:58284  ( optional )*/
    goToDelimiter(':', fi);
    if ( fseek(fi, -10, SEEK_CUR) ) {
        HError(999, "ERROR: cannot rewind 10 bytes");
    }
    if ( fscanf(fi, "%9s", buf) != 1 ) {
        HError(999, "ERROR: cannot read 10 bytes");
    }
    if ( strcmp(buf, "dict_size") ){   /*  should be Inputlist_size */
        goToDelimiter(':', fi);
        fscanf(fi, "%d", &a);
        if ( (a-1) !=rnnlmodel->in_num_word )
        {
            HError(999,"ERROR: in LoadFRNLMwgt, number of input word (%d in %s) does not match (%d)",
                a, modelfn, rnnlmodel->in_num_word);
        }
        /*  outputlist size: 20001 */
        goToDelimiter(':', fi);
        fscanf(fi, "%d", &a);
        if ( (a-1) !=rnnlmodel->out_num_word )
        {
            HError(999,"ERROR: in LoadFRNLMwgt, number of output word (%d in %s) does not match (%d)",
                a, modelfn, rnnlmodel->out_num_word);
        }

    }
    else
    {
        goToDelimiter(':', fi);
        /*  should be fulldict_size: */
        fscanf(fi, "%d", &rnnlmodel->fulldict_size);
        printf ("fulldict_size: %d\n", rnnlmodel->fulldict_size);
        /*  inputlist_size:31857 */
        goToDelimiter(':', fi);
        fscanf(fi, "%d", &a);
        if ( (a-1) !=rnnlmodel->in_num_word )
        {
            HError(999,"ERROR: in LoadFRNLMwgt, number of input word (%d in %s) does not match (%d)",
                a, modelfn, rnnlmodel->in_num_word);
        }
        /*  outputlist size: 20001 */
        goToDelimiter(':', fi);
        fscanf(fi, "%d", &a);
        if ( (a-1) !=rnnlmodel->out_num_word )
        {
            HError(999,"ERROR: in LoadFRNLMwgt, number of output word (%d in %s) does not match (%d)",
                a, modelfn, rnnlmodel->out_num_word);
        }
    }

    rnnlmodel->layer0_topic_size = 0;
    fscanf(fi, "%s", buf);
    if (strcmp(buf, "Topic") == 0)
       {
          goToDelimiter (':', fi);
          fscanf (fi, "%d", &a);
          if (a>0) rnnlmodel->layer0_topic_size = a;
       }










#else
    /* inputlist_size: 10001 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &a);       /* the inputlist size in f-rnnlm model contains <OOS> */
    if ( (a-1) !=rnnlmodel->in_num_word )
    {
        HError(999,"ERROR: in LoadFRNLMwgt, number of input word (%d in %s) does not match (%d)",
                a, modelfn, rnnlmodel->in_num_word);
    }
    /*  outputlist size: 20001 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &a);       /* the outputlist size in f-rnnlm model contains <OOS> */
    if ( (a-1) !=rnnlmodel->out_num_word )
    {
        HError(999,"ERROR: in LoadFRNLMwgt, number of output word (%d in %s) does not match (%d)",
                a, modelfn, rnnlmodel->out_num_word);
    }
#endif

    if (rnnlmodel->fulldict_size == 0)
    {
        rnnlmodel->fulldict_size = rnnlmodel->in_num_word;    /*if fulldict_size is not specified from model file, then use the input list size instead*/
        printf ("fulldict_size is not specified in model file, use the size of input list instead.\n");
        printf ("fulldict_size: %d\n", rnnlmodel->fulldict_size);
    }

    if ( readVocab )
        /*  old format, specify vocabulary and class in the model file,
         *  This format is no longer used !!!
         *  TODO to disable this path */
    {

        /*-----------------------------------------------------------------------------
         *  initial hash table (in and out)
         *-----------------------------------------------------------------------------*/
        InitalHashTable(HASHTABLESIZE, sizeof(int), &rnnlmodel->in_vocab);
        InitalHashTable(HASHTABLESIZE, sizeof(int), &rnnlmodel->out_vocab);

        if (rnnlmodel->vocab_max_size<rnnlmodel->vocab_size) {
            if (rnnlmodel->vocab!=NULL)
                rnnlmodel->vocab_max_size=rnnlmodel->vocab_size+1000;
            rnnlmodel->vocab=(struct vocab_word *)New(rnnlmodel->x, sizeof(struct vocab_word)* rnnlmodel->vocab_max_size );
        }
        /* Vocabulary: */
        goToDelimiter(':', fi);
        for (a=0; a<rnnlmodel->vocab_size; a++) {
            fscanf(fi, "%d%d", &b, &rnnlmodel->vocab[a].cn);
            readWord(rnnlmodel->vocab[a].word, fi);
            fscanf(fi, "%d", &rnnlmodel->vocab[a].class_index);
            InsertToHashTable(&rnnlmodel->in_vocab, rnnlmodel->vocab[a].word, &b);
            InsertToHashTable(&rnnlmodel->out_vocab, rnnlmodel->vocab[a].word, &b);
        }
        rnnlmodel->setvocab = TRUE;
    }
    else    /*  read from external file later, consider to delete it later. TODO */
    {
        if (rnnlmodel->vocab_max_size<rnnlmodel->vocab_size) {
            if (rnnlmodel->vocab!=NULL)
                rnnlmodel->vocab_max_size=rnnlmodel->vocab_size+1000;
            rnnlmodel->vocab=(struct vocab_word *)New(rnnlmodel->x, sizeof(struct vocab_word)* rnnlmodel->vocab_max_size );
        }
        rnnlmodel->setvocab = FALSE;
    }


    if (rnnlmodel->neu0==NULL) initNet(rnnlmodel);		/*memory allocation here */

    /* read in hidden layer activation  */
    if (rnnlmodel->filetype==TEXT) {
        goToDelimiter(':', fi);
        for (a=0; a<rnnlmodel->layer1_size; a++) {
            fscanf(fi, "%lf", &d);
            rnnlmodel->neu1[a].ac=d;
        }
    }
    if (rnnlmodel->filetype==BINARY) {
        fgetc(fi);
        for (a=0; a<rnnlmodel->layer1_size; a++) {
            fread(&fl, 4, 1, fi);
            rnnlmodel->neu1[a].ac=fl;
        }
    }
    /* read in weights 0->1  */
    if (rnnlmodel->filetype==TEXT) {
        goToDelimiter(':', fi);
        for (b=0; b<rnnlmodel->layer1_size; b++) {
            for (a=0; a<rnnlmodel->layer0_size; a++) {
                fscanf(fi, "%lf", &d);
                rnnlmodel->syn0[a+b*rnnlmodel->layer0_size].weight=d;
            }
        }
    }
    if (rnnlmodel->filetype==BINARY) {
        for (b=0; b<rnnlmodel->layer1_size; b++) {
            for (a=0; a<rnnlmodel->layer0_size; a++) {
                fread(&fl, 4, 1, fi);
                rnnlmodel->syn0[a+b*rnnlmodel->layer0_size].weight=fl;
            }
        }
    }

    if (rnnlmodel->filetype==TEXT && rnnlmodel->layer0_topic_size > 0)
       {
          goToDelimiter(':', fi);
          for (b=0; b<rnnlmodel->layer1_size; b++)
             {
                for (a=0; a<rnnlmodel->layer0_topic_size; a++)
                   {
                      fscanf(fi, "%lf", &d);
                      rnnlmodel->syn0_topic[a+rnnlmodel->layer0_topic_size*b].weight=d;
                   }
             }
          goToDelimiter(':', fi);
          for (b=0; b<rnnlmodel->layer2_size; b++)
             {
                for (a=0; a<rnnlmodel->layer0_topic_size; a++)
                   {
                      fscanf(fi, "%lf", &d);
                      rnnlmodel->syn_d_topic[a+rnnlmodel->layer0_topic_size*b].weight=d;
                   }
             }
       }

    /* read in weights 1->2 */
    if (rnnlmodel->filetype==TEXT) {
        goToDelimiter(':', fi);
        if (rnnlmodel->layerc_size==0) {	/*no compress layer */
            for (b=0; b<rnnlmodel->layer2_size; b++) {
                for (a=0; a<rnnlmodel->layer1_size; a++) {
                    fscanf(fi, "%lf", &d);
                    rnnlmodel->syn1[a+b*rnnlmodel->layer1_size].weight=d;
                }
            }
        }
        else
        {				/*with compress layer */
            for (b=0; b<rnnlmodel->layerc_size; b++) {
                for (a=0; a<rnnlmodel->layer1_size; a++) {
                    fscanf(fi, "%lf", &d);
                    rnnlmodel->syn1[a+b*rnnlmodel->layer1_size].weight=d;
                }
            }

            goToDelimiter(':', fi);

            for (b=0; b<rnnlmodel->layer2_size; b++) {
                for (a=0; a<rnnlmodel->layerc_size; a++) {
                    fscanf(fi, "%lf", &d);
                    rnnlmodel->sync[a+b*rnnlmodel->layerc_size].weight=d;
                }
            }
        }
    }
    if (rnnlmodel->filetype==BINARY) {
        if (rnnlmodel->layerc_size==0) {	/*no compress layer */
            for (b=0; b<rnnlmodel->layer2_size; b++) {
                for (a=0; a<rnnlmodel->layer1_size; a++) {
                    fread(&fl, 4, 1, fi);
                    rnnlmodel->syn1[a+b*rnnlmodel->layer1_size].weight=fl;
                }
            }
        }
        else
        {				/*with compress layer */
            for (b=0; b<rnnlmodel->layerc_size; b++) {
                for (a=0; a<rnnlmodel->layer1_size; a++) {
                    fread(&fl, 4, 1, fi);
                    rnnlmodel->syn1[a+b*rnnlmodel->layer1_size].weight=fl;
                }
            }

            for (b=0; b<rnnlmodel->layer2_size; b++) {
                for (a=0; a<rnnlmodel->layerc_size; a++) {
                    fread(&fl, 4, 1, fi);
                    rnnlmodel->sync[a+b*rnnlmodel->layerc_size].weight=fl;
                }
            }
        }
    }
    /* read in direct connection */
    if (rnnlmodel->filetype==TEXT) {
        goToDelimiter(':', fi);		/*direct conenctions */
        for (aa=0; aa<rnnlmodel->direct_size; aa++) {
            fscanf(fi, "%lf", &d);
            rnnlmodel->syn_d[aa]=d;
        }
    }
    if (rnnlmodel->filetype==BINARY) {
        long long aa;
        for (aa=0; aa<rnnlmodel->direct_size; aa++) {
            fread(&fl, 4, 1, fi);
            rnnlmodel->syn_d[aa]=fl;
        }
    }

    printf ("Reading F-RNNLM model file completed!\n");
    saveWeights(rnnlmodel);
    fclose(fi);
}


/*  EXPORT-> LoadRNNLM  */
void LoadRNNLM(char* modelfn, RNNLM* rnnlmodel){

    FILE* fi = NULL ;
    int ver=0;
    char buf[MAXSTRLEN];
    double d;
    int a = 0, b=0;
    float fl;
    Boolean readVocab=TRUE;
    long long aa;

    fi=fopen(modelfn, "rb");
    if (fi==NULL)
    {
	    printf("ERROR: model file '%s' not found!\n", modelfn);
	    exit(1);
    }

    goToDelimiter(':', fi);
    fscanf(fi, "%d", &ver);
    if ((ver==4) && (rnnlmodel->version==5))
    {
        /* we will solve this later.. */ ;
    }
    else {
        if (ver!=rnnlmodel->version)
        {
            printf("Unknown version of file %s\n", modelfn );
            exit(1);
        }
    }
    /* file format: 1 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->filetype);
    /* training data file: train */
    goToDelimiter(':', fi);
    if (rnnlmodel->train_file_set==0)
	    fscanf(fi, "%s", rnnlmodel->train_file);
    else
        fscanf(fi, "%s", buf);
    /* validation data file: valid */
    goToDelimiter(':', fi);
    fscanf(fi, "%s", rnnlmodel->valid_file);
    /* last probability of validation data: -12628.747010 */
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &rnnlmodel->llogp);
    /* number of finished iterations: 9 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->iter);
    /* current position in training data: 0 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->train_cur_pos);
    /* current probability of training data: -12602.855425 */
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &rnnlmodel->logp);
    /* save after processing # words: 0 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->anti_k);
    /* # of training words: 81350 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->train_words);
    /* input layer size: 3735 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->layer0_size);
    /* hidden layer size: 15 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->layer1_size);
    /* compression layer size: 0 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->layerc_size);
    /* output layer size: 3820 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->layer2_size);

    if (rnnlmodel->version>5) {
	    goToDelimiter(':', fi);
        /* direct connections: 2000000 */
	    fscanf(fi, "%lld", &rnnlmodel->direct_size);
    }
    /*  */
    if (rnnlmodel->version>6) {
        goToDelimiter(':', fi);
        /* direct order: 3 */
        fscanf(fi, "%d", &rnnlmodel->direct_order);
    }
    /* bptt: 5 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->bptt);
    /* bptt block: 10 */
    if (rnnlmodel->version>4) {
        goToDelimiter(':', fi);
        fscanf(fi, "%d", &rnnlmodel->bptt_block);
    } else
        rnnlmodel->bptt_block=10;
    /* vocabulary size: 3720 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->vocab_size);
    /* class size: 100 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->class_size);
    /* old classes: 0 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->old_classes);
    /* independent sentences mode: 0 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->independent);
    /* starting learning rate: 0.100000 */
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &d);
    rnnlmodel->starting_alpha=d;
    /* current learning rate: 0.006250 */
    goToDelimiter(':', fi);
    if (rnnlmodel->alpha_set==0) {
	    fscanf(fi, "%lf", &d);
	    rnnlmodel->alpha=d;
    } else
        fscanf(fi, "%lf", &d);
    /* learning rate decrease: 1 */
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &rnnlmodel->alpha_divide);
    /* */

    /* flag_specifylist:true  */
    goToDelimiter(':', fi);
    readWord(buf, fi);
    if ( !strcmp(buf, "true")) {
        readVocab = FALSE ;
    }
    for (a=0;a<strlen(buf);a++) buf[a]=tolower(buf[a]);
    /*  fulldict_size:58284  ( optional )*/
    goToDelimiter(':', fi);
    if ( fseek(fi, -10, SEEK_CUR) ) {
        HError(999, "ERROR: cannot rewind 10 bytes");
    }
    if ( fscanf(fi, "%9s", buf) != 1 ) {
        HError(999, "ERROR: cannot read 10 bytes");
    }
    if ( strcmp(buf, "dict_size") ){   /*  should be Inputlist_size */
        goToDelimiter(':', fi);
        fscanf(fi, "%d", &a);
        if ( a!=rnnlmodel->in_num_word )
        {
            HError(999,"ERROR: in LoadRNLMwgt, number of input word (%d in %s) does not match (%d)",
                a, modelfn, rnnlmodel->in_num_word);
        }
        /*  outputlist size: 20001 */
        goToDelimiter(':', fi);
        fscanf(fi, "%d", &a);
        if ( a!=rnnlmodel->out_num_word )
        {
            HError(999,"ERROR: in LoadRNLMwgt, number of output word (%d in %s) does not match (%d)",
                a, modelfn, rnnlmodel->out_num_word);
        }

    }
    else
    {
        goToDelimiter(':', fi);
        /*  should be fulldict_size: */
        fscanf(fi, "%d", &rnnlmodel->fulldict_size);
        /*  inputlist_size:31857 */
        goToDelimiter(':', fi);
        fscanf(fi, "%d", &a);
        if ( a!=rnnlmodel->in_num_word )
        {
            HError(999,"ERROR: in LoadRNLMwgt, number of input word (%d in %s) does not match (%d)",
                a, modelfn, rnnlmodel->in_num_word);
        }
        /*  outputlist size: 20001 */
        goToDelimiter(':', fi);
        fscanf(fi, "%d", &a);
        if ( a!=rnnlmodel->out_num_word )
        {
            HError(999,"ERROR: in LoadRNLMwgt, number of output word (%d in %s) does not match (%d)",
                a, modelfn, rnnlmodel->out_num_word);
        }
    }


    if (rnnlmodel->fulldict_size == 0)
    {
        rnnlmodel->fulldict_size = rnnlmodel->in_num_word;    /*if fulldict_size is not specified from model file, then use the input list size instead*/
        printf ("fulldict_size is not specified in model file, use the size of input list instead.\n");
        printf ("fulldict_size: %d\n", rnnlmodel->fulldict_size);
    }


    if ( readVocab )
        /*  old format, specify vocabulary and class in the model file,
         *  This format is no longer used !!!
         *  TODO to disable this path */
    {

        /*-----------------------------------------------------------------------------
         *  initial hash table (in and out)
         *-----------------------------------------------------------------------------*/
        InitalHashTable(HASHTABLESIZE, sizeof(int), &rnnlmodel->in_vocab);
        InitalHashTable(HASHTABLESIZE, sizeof(int), &rnnlmodel->out_vocab);

        if (rnnlmodel->vocab_max_size<rnnlmodel->vocab_size) {
            if (rnnlmodel->vocab!=NULL)
                rnnlmodel->vocab_max_size=rnnlmodel->vocab_size+1000;
            rnnlmodel->vocab=(struct vocab_word *)New(rnnlmodel->x, sizeof(struct vocab_word)* rnnlmodel->vocab_max_size );
        }
        /* Vocabulary: */
        goToDelimiter(':', fi);
        for (a=0; a<rnnlmodel->vocab_size; a++) {
            fscanf(fi, "%d%d", &b, &rnnlmodel->vocab[a].cn);
            readWord(rnnlmodel->vocab[a].word, fi);
            fscanf(fi, "%d", &rnnlmodel->vocab[a].class_index);
            InsertToHashTable(&rnnlmodel->in_vocab, rnnlmodel->vocab[a].word, &b);
            InsertToHashTable(&rnnlmodel->out_vocab, rnnlmodel->vocab[a].word, &b);
        }
        rnnlmodel->setvocab = TRUE;
    }
    else    /*  read from external file later */
    {
        if (rnnlmodel->vocab_max_size<rnnlmodel->vocab_size) {
            if (rnnlmodel->vocab!=NULL)
                rnnlmodel->vocab_max_size=rnnlmodel->vocab_size+1000;
            rnnlmodel->vocab=(struct vocab_word *)New(rnnlmodel->x, sizeof(struct vocab_word)* rnnlmodel->vocab_max_size );
        }
        rnnlmodel->setvocab = FALSE;
    }


    if (rnnlmodel->neu0==NULL) initNet(rnnlmodel);		/*memory allocation here */

    /* read in hidden layer activation  */
    if (rnnlmodel->filetype==TEXT) {
        goToDelimiter(':', fi);
        for (a=0; a<rnnlmodel->layer1_size; a++) {
            fscanf(fi, "%lf", &d);
            rnnlmodel->neu1[a].ac=d;
        }
    }
    if (rnnlmodel->filetype==BINARY) {
        fgetc(fi);
        for (a=0; a<rnnlmodel->layer1_size; a++) {
            fread(&fl, 4, 1, fi);
            rnnlmodel->neu1[a].ac=fl;
        }
    }
    /* read in weights 0->1  */
    if (rnnlmodel->filetype==TEXT) {
        goToDelimiter(':', fi);
        for (b=0; b<rnnlmodel->layer1_size; b++) {
            for (a=0; a<rnnlmodel->layer0_size; a++) {
                fscanf(fi, "%lf", &d);
                rnnlmodel->syn0[a+b*rnnlmodel->layer0_size].weight=d;
            }
        }
    }
    if (rnnlmodel->filetype==BINARY) {
        for (b=0; b<rnnlmodel->layer1_size; b++) {
            for (a=0; a<rnnlmodel->layer0_size; a++) {
                fread(&fl, 4, 1, fi);
                rnnlmodel->syn0[a+b*rnnlmodel->layer0_size].weight=fl;
            }
        }
    }
    /* read in weights 1->2 */
    if (rnnlmodel->filetype==TEXT) {
        goToDelimiter(':', fi);
        if (rnnlmodel->layerc_size==0) {	/*no compress layer */
            for (b=0; b<rnnlmodel->layer2_size; b++) {
                for (a=0; a<rnnlmodel->layer1_size; a++) {
                    fscanf(fi, "%lf", &d);
                    rnnlmodel->syn1[a+b*rnnlmodel->layer1_size].weight=d;
                }
            }
        }
        else
        {				/*with compress layer */
            for (b=0; b<rnnlmodel->layerc_size; b++) {
                for (a=0; a<rnnlmodel->layer1_size; a++) {
                    fscanf(fi, "%lf", &d);
                    rnnlmodel->syn1[a+b*rnnlmodel->layer1_size].weight=d;
                }
            }

            goToDelimiter(':', fi);

            for (b=0; b<rnnlmodel->layer2_size; b++) {
                for (a=0; a<rnnlmodel->layerc_size; a++) {
                    fscanf(fi, "%lf", &d);
                    rnnlmodel->sync[a+b*rnnlmodel->layerc_size].weight=d;
                }
            }
        }
    }
    if (rnnlmodel->filetype==BINARY) {
        if (rnnlmodel->layerc_size==0) {	/*no compress layer */
            for (b=0; b<rnnlmodel->layer2_size; b++) {
                for (a=0; a<rnnlmodel->layer1_size; a++) {
                    fread(&fl, 4, 1, fi);
                    rnnlmodel->syn1[a+b*rnnlmodel->layer1_size].weight=fl;
                }
            }
        }
        else
        {				/*with compress layer */
            for (b=0; b<rnnlmodel->layerc_size; b++) {
                for (a=0; a<rnnlmodel->layer1_size; a++) {
                    fread(&fl, 4, 1, fi);
                    rnnlmodel->syn1[a+b*rnnlmodel->layer1_size].weight=fl;
                }
            }

            for (b=0; b<rnnlmodel->layer2_size; b++) {
                for (a=0; a<rnnlmodel->layerc_size; a++) {
                    fread(&fl, 4, 1, fi);
                    rnnlmodel->sync[a+b*rnnlmodel->layerc_size].weight=fl;
                }
            }
        }
    }
    /* read in direct connection */
    if (rnnlmodel->filetype==TEXT) {
        goToDelimiter(':', fi);		/*direct conenctions */
        for (aa=0; aa<rnnlmodel->direct_size; aa++) {
            fscanf(fi, "%lf", &d);
            rnnlmodel->syn_d[aa]=d;
        }
    }
    if (rnnlmodel->filetype==BINARY) {
        long long aa;
        for (aa=0; aa<rnnlmodel->direct_size; aa++) {
            fread(&fl, 4, 1, fi);
            rnnlmodel->syn_d[aa]=fl;
        }
    }
    printf ("Reading C-RNNLM model file completed!\n");
    saveWeights(rnnlmodel);
    fclose(fi);
}

static char *ReadLMWord(Source* source, char *buf)
{
   int i, c;

   if (rawMITFormat) {
      while (isspace(c=GetCh(source)));
      i=0;
      while (!isspace(c) && c!=EOF && i<MAXSTRLEN){
         buf[i++] = c; c=GetCh(source);
      }
      buf[i] = '\0';
      UnGetCh(c,source);
      if (i>0)
         return buf;
      else
         return NULL;
   }
   else {
      if (ReadString(source,buf))
         return buf;
      else
         return NULL;
   }
}




static void GenInMap(char* fname, HashTable* t, int num_word)
{
    Source src;
    int k=0;
    int load_word=0;
    char buf[4096];

    if (InitSource(fname, &src, NoFilter) < SUCCESS ){
        HError(8110, "LoadRNLMwgt: Cannot open file %s", fname);
    }

    InitalHashTable(HASHTABLESIZE, sizeof(int), t);
    SkipWhiteSpace(&src);
    while ( load_word < num_word)
    {
        if (!ReadString(&src, buf)){
            HError(8110, "LoadRNLMwgt: Cannot read string");
        }
        if ( sscanf(buf, "%d", &k) != 1 ) {
            HError(8110, "LoadRNLMwgt: expect an integer");
        }
        if (!ReadLMWord(&src, buf)){
            HError(8110, "LoadRNLMwgt: Cannot read string");
        }
        if (FindInHashTable(t, buf)){
            HError(8110, "LoadRNLMwgt: duplicate word %s!", buf);
        }
        InsertToHashTable(t, buf, &k);
        load_word ++;
    }
    CloseSource(&src);

    if ( inputContainOOS ) {
        InsertToHashTable(t, oosNodeWord, &num_word);
    }
}
static void GenOutMap(char* fname, HashTable* t, int num_word, struct vocab_word *vocab)
{
    Source src;
    int k,j,i=0;
    int load_word=0;
    char buf[4096];
    char word[4096];
    int max_classid= 0;

    if (InitSource(fname, &src, NoFilter) < SUCCESS ){
        HError(8110, "LoadRNLMwgt: Cannot open file %s", fname);
    }

    InitalHashTable(HASHTABLESIZE, sizeof(int), t);
    SkipWhiteSpace(&src);
    while ( load_word < num_word)
    {
        /*  voc Id */
        if (!ReadString(&src, buf)){
            HError(8110, "LoadRNLMwgt: Cannot read string");
        }
        if ( sscanf(buf, "%d", &k) != 1 ) {
            HError(8110, "LoadRNLMwgt: expect an integer");
        }
        /*  word count */
        if (!ReadString(&src, buf)) {
            HError(8110, "LoadRNLMwgt: Cannot read string");
        }
        if ( sscanf(buf, "%d", &i) != 1 ) {
            HError(8110, "LoadRNLMwgt: expect an integer");
        }
        /*  LM word */
        if (!ReadLMWord(&src, buf)){
            HError(8110, "LoadRNLMwgt: Cannot read string");
        }
        strcpy(word, buf);
        if (FindInHashTable(t, buf)){
            HError(8110, "LoadRNLMwgt: duplicate word %s!", buf);
        }
        /*  class id */
        if (!ReadString(&src, buf)) {
            HError(8110, "LoadRNLMwgt: Cannot read string");
        }
        if ( sscanf(buf, "%d", &j) != 1 ) {
            HError(8110, "LoadRNLMwgt: expect an integer");
        }
        /*  k is the vocab id; i is the count ; j is the class id */
        InsertToHashTable(t, word, &k);
        strcpy(vocab[k].word, word);
        vocab[k].class_index=j ;
        if ( j> max_classid ) max_classid = j ;
        vocab[k].cn = i;
        load_word ++;
    }
    CloseSource(&src);

    if ( outputContainOOS ) {
        InsertToHashTable(t, oosNodeWord, &num_word ) ;
        k=num_word;
        strcpy(vocab[k].word, oosNodeWord);
        vocab[k].cn= 0 ;        /*  make sure cn is not used  */
        if ( oosNodeClass < 0 )
            vocab[k].class_index = max_classid+1;
        else
            vocab[k].class_index = oosNodeClass;
    }

}

void LoadCUEDRNLMwgt_v1_1 (char *orgmodelfn, char *inmap_file, char *outmap_file,
                      int in_num_word, int out_num_word, RNNLM* rnnlm)
{
    int *p = NULL;

    rnnlm->in_num_word = in_num_word;
    rnnlm->out_num_word = out_num_word;
    /*-----------------------------------------------------------------------------
     *  1. load in original formatted RNN model
     *-----------------------------------------------------------------------------*/
     LoadCUEDRNNLM_v1_1(orgmodelfn, rnnlm);
     rnnlm->usecuedrnnlm = TRUE ;
     rnnlm->cuedversion = 1.1;

     if (rnnlm->fulldict_size = 0)
     {
         rnnlm->fulldict_size = rnnlm->layersizes[0];
     }

    if ( !rnnlm->setvocab ) {
        /*-----------------------------------------------------------------------------
         *  2. load in input vocabulary map and make the in_hash map
         *-----------------------------------------------------------------------------*/
        GenInMap(inmap_file, &rnnlm->in_vocab, in_num_word);
        if ( !(p=(int*)FindInHashTable(&rnnlm->in_vocab, oosNodeWord)) ) {
            HError(999, "ERROR: in LoadCUEDRNLMwgt, did not find oos word(%s) in input vocabulary", oosNodeWord );
        }
        rnnlm->in_oos_nodeid = *p;
        if ( inputContainOOS ) {
            rnnlm->in_num_word ++;
        }


        /*-----------------------------------------------------------------------------
         *  3. load in output vocabulary map and make the out_hash map
         *-----------------------------------------------------------------------------*/
        /* for f-rnnlm, the output and input map list are with the same format. */
        GenInMap (outmap_file, &rnnlm->out_vocab, out_num_word);
        /* GenOutMap(outmap_file, &rnnlm->out_vocab, out_num_word, rnnlm->vocab); */
         if ( !(p=(int*)FindInHashTable(&rnnlm->out_vocab, oosNodeWord)) ) {
            HError(999, "ERROR: in LoadCUEDRNLMwgt, did not find oos word(%s) in output vocabulary", oosNodeWord );
        }
        rnnlm->out_oos_nodeid = *p;
        /* MakeVocabularyClass(rnnlm); */
        if ( outputContainOOS ) {
            rnnlm->out_num_word ++ ;
        }

        rnnlm->setvocab = TRUE ;
    }
}

void LoadCUEDRNLMwgt_v1_0 (char *orgmodelfn, char *inmap_file, char *outmap_file,
                      int in_num_word, int out_num_word, RNNLM* rnnlm)
{
    int *p = NULL;

    rnnlm->in_num_word = in_num_word;
    rnnlm->out_num_word = out_num_word;
    /*-----------------------------------------------------------------------------
     *  1. load in original formatted RNN model
     *-----------------------------------------------------------------------------*/
     LoadCUEDRNNLM_v1_0(orgmodelfn, rnnlm);
     rnnlm->usecuedrnnlm = TRUE ;
     rnnlm->cuedversion = 1.0;

     if (rnnlm->fulldict_size = 0)
     {
         rnnlm->fulldict_size = rnnlm->layersizes[0];
     }

    if ( !rnnlm->setvocab ) {
        /*-----------------------------------------------------------------------------
         *  2. load in input vocabulary map and make the in_hash map
         *-----------------------------------------------------------------------------*/
        GenInMap(inmap_file, &rnnlm->in_vocab, in_num_word);
        if ( !(p=(int*)FindInHashTable(&rnnlm->in_vocab, oosNodeWord)) ) {
            HError(999, "ERROR: in LoadCUEDRNLMwgt, did not find oos word(%s) in input vocabulary", oosNodeWord );
        }
        rnnlm->in_oos_nodeid = *p;
        if ( inputContainOOS ) {
            rnnlm->in_num_word ++;
        }


        /*-----------------------------------------------------------------------------
         *  3. load in output vocabulary map and make the out_hash map
         *-----------------------------------------------------------------------------*/
        /* for f-rnnlm, the output and input map list are with the same format. */
        GenInMap (outmap_file, &rnnlm->out_vocab, out_num_word);
        /* GenOutMap(outmap_file, &rnnlm->out_vocab, out_num_word, rnnlm->vocab); */
         if ( !(p=(int*)FindInHashTable(&rnnlm->out_vocab, oosNodeWord)) ) {
            HError(999, "ERROR: in LoadCUEDRNLMwgt, did not find oos word(%s) in output vocabulary", oosNodeWord );
        }
        rnnlm->out_oos_nodeid = *p;
        /* MakeVocabularyClass(rnnlm); */
        if ( outputContainOOS ) {
            rnnlm->out_num_word ++ ;
        }

        rnnlm->setvocab = TRUE ;
    }
}

void LoadCUEDRNLMwgt (char *orgmodelfn, char *inmap_file, char *outmap_file,
                      int in_num_word, int out_num_word, RNNLM* rnnlm)
{
    int *p = NULL;

    rnnlm->in_num_word = in_num_word;
    rnnlm->out_num_word = out_num_word;
    /*-----------------------------------------------------------------------------
     *  1. load in original formatted RNN model
     *-----------------------------------------------------------------------------*/
     LoadCUEDRNNLM(orgmodelfn, rnnlm);
     rnnlm->usecuedrnnlm = TRUE ;
     rnnlm->cuedversion = 0.1;

     if (rnnlm->fulldict_size = 0)
     {
         rnnlm->fulldict_size = rnnlm->layersizes[0];
     }

    if ( !rnnlm->setvocab ) {
        /*-----------------------------------------------------------------------------
         *  2. load in input vocabulary map and make the in_hash map
         *-----------------------------------------------------------------------------*/
        GenInMap(inmap_file, &rnnlm->in_vocab, in_num_word);
        if ( !(p=(int*)FindInHashTable(&rnnlm->in_vocab, oosNodeWord)) ) {
            HError(999, "ERROR: in LoadCUEDRNLMwgt, did not find oos word(%s) in input vocabulary", oosNodeWord );
        }
        rnnlm->in_oos_nodeid = *p;
        if ( inputContainOOS ) {
            rnnlm->in_num_word ++;
        }


        /*-----------------------------------------------------------------------------
         *  3. load in output vocabulary map and make the out_hash map
         *-----------------------------------------------------------------------------*/
        /* for f-rnnlm, the output and input map list are with the same format. */
        GenInMap (outmap_file, &rnnlm->out_vocab, out_num_word);
        /* GenOutMap(outmap_file, &rnnlm->out_vocab, out_num_word, rnnlm->vocab); */
         if ( !(p=(int*)FindInHashTable(&rnnlm->out_vocab, oosNodeWord)) ) {
            HError(999, "ERROR: in LoadCUEDRNLMwgt, did not find oos word(%s) in output vocabulary", oosNodeWord );
        }
        rnnlm->out_oos_nodeid = *p;
        /* MakeVocabularyClass(rnnlm); */
        if ( outputContainOOS ) {
            rnnlm->out_num_word ++ ;
        }

        rnnlm->setvocab = TRUE ;
    }
}

/* EXPORT-> LoadRNLMwgt for full connected RNLM */
void LoadFRNLMwgt (char *orgmodelfn, char *inmap_file, char *outmap_file,
                   int in_num_word, int out_num_word, RNNLM* rnnlm)
{
    int *p = NULL;

    rnnlm->in_num_word = in_num_word;
    rnnlm->out_num_word = out_num_word;
    /*-----------------------------------------------------------------------------
     *  1. load in original formatted RNN model
     *-----------------------------------------------------------------------------*/
     LoadFRNNLM(orgmodelfn, rnnlm);
     rnnlm->usefrnnlm = TRUE ;

    if ( !rnnlm->setvocab ) {
        /*-----------------------------------------------------------------------------
         *  2. load in input vocabulary map and make the in_hash map
         *-----------------------------------------------------------------------------*/
        GenInMap(inmap_file, &rnnlm->in_vocab, in_num_word);
        if ( !(p=(int*)FindInHashTable(&rnnlm->in_vocab, oosNodeWord)) ) {
            HError(999, "ERROR: in LoadFRNLMwgt, did not find oos word(%s) in input vocabulary", oosNodeWord );
        }
        rnnlm->in_oos_nodeid = *p;
        if ( inputContainOOS ) {
            rnnlm->in_num_word ++;
        }


        /*-----------------------------------------------------------------------------
         *  3. load in output vocabulary map and make the out_hash map
         *-----------------------------------------------------------------------------*/
        /* for f-rnnlm, the output and input map list are with the same format. */
        GenInMap (outmap_file, &rnnlm->out_vocab, out_num_word);
        /* GenOutMap(outmap_file, &rnnlm->out_vocab, out_num_word, rnnlm->vocab); */
         if ( !(p=(int*)FindInHashTable(&rnnlm->out_vocab, oosNodeWord)) ) {
            HError(999, "ERROR: in LoadFRNLMwgt, did not find oos word(%s) in output vocabulary", oosNodeWord );
        }
        rnnlm->out_oos_nodeid = *p;
        /* MakeVocabularyClass(rnnlm); */
        if ( outputContainOOS ) {
            rnnlm->out_num_word ++ ;
        }

        rnnlm->setvocab = TRUE ;
    }
}

/*  EXPORT-> LoadRNLMwgt */
void LoadRNLMwgt(char* orgmodelfn, char* inmap_file, char* outmap_file,
                int in_num_word, int out_num_word, RNNLM* rnnlm)
{
    int *p = NULL;

    rnnlm->in_num_word = in_num_word;
    rnnlm->out_num_word = out_num_word;
    /*-----------------------------------------------------------------------------
     *  1. load in original formatted RNN model
     *-----------------------------------------------------------------------------*/
    LoadRNNLM(orgmodelfn, rnnlm );

    if ( !rnnlm->setvocab ) {
        /*-----------------------------------------------------------------------------
         *  2. load in input vocabulary map and make the in_hash map
         *-----------------------------------------------------------------------------*/
        GenInMap(inmap_file, &rnnlm->in_vocab, in_num_word);
        if ( !(p=(int*)FindInHashTable(&rnnlm->in_vocab, oosNodeWord)) ) {
            HError(999, "ERROR: in LoadRNLMwgt, did not find oos word(%s) in input vocabulary", oosNodeWord );
        }
        rnnlm->in_oos_nodeid = *p;
        if ( inputContainOOS ) {
            rnnlm->in_num_word ++;
        }


        /*-----------------------------------------------------------------------------
         *  3. load in output vocabulary map and make the out_hash map
         *-----------------------------------------------------------------------------*/
        GenOutMap(outmap_file, &rnnlm->out_vocab, out_num_word, rnnlm->vocab);
         if ( !(p=(int*)FindInHashTable(&rnnlm->out_vocab, oosNodeWord)) ) {
            HError(999, "ERROR: in LoadRNLMwgt, did not find oos word(%s) in output vocabulary", oosNodeWord );
        }
        rnnlm->out_oos_nodeid = *p;
        MakeVocabularyClass(rnnlm);
        if ( outputContainOOS ) {
            rnnlm->out_num_word ++ ;
        }

        rnnlm->setvocab = TRUE ;
    }
}

/*  EXPORT-> RNNLMStart */
void RNNLMStart(RNNLM* rnnlm)
{
    int a =0;
    if (rnnlm->usecuedrnnlm)
    {
        if (rnnlm->independent )
        {
            netReset(rnnlm);
        }
    }
    else
    {
        if (rnnlm->independent )
            netReset(rnnlm);
        /*  initialise history */
        for (a=0; a<MAX_NGRAM_ORDER; a++) rnnlm->history[a]=0;
        copyHiddenLayerToInput(rnnlm);
    }

}

/*  EXPORT-> RNNLMEnd */
void RNNLMEnd(RNNLM* rnnlm)
{
    netReset(rnnlm);
}

/* Given the input word: lastword and history vector: hist, forwarding in RNNLM*/
void RNNLMCalcProb(RNNLM* rnnlm, Vector hist, int lastword)
{
    AssignRNNLMHiddenVector (rnnlm, hist);
    copyHiddenLayerToInput(rnnlm);
    if (! isProbTableCached())
    {
        HError(999, "ERROR: in RNNLMCalcProb, the Probablity must be cached");
    }
    if (rnnlm->usecuedrnnlm)
    {
        if (fabs(rnnlm->cuedversion - 1.1) < 1e-6)
        {
            CUEDRNNLMAcceptWord_v1_1 (rnnlm, lastword, 0);
        }
        else if (rnnlm->cuedversion == 1.0)
        {
            CUEDRNNLMAcceptWord_v1_0 (rnnlm, lastword, 0);
        }
        else
        {
            CUEDRNNLMAcceptWord (rnnlm, lastword, 0);
        }
    }
    else if (rnnlm->usefrnnlm)
        FRNNLMAcceptWord (rnnlm, lastword, 0);
    else
        RNNLMAcceptWord (rnnlm, lastword, 0);
}

void softmax (real *ac, int start, int end)
{
    int a;
    real max = ac[start], sum, val;
    for (a=start+1; a<end; a++)
    {
        if (ac[a] > max)
        {
            max = ac[a];
        }
    }
    sum = 0;
    for (a=start; a<end; a++)
    {
        val = ac[a] - max;
        ac[a] = exp(val);
        sum += ac[a];
    }
    for (a=start; a<end; a++)
    {
        ac[a] = ac[a]/sum;
    }
}

void sigmoid (real *ac, int start, int end)
{
    int a;
    for (a=start; a<end; a++)
    {
        ac[a] = 1/(1+exp(-ac[a]));
    }
}

void relu (real *ac, real reluratio, int start, int end)
{
    int a;
    for (a=start; a<end; a++)
    {
        if (ac[a] < 0)  ac[a] = 0;
        else ac[a] = reluratio*ac[a];
    }
}

float CUEDRNNLMAcceptWord_v1_1_SU (RNNLM* rnnlm, int lastword, int curword, int *succword)
    /*
     * ask RNN to accept the last word in history, and return
     * p(curword|lastword ... <s> )
     *
     * after calling this function, the recurrent history vector
     * and the history will update
     *
     */
{
    int a, b, l, nlayer, i;
    float log10p;
    nlayer = rnnlm->num_layer;


    /* </s>, no succeeding word */
    memset (rnnlm->neu_ac_succ[2], 0, sizeof(float)*rnnlm->layersizes[2]);
    for (i=0; i<rnnlm->succeedwords; i++)
    {
        if (succword[i] != INVALID_INT)
        {
            forward (&(rnnlm->_layers[0]), succword[i], curword, rnnlm->neu_ac[0], rnnlm->neu1_ac_succ[i]);
        }
        else
        {
            memset (rnnlm->neu1_ac_succ[i], 0, sizeof(float)*rnnlm->layersizes[1]);
        }
        forward_nosigm (&(rnnlm->layer1_succ[i]), succword, curword, rnnlm->neu1_ac_succ[i], rnnlm->neu_ac_succ[2]);
    }


    sigmoidij (rnnlm->neu_ac_succ[2], rnnlm->layersizes[2]);

    for (l=2; l<rnnlm->nsucclayer; l++)
    {
        forward (&(rnnlm->succlayers[l]), lastword, curword, rnnlm->neu_ac_succ[l], rnnlm->neu_ac_succ[l+1]);
    }

    for (l=0; l<nlayer; l++)
    {
        float *neu0_ac = rnnlm->neu_ac[l];
        float *neu1_ac = rnnlm->neu_ac[l+1];
        if (l == rnnlm->succmergelayer-1)
        {
            forward_nosigm (&(rnnlm->_layers[l]), lastword, curword, neu0_ac, neu1_ac);
            add (neu1_ac, rnnlm->neu_ac_succ[rnnlm->nsucclayer], rnnlm->layersizes[rnnlm->nsucclayer]);
            sigmoidij (neu1_ac, rnnlm->layersizes[3]);
        }
        else
        {
            forward (&(rnnlm->_layers[l]), lastword, curword, neu0_ac, neu1_ac);
        }
    }

    float *neu1_ac = rnnlm->neu_ac[nlayer];
    neu1_ac[rnnlm->outputlayersize-1] = neu1_ac[rnnlm->outputlayersize-1] / (rnnlm->fulldict_size - rnnlm->layersizes[nlayer]+1);
    for (a=0; a<rnnlm->layersizes[nlayer]; a++)
    {
        rnnlm->outP_arr[a] = neu1_ac[a];
    }
    log10p = log10(neu1_ac[curword]);

    return log10p;
}

/* BEGIN modification */
float
calc_vr_prob (
	      RNNLM       *rnnlm,
	      int          curword,
	      const float *input
	      )
{
  
  float ret;
  const float *wgt_i;
  int nrows,j;
  const Layer *layer;
  
  
  assert ( rnnlm->lognormconst > 0.0 );
  
  layer= &(rnnlm->_layers[rnnlm->num_layer-1]);
  nrows= layer->nrows;
  wgt_i= &(layer->U[curword*nrows]);
  
  ret= 0.0;
  for ( j= 0; j < nrows; ++j )
    ret+= input[j]*wgt_i[j];
  ret= exp(ret-rnnlm->lognormconst);
  
  /* Indeed, this prob is later set to 0. */
  if ( curword == rnnlm->outputlayersize-1 )
    /*ret= 99.0;*/
    ret/= (rnnlm->fulldict_size - rnnlm->layersizes[rnnlm->num_layer]+1);
  
  return ret;
  
}
/* END modification */

float CUEDRNNLMAcceptWord_v1_1 (RNNLM* rnnlm, int lastword, int curword)
    /*
     * ask RNN to accept the last word in history, and return
     * p(curword|lastword ... <s> )
     *
     * after calling this function, the recurrent history vector
     * and the history will update
     *
     */
{
    int a, b, l, nlayer;
    float log10p,prob;
    nlayer = rnnlm->num_layer;
    /* BEGIN modification */
    if ( rnnlm->lognormconst <= 0 ) /* No vr */
      {
	for (l=0; l<nlayer; l++)
	  {
	    float *neu0_ac = rnnlm->neu_ac[l];
	    float *neu1_ac = rnnlm->neu_ac[l+1];
	    forward (&(rnnlm->_layers[l]), lastword, curword, neu0_ac, neu1_ac);
	  }
	float *neu1_ac = rnnlm->neu_ac[nlayer];
	neu1_ac[rnnlm->outputlayersize-1] = neu1_ac[rnnlm->outputlayersize-1] / (rnnlm->fulldict_size - rnnlm->layersizes[nlayer]+1);
	for (a=0; a<rnnlm->layersizes[nlayer]; a++)
	  {
	    rnnlm->outP_arr[a] = neu1_ac[a];
	  }
	log10p = log10(neu1_ac[curword]);
      }
    else /* Vr */
      {
	for (l=0; l<(nlayer-1); l++)
          {
            float *neu0_ac = rnnlm->neu_ac[l];
            float *neu1_ac = rnnlm->neu_ac[l+1];
            forward (&(rnnlm->_layers[l]), lastword, curword, neu0_ac, neu1_ac);
          }
	for (a=0; a<rnnlm->layersizes[nlayer]; a++)
	  rnnlm->outP_arr[a] = -99.0; /* Must be calculated. */
	rnnlm->outP_arr[curword]= prob=
	  calc_vr_prob ( rnnlm, curword, rnnlm->neu_ac[nlayer-1] );
	memcpy ( rnnlm->in_llayer_arr, rnnlm->neu_ac[nlayer-1],
		 sizeof(float)*rnnlm->layersizes[nlayer-1]);
	log10p= log10 ( prob );
      }
    /* END modification */
    return log10p;
}

float CUEDRNNLMAcceptWord_v1_0 (RNNLM* rnnlm, int lastword, int curword)
    /*
     * ask RNN to accept the last word in history, and return
     * p(curword|lastword ... <s> )
     *
     * after calling this function, the recurrent history vector
     * and the history will update
     *
     */
{
    int a, b, l, nlayer;
    float log10p;
    nlayer = rnnlm->num_layer;
    for (l=0; l<nlayer; l++)
    {
        float *neu0_ac = rnnlm->neu_ac[l];
        float *neu1_ac = rnnlm->neu_ac[l+1];
        forward (&(rnnlm->_layers[l]), lastword, curword, neu0_ac, neu1_ac);
    }
    float *neu1_ac = rnnlm->neu_ac[nlayer];
    neu1_ac[rnnlm->outputlayersize-1] = neu1_ac[rnnlm->outputlayersize-1] / (rnnlm->fulldict_size - rnnlm->layersizes[nlayer]+1);
    for (a=0; a<rnnlm->layersizes[nlayer]; a++)
    {
        rnnlm->outP_arr[a] = neu1_ac[a];
    }
    log10p = log10(neu1_ac[curword]);
    return log10p;
}

float CUEDRNNLMAcceptWord(RNNLM* rnnlm, int lastword, int curword)
    /*
     * ask RNN to accept the last word in history, and return
     * p(curword|lastword ... <s> )
     *
     * after calling this function, the recurrent history vector
     * and the history will update
     *
     */
{
#ifdef SAVEACTNOSIGMOID   /*save the active without sigmoid function in the hidden layer.*/
    double *acwithnosigmoid = (double *)malloc (rnnlm->layersizes[rnnlm->num_layer] * sizeof(double));
#endif
#ifdef SAVEACTWITHINPUTWEIGHTMATRIX   /*save the active after it pass the first layer, before adding with the word.*/
    double *actforhist =  (double *)malloc(rnnlm->layersizes[1] * sizeof(double));
#endif
    int a, b, l, nlayer;
    real val;
    double sum;
    float log10p;
    real *srcac, *dstac, *wgt;
    real prob;
    int nrow, ncol;
    /*   sum is used for normalization: it's better to have larger precision as many numbers are summed together here */

    /*-----------------------------------------------------------------------------
     *  set input neuron value
     *-----------------------------------------------------------------------------*/
    rnnlm->neu_ac[0][lastword] = 1;

    /*-----------------------------------------------------------------------------
     *  propagate 0->1
     *-----------------------------------------------------------------------------*/
    nlayer = rnnlm->num_layer;
    for (l=0; l<nlayer; l++)
    {
        if (l==0)
        {
            srcac = rnnlm->neu0_ac_hist;
            wgt = rnnlm->layer0_hist;
            dstac = rnnlm->neu_ac[1];
            nrow = rnnlm->layersizes[1];
            ncol = rnnlm->layersizes[1];
            memset (dstac, 0, sizeof(real)*ncol);
            for (b=0; b<ncol; b++)
            {
                int nrow_word = rnnlm->layersizes[0];
                dstac[b] = rnnlm->layers[0][lastword+b*nrow_word];
            }
            if (rnnlm->dim_fea > 0)
            {
                real *feasrcac = rnnlm->neu0_ac_fea;
                real *feawgt = rnnlm->layer0_fea;
                matrixXvector_cuedrnnlm (feasrcac, feawgt, dstac, rnnlm->dim_fea, ncol);
            }
        }
        else
        {
            srcac = rnnlm->neu_ac[l];
            wgt = rnnlm->layers[l];
            dstac = rnnlm->neu_ac[l+1];
            nrow = rnnlm->layersizes[l];
            ncol = rnnlm->layersizes[l+1];
            memset (dstac, 0, sizeof(real)*ncol);
        }
        if (l+1 == nlayer)
	  {
            if (rnnlm->lognormconst < 0)
            {
                if (rnnlm->nclass > 0)
		  {
                    float *clsdstac = rnnlm->neuN_ac_class;
                    float *clswgt   = rnnlm->layerN_class;
                    int ncol_cls = rnnlm->nclass;
                    matrixXvector_cuedrnnlm (srcac, clswgt, clsdstac, nrow, ncol_cls);
                    /* softmax function */
                    softmax (clsdstac, 0, ncol_cls);
                    int clsid = rnnlm->word2class[curword];
                    int swordid = rnnlm->classinfo[clsid*3];
                    int ewordid = rnnlm->classinfo[clsid*3+1];
                    int nword   = rnnlm->classinfo[clsid*3+2];
                    matrixXvector_cuedrnnlm (srcac, wgt+swordid*nrow, dstac+swordid, nrow, nword);
                    softmax(dstac, swordid, ewordid+1);
                }
                else
                {
                    matrixXvector_cuedrnnlm (srcac, wgt, dstac, nrow, ncol);
                    softmax (dstac, 0, ncol);
                }
            }
            else
            {
#if 0
                float v = 0;
                for (a=0; a<nrow; a++)
                {
                    v += srcac[a]*wgt[a+nrow*curword];
                }
                dstac[curword] = exp(v-rnnlm->lognormconst);
#else
                matrixXvector_cuedrnnlm (srcac, wgt, dstac, nrow, ncol);
                for (a=0; a<rnnlm->layersizes[nlayer]; a++)
                {
                    val = rnnlm->neu_ac[nlayer][a] - rnnlm->lognormconst;
                    val = exp(val);
                    rnnlm->neu_ac[nlayer][a] = val;
                }
#endif
            }
            /* compute OOS node probability */
            rnnlm->neu_ac[nlayer][rnnlm->outputlayersize-1] = rnnlm->neu_ac[nlayer][rnnlm->outputlayersize-1] / (rnnlm->fulldict_size - rnnlm->layersizes[nlayer]+1);
            for (a=0; a<rnnlm->layersizes[nlayer]; a++)
            {
                rnnlm->outP_arr[a] = rnnlm->neu_ac[nlayer][a];
            }
        }
        else
        {
            matrixXvector_cuedrnnlm (srcac, wgt, dstac, nrow, ncol);
            if (rnnlm->nodetype == 0)
            {
                sigmoid (dstac, 0, ncol);
            }
            else if (rnnlm->nodetype == 1)
            {
                relu (dstac, rnnlm->reluratio, 0, ncol);
            }
            else
            {
                printf ("ERROR: unknown type of hidde nnode: %d\n", rnnlm->nodetype);
                exit (0);
            }
        }
    }
    if (rnnlm->nclass  > 0)
    {
        int clsid = rnnlm->word2class[curword];
        prob = dstac[curword]*rnnlm->neuN_ac_class[clsid];
    }
    else
    {
        prob = dstac[curword];
    }
    log10p = log10(rnnlm->neu_ac[nlayer][curword]);


    copyHiddenLayerToInput (rnnlm);
    /* copy ac from hidden layer to input layer */
    for (a=0; a<rnnlm->layersizes[1]; a++) {
        rnnlm->neu0_ac_hist[a] = rnnlm->neu_ac[1][a];
    }
    return log10p;
}

/*  EXPORT-> FRNNLMAcceptWord */
float FRNNLMAcceptWord(RNNLM* rnnlm, int lastword, int curword)
    /*
     * ask RNN to accept the last word in history, and return
     * p(curword|lastword ... <s> )
     *
     * after calling this function, the recurrent history vector
     * and the history will update
     *
     */
{
#ifdef SAVEACTNOSIGMOID   /*save the active without sigmoid function in the hidden layer.*/
    double *acwithnosigmoid =  (double *)malloc(rnnlm->layer1_size * sizeof(double));
#endif
#ifdef SAVEACTWITHINPUTWEIGHTMATRIX   /*save the active after it pass the first layer, before adding with the word.*/
    double *actforhist =  (double *)malloc(rnnlm->layer1_size * sizeof(double));
#endif
    int a, b;
    real val;
    double sum;
    float log10p;
    /*   sum is used for normalization: it's better to have larger precision as many numbers are summed together here */

#if 0
    unsigned long long hash[MAX_NGRAM_ORDER];
    /* this will hold pointers to syn_d that contains hash parameters */

    int classid=0;
#endif

    /*-----------------------------------------------------------------------------
     *  set input neuron value
     *-----------------------------------------------------------------------------*/
    if (lastword!=-1) rnnlm->neu0[lastword].ac=1;

    /*-----------------------------------------------------------------------------
     *  propagate 0->1
     *-----------------------------------------------------------------------------*/
    for (a=0; a<rnnlm->layer1_size; a++) rnnlm->neu1[a].ac=0;
    for (a=0; a<rnnlm->layerc_size; a++) rnnlm->neuc[a].ac=0;
    /*  propagate delayed hidden vector to current hidden vector */
    matrixXvector(  rnnlm->neu1, rnnlm->neu0,               /*  destvec, srcvec */
                    rnnlm->syn0, rnnlm->layer0_size,        /*  matrix,  matrix_width */
                    0, rnnlm->layer1_size,                  /*  dest start id, dest end id */
rnnlm->layer0_size-rnnlm->layer1_size, rnnlm->layer0_size,  /*  src  statt id, src  end id */
                    0                                       /*  type */                );

#ifdef SAVEACTWITHINPUTWEIGHTMATRIX
    for (b =0; b<rnnlm->layer1_size; b++)
    {
        actforhist[b] = rnnlm->neu1[b].ac;
    }
    for (a=0; a<rnnlm->layer1_size; a++) {
        if (actforhist[a]>50) actforhist[a]=50;  /* for numerical stability */
        if (actforhist[a]<-50) actforhist[a]=-50;  /* for numerical stability */
        val=-actforhist[a];
#ifdef FASTEXP
        actforhist[a]=1/(1+FAST_EXP(val));
#else
        actforhist[a]=1/(1+expf(val));
#endif
    }
#endif
    /*  propagate input topic information to current hidden vector */
    if (rnnlm->layer0_topic_size>0)
       {
          matrixXvector(rnnlm->neu1, rnnlm->neu0_topic, rnnlm->syn0_topic, rnnlm->layer0_topic_size, 0, rnnlm->layer1_size, 0, rnnlm->layer0_topic_size, 0);
       }

    /*  propagate last word's weight to current hidden vector */
    for (b=0; b<rnnlm->layer1_size; b++) {
        a=lastword;
        if (a!=-1) rnnlm->neu1[b].ac +=  rnnlm->syn0[a+b*rnnlm->layer0_size].weight;
    }


#ifdef SAVEACTNOSIGMOID   /*temp code */
    for (b =0; b<rnnlm->layer1_size; b++)
    {
        acwithnosigmoid[b] = rnnlm->neu1[b].ac;
    }
#endif
    /*-----------------------------------------------------------------------------
     *  sigmoid of neuron 1
     *-----------------------------------------------------------------------------*/
    for (a=0; a<rnnlm->layer1_size; a++) {
        if (rnnlm->neu1[a].ac>50) rnnlm->neu1[a].ac=50;  /* for numerical stability */
        if (rnnlm->neu1[a].ac<-50) rnnlm->neu1[a].ac=-50;  /* for numerical stability */
        val=-rnnlm->neu1[a].ac;
#ifdef FASTEXP
        rnnlm->neu1[a].ac=1/(1+FAST_EXP(val));
#else
        rnnlm->neu1[a].ac=1/(1+expf(val));
#endif

    }
    if (rnnlm->layerc_size>0) {
        matrixXvector(rnnlm->neuc, rnnlm->neu1,
                        rnnlm->syn1, rnnlm->layer1_size,
                        0, rnnlm->layerc_size,
                        0, rnnlm->layer1_size,
                    0);
        /* activate compression      --sigmoid */
        for (a=0; a<rnnlm->layerc_size; a++) {
            if (rnnlm->neuc[a].ac>50) rnnlm->neuc[a].ac=50;  /* for numerical stability */
            if (rnnlm->neuc[a].ac<-50) rnnlm->neuc[a].ac=-50;  /* for numerical stability */
            val=-rnnlm->neuc[a].ac;
#ifdef FASTEXP
            rnnlm->neuc[a].ac=1/(1+FAST_EXP(val));
#else
            rnnlm->neuc[a].ac=1/(1+expf(val));
#endif
        }
    }

    /*-----------------------------------------------------------------------------
     *  propagate 1-> 2.words
     *-----------------------------------------------------------------------------*/
    if (curword != -1)
    {
        for (b=0; b<rnnlm->layer2_size; b++)  rnnlm->neu2[b].ac = 0;
        if (rnnlm->layerc_size > 0)
        {
            HError(999,"In RNNLM: for the F-RNNLM, layerc_size is not implemented yet.");
        }
        else
        {
            matrixXvector (rnnlm->neu2, rnnlm->neu1, rnnlm->syn1,
                           rnnlm->layer1_size, 0,
                           rnnlm->layer2_size, 0,
                           rnnlm->layer1_size, 0);
        }
        if (rnnlm->layer0_topic_size > 0)
           {
              matrixXvector(rnnlm->neu2, rnnlm->neu0_topic, rnnlm->syn_d_topic,
                            rnnlm->layer0_topic_size, 0,
                            rnnlm->layer2_size, 0,
                            rnnlm->layer0_topic_size, 0);
           }
    }

    /*-----------------------------------------------------------------------------
     *  softmax function on words
     *-----------------------------------------------------------------------------*/
    if (rnnlm->direct_size > 0)
    {
        HError(999,"In RNNLM: for the F-RNNLM, direct_size is not implemented yet.");
    }
    sum = 0;
    if (curword != -1)
    {
        if (rnnlm->lognorm >= 0)
        {
            for (a=0; a<rnnlm->layer2_size; a++)
            {
                val = rnnlm->neu2[a].ac - rnnlm->lognorm;
                val = expf(val);
                rnnlm->neu2[a].ac = val;
                rnnlm->outP_arr[a]=rnnlm->neu2[a].ac;
            }
        }
        else
        {
            for (a=0; a<rnnlm->layer2_size; a++)
            {
                if (rnnlm->neu2[a].ac > 50) rnnlm->neu2[a].ac = 50;  /* for numerical stability */
                if (rnnlm->neu2[a].ac < -50) rnnlm->neu2[a].ac = -50;  /* for numerical stability */
#ifdef FASTEXP
                val = FAST_EXP(rnnlm->neu2[a].ac);
#else
                val = expf(rnnlm->neu2[a].ac);
#endif
                sum += val;
                rnnlm->neu2[a].ac = val;
            }
            for (a=0; a<rnnlm->layer2_size; a++)
            {
                rnnlm->neu2[a].ac /= sum;
                rnnlm->outP_arr[a]=rnnlm->neu2[a].ac;
            }
        }
    }
    /* special case: oos */
    if (rnnlm->out_oos_nodeid != rnnlm->layer2_size-1)
    {
        HError(999,"In RNNLM: oosNodeWord(%d) should be last words in layer2 (size: %d).", oosNodeWord, rnnlm->layer2_size);
    }
    /* rnnlm->fulldict_size = 28876; */
    if (rnnlm->fulldict_size == 0)
    {
        HError(999,"In RNNLM: fulldict_size is not specified.");
    }
    if (rnnlm->fulldict_size < rnnlm->layer2_size)
    {
        HError(999, "In RNNLM, fulldict_size should not be less than layer2_size.");
    }
    rnnlm->neu2[rnnlm->layer2_size-1].ac = rnnlm->neu2[rnnlm->layer2_size-1].ac / (rnnlm->fulldict_size - rnnlm->layer2_size + 1);
    rnnlm->outP_arr[rnnlm->layer2_size-1] = rnnlm->outP_arr[rnnlm->layer2_size-1] / (rnnlm->fulldict_size - rnnlm->layer2_size + 1);


    /*-----------------------------------------------------------------------------
     *  final, compute log10 probability
     *-----------------------------------------------------------------------------*/
    if (curword != -1)
    {
        log10p = log10(rnnlm->neu2[curword].ac);
    }
    else
    {
        log10p = 0;
    }

    /* cancel activation caused by lastword */
    if ( lastword != -1 )
        rnnlm->neu0[lastword].ac= 0.0 ;
    for (a=MAX_NGRAM_ORDER-1; a>0; a--)
       rnnlm->history[a]=rnnlm->history[a-1];

    /*  curword is the id maps by out_vocab
     *  need to convert to id in in_vocab,
     *  update rnnlm->histroy[0] is moved to where this function is called */

    copyHiddenLayerToInput(rnnlm);

#ifdef SAVEACTNOSIGMOID
    for (b =0; b<rnnlm->layer1_size; b++)
    {
        rnnlm->neu1[b].ac =  acwithnosigmoid[b];
    }
    free (acwithnosigmoid);
#endif
#ifdef SAVEACTWITHINPUTWEIGHTMATRIX
    for (b =0; b<rnnlm->layer1_size; b++)
    {
        rnnlm->neu1[b].ac =  actforhist[b];
    }
    free (actforhist);
#endif
    return log10p;
}

/*  EXPORT-> RNNLMAcceptWord */
float RNNLMAcceptWord(RNNLM* rnnlm, int lastword, int curword)
    /*
     * ask RNN to accept the last word in history, and return
     * p(curword|lastword ... <s> )
     *
     * after calling this function, the recurrent history vector
     * and the history will update
     *
     */
{
#ifdef SAVEACTNOSIGMOID   /*save the active without sigmoid function in the hidden layer.*/
    real *acwithnosigmoid =  (real *)malloc(rnnlm->layer1_size * sizeof(real));
#endif
#ifdef SAVEACTWITHINPUTWEIGHTMATRIX   /*save the active after it pass the first layer, before adding with the word.*/
    real *actforhist =  (real *)malloc(rnnlm->layer1_size * sizeof(real));
#endif
    int a, b, c;
    real val;
    double sum;
    float log10p;
    /*   sum is used for normalization: it's better to have larger precision as many numbers are summed together here */

    unsigned long long hash[MAX_NGRAM_ORDER];
    /* this will hold pointers to syn_d that contains hash parameters */

    int classid=0;

    /*-----------------------------------------------------------------------------
     *  set input neuron value
     *-----------------------------------------------------------------------------*/
    if (lastword!=-1) rnnlm->neu0[lastword].ac=1;

    /*-----------------------------------------------------------------------------
     *  propagate 0->1
     *-----------------------------------------------------------------------------*/
    for (a=0; a<rnnlm->layer1_size; a++) rnnlm->neu1[a].ac=0;
    for (a=0; a<rnnlm->layerc_size; a++) rnnlm->neuc[a].ac=0;
    /*  propagate delayed hidden vector to current hidden vector */
    matrixXvector(  rnnlm->neu1, rnnlm->neu0,               /*  destvec, srcvec */
                    rnnlm->syn0, rnnlm->layer0_size,        /*  matrix,  matrix_width */
                    0, rnnlm->layer1_size,                  /*  dest start id, dest end id */
rnnlm->layer0_size-rnnlm->layer1_size, rnnlm->layer0_size,  /*  src  statt id, src  end id */
                    0                                       /*  type */                );

#ifdef SAVEACTWITHINPUTWEIGHTMATRIX
    for (b =0; b<rnnlm->layer1_size; b++)
    {
        actforhist[b] = rnnlm->neu1[b].ac;
    }
    for (a=0; a<rnnlm->layer1_size; a++) {
        if (actforhist[a]>50) actforhist[a]=50;  /* for numerical stability */
        if (actforhist[a]<-50) actforhist[a]=-50;  /* for numerical stability */
        val=-actforhist[a];
        actforhist[a]=1/(1+FAST_EXP(val));
    }
#endif
    /*  propagate last word's weight to current hidden vector */
    for (b=0; b<rnnlm->layer1_size; b++) {
        a=lastword;
        if (a!=-1) rnnlm->neu1[b].ac +=  rnnlm->syn0[a+b*rnnlm->layer0_size].weight;
    }


#ifdef SAVEACTNOSIGMOID   /*temp code */
    for (b =0; b<rnnlm->layer1_size; b++)
    {
        acwithnosigmoid[b] = rnnlm->neu1[b].ac;
    }
#endif
    /*-----------------------------------------------------------------------------
     *  sigmoid of neuron 1
     *-----------------------------------------------------------------------------*/
    for (a=0; a<rnnlm->layer1_size; a++) {
        if (rnnlm->neu1[a].ac>50) rnnlm->neu1[a].ac=50;  /* for numerical stability */
        if (rnnlm->neu1[a].ac<-50) rnnlm->neu1[a].ac=-50;  /* for numerical stability */
        val=-rnnlm->neu1[a].ac;
        rnnlm->neu1[a].ac=1/(1+FAST_EXP(val));
    }
    if (rnnlm->layerc_size>0) {
        matrixXvector(rnnlm->neuc, rnnlm->neu1,
                        rnnlm->syn1, rnnlm->layer1_size,
                        0, rnnlm->layerc_size,
                        0, rnnlm->layer1_size,
                    0);
        /* activate compression      --sigmoid */
        for (a=0; a<rnnlm->layerc_size; a++) {
            if (rnnlm->neuc[a].ac>50) rnnlm->neuc[a].ac=50;  /* for numerical stability */
            if (rnnlm->neuc[a].ac<-50) rnnlm->neuc[a].ac=-50;  /* for numerical stability */
            val=-rnnlm->neuc[a].ac;
            rnnlm->neuc[a].ac=1/(1+FAST_EXP(val));
        }
    }

#if 0    /* temp code */
    for (b=0; b<rnnlm->layer2_size; b++) rnnlm->neu2[b].ac=0;
        /*  calculate the output cluster node's activation  */
	    matrixXvector(rnnlm->neu2, rnnlm->neu1,
                        rnnlm->syn1, rnnlm->layer1_size,
                        0, rnnlm->vocab_size,
                        0, rnnlm->layer1_size,
                        0);

    /*-----------------------------------------------------------------------------
     *  layer 2 -- softmax
     *-----------------------------------------------------------------------------*/
    sum=0;
    for (a=0; a<rnnlm->vocab_size; a++) {
        if (rnnlm->neu2[a].ac>50) rnnlm->neu2[a].ac=50;  /* for numerical stability */
        if (rnnlm->neu2[a].ac<-50) rnnlm->neu2[a].ac=-50;  /* for numerical stability */
        val=FAST_EXP(rnnlm->neu2[a].ac);
        sum+=val;
        rnnlm->neu2[a].ac=val;
    }
    for (a=0; a<rnnlm->vocab_size; a++) rnnlm->neu2[a].ac/=sum;
    /*-----------------------------------------------------------------------------
     *  prob table caching
     *-----------------------------------------------------------------------------*/
    if ( calcProbCache )
    {
        for (a=0; a<rnnlm->vocab_size; a++) rnnlm->outP_arr[a] = rnnlm->neu2[a].ac;
    }

    /*-----------------------------------------------------------------------------
     *  final, compute log10 probability
     *-----------------------------------------------------------------------------*/
    if ( curword!=-1)
    {
        log10p = log10(rnnlm->neu2[curword].ac);
    }
    else
    {
        /*  do not handle oov on rnn its own !!! */
        log10p = 0;
    }

    /* cancel activation caused by lastword */
    if ( lastword != -1 )
        rnnlm->neu0[lastword].ac= 0.0 ;
    for (a=MAX_NGRAM_ORDER-1; a>0; a--)
       rnnlm->history[a]=rnnlm->history[a-1];

    /*  curword is the id maps by out_vocab
     *  need to convert to id in in_vocab,
     *  update rnnlm->histroy[0] is moved to where this function is called */

    copyHiddenLayerToInput(rnnlm);

#ifdef SAVEACTNOSIGMOID
    for (b =0; b<rnnlm->layer1_size; b++)
    {
        rnnlm->neu1[b].ac =  acwithnosigmoid[b];
    }
    free (acwithnosigmoid);
#endif
#ifdef SAVEACTWITHINPUTWEIGHTMATRIX
    for (b =0; b<rnnlm->layer1_size; b++)
    {
        rnnlm->neu1[b].ac =  actforhist[b];
    }
    free (actforhist);
#endif

    return log10p;
#endif

    /*-----------------------------------------------------------------------------
     *  propagate 1-> 2.classes
     *-----------------------------------------------------------------------------*/
    for (b=rnnlm->vocab_size; b<rnnlm->layer2_size; b++) rnnlm->neu2[b].ac=0;
    if (rnnlm->layerc_size>0) {
	    matrixXvector(rnnlm->neu2, rnnlm->neuc,
                        rnnlm->sync, rnnlm->layerc_size,
                        rnnlm->vocab_size, rnnlm->layer2_size,
                        0, rnnlm->layerc_size,
                        0);
    }
    else
    {
        /*  calculate the output cluster node's activation  */
	    matrixXvector(rnnlm->neu2, rnnlm->neu1,
                        rnnlm->syn1, rnnlm->layer1_size,
                        rnnlm->vocab_size, rnnlm->layer2_size,
                        0, rnnlm->layer1_size,
                        0);
        /*  forward to class node */
    }


    /*-----------------------------------------------------------------------------
     *  direct connection -- part 1. to classes
     *-----------------------------------------------------------------------------*/
    if (rnnlm->direct_size>0) {
        if ( calcProbCache == TRUE) {
            HError(999,"In RNNLM: direction connection with prob table caching is not supported at the moment.");
        }

        for (a=0; a<rnnlm->direct_order; a++) hash[a]=0;

        for (a=0; a<rnnlm->direct_order; a++) {
            b=0;
            if (a>0 && rnnlm->history[a-1]==-1)
                break;
            /* if OOV was in history, do not use this N-gram feature and higher orders */
            hash[a]=PRIMES[0]*PRIMES[1];

            for (b=1; b<=a; b++)
                hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(rnnlm->history[b-1]+1);
            /* update hash value based on words from the history */
            hash[a]=hash[a]%(rnnlm->direct_size/2);
            /* make sure that starting hash index is in the first half of syn_d
             * (second part is reserved for history->words features) */
        }

        for (a=rnnlm->vocab_size; a<rnnlm->layer2_size; a++)
            /*  for each class node */
        {
            for (b=0; b<rnnlm->direct_order; b++)
            {
                if (hash[b])
                {
                    rnnlm->neu2[a].ac+=rnnlm->syn_d[hash[b]];		/* apply current parameter and move to the next one */
                    hash[b]++;
                }
                else
                    break;
            }
        }
    }

    /*-----------------------------------------------------------------------------
     *  layer 2.class -- softmax
     *-----------------------------------------------------------------------------*/
    sum=0;
    for (a=rnnlm->vocab_size; a<rnnlm->layer2_size; a++) {
        if (rnnlm->neu2[a].ac>50) rnnlm->neu2[a].ac=50;  /* for numerical stability */
        if (rnnlm->neu2[a].ac<-50) rnnlm->neu2[a].ac=-50;  /* for numerical stability */
        val=FAST_EXP(rnnlm->neu2[a].ac);
        sum+=val;
        rnnlm->neu2[a].ac=val;
    }
    for (a=rnnlm->vocab_size; a<rnnlm->layer2_size; a++) rnnlm->neu2[a].ac/=sum;
    /* output layer activations now sum exactly to 1 */

    /*-----------------------------------------------------------------------------
     *  propagate 1 -> 2.word
     *-----------------------------------------------------------------------------*/

    if ( calcProbCache == FALSE ) {
        if (curword!=-1) {
            /*  not oov */
            int classid=rnnlm->vocab[curword].class_index;
            int wordInClass= rnnlm->class_cn[classid];
            for (c=0; c< wordInClass; c++)
            {
                rnnlm->neu2[rnnlm->class_words[classid][c]].ac=0;
            }
            if (rnnlm->layerc_size>0) {
                matrixXvector(rnnlm->neu2, rnnlm->neuc,                 /*  dest = neur2 ; src = neuc */
                    rnnlm->sync, rnnlm->layerc_size,        /*  use sync matrix, width=layerc_size */
                    rnnlm->class_words[classid][0],         /*  classid-th class start id */
                    rnnlm->class_words[classid][0]+rnnlm->class_cn[classid], /*  classid-th class end id */
                    0, rnnlm->layerc_size,
                    0);
            }
            else
            {
                matrixXvector(rnnlm->neu2, rnnlm->neu1,
                    rnnlm->syn1, rnnlm->layer1_size,
                    rnnlm->class_words[classid][0],
                    rnnlm->class_words[classid][0]+rnnlm->class_cn[classid],
                    0, rnnlm->layer1_size,
                    0);
            }
        }
    }
    else
    {
        int classid, wordInClass = 0 ;
        for (classid = 0 ; classid < rnnlm->class_size; classid ++) {
            wordInClass = rnnlm->class_cn[classid] ;
            if ( wordInClass == 1 ) {   /*  no need to calculate  */

            }
            else
            {
                for (c=0; c<wordInClass; c++) rnnlm->neu2[rnnlm->class_words[classid][c]].ac = 0 ;
                if (rnnlm->layerc_size>0) {
                    matrixXvector(rnnlm->neu2, rnnlm->neuc,                 /*  dest = neur2 ; src = neuc */
                        rnnlm->sync, rnnlm->layerc_size,        /*  use sync matrix, width=layerc_size */
                        rnnlm->class_words[classid][0],         /*  classid-th class start id */
                        rnnlm->class_words[classid][0]+rnnlm->class_cn[classid], /*  classid-th class end id */
                        0, rnnlm->layerc_size,
                        0);
                }
                else
                {
                    matrixXvector(rnnlm->neu2, rnnlm->neu1,
                        rnnlm->syn1, rnnlm->layer1_size,
                        rnnlm->class_words[classid][0],
                        rnnlm->class_words[classid][0]+rnnlm->class_cn[classid],
                        0, rnnlm->layer1_size,
                        0);
                }
            }
        }
    }

    /*-----------------------------------------------------------------------------
     *  direct connections -- part 2 to words
     *-----------------------------------------------------------------------------*/
    if (curword!=-1 &&  rnnlm-> direct_size>0) {
        unsigned long long hash[MAX_NGRAM_ORDER];
        int classid = rnnlm->vocab[curword].class_index;

        for (a=0; a<rnnlm->direct_order; a++)
            hash[a]=0;

        for (a=0; a<rnnlm->direct_order; a++) {
            b=0;
            if (a>0 && rnnlm->history[a-1]==-1)
                break;
            hash[a]=PRIMES[0]*PRIMES[1]*(unsigned long long)(classid+1);

            for (b=1; b<=a; b++) hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(rnnlm->history[b-1]+1);
            hash[a]=(hash[a]%(rnnlm->direct_size/2))+(rnnlm->direct_size)/2;
        }

        for (c=0; c<rnnlm->class_cn[classid]; c++) {
            a=rnnlm->class_words[classid][c];

            for (b=0; b<rnnlm->direct_order; b++)
            {
                if (hash[b])
                {
                    rnnlm->neu2[a].ac+=rnnlm->syn_d[hash[b]];
                    hash[b]++;
                    hash[b]=hash[b]%rnnlm->direct_size;
                } else
                    break;
            }
        }
    }


    /*-----------------------------------------------------------------------------
     *  layer 2.word -- softmax
     *-----------------------------------------------------------------------------*/
    sum=0;

    if ( calcProbCache == FALSE  ) {
        if (curword!=-1) {
            classid =  rnnlm->vocab[curword].class_index;
            for (c=0; c<rnnlm->class_cn[classid]; c++)
            {
                a=rnnlm->class_words[classid][c];
                if (rnnlm->neu2[a].ac>50) rnnlm->neu2[a].ac=50;  /* for numerical stability */
                if (rnnlm->neu2[a].ac<-50) rnnlm->neu2[a].ac=-50;  /* for numerical stability */
                val=FAST_EXP(rnnlm->neu2[a].ac);
                sum+=val;
                rnnlm->neu2[a].ac=val;
            }
            for (c=0; c<rnnlm->class_cn[classid]; c++) {
                int wordid=rnnlm->class_words[classid][c];
                rnnlm->neu2[wordid].ac/=sum;
            }
        }
    }
    else
    {
        for (classid=0; classid< rnnlm->class_size; classid++) {
            if ( rnnlm->class_cn[classid] == 1 ) {
                rnnlm->neu2[rnnlm->class_words[classid][0]].ac = 1.0f;
            }
            else{
                sum=0;
                for (c=0; c<rnnlm->class_cn[classid]; c++)
                {
                    a=rnnlm->class_words[classid][c];
                    if (rnnlm->neu2[a].ac>50) rnnlm->neu2[a].ac=50;  /* for numerical stability */
                    if (rnnlm->neu2[a].ac<-50) rnnlm->neu2[a].ac=-50;  /* for numerical stability */
                    val=FAST_EXP(rnnlm->neu2[a].ac);
                    sum+=val;
                    rnnlm->neu2[a].ac=val;
                }
                for (c=0; c<rnnlm->class_cn[classid]; c++) {
                    int wordid=rnnlm->class_words[classid][c];
                    rnnlm->neu2[wordid].ac/=sum;
                }
            }
        }
    }


    /*-----------------------------------------------------------------------------
     *  prob table caching
     *-----------------------------------------------------------------------------*/
    if ( calcProbCache )
    {
        /* outP_arr stores the real probability value */
        for (classid = 0 ; classid < rnnlm->class_size; classid ++)
        {
            float classprob = rnnlm->neu2[classid+rnnlm->vocab_size].ac;
            for ( c=0; c<rnnlm->class_cn[classid] ; c++){
                int wordid=rnnlm->class_words[classid][c];
                rnnlm->outP_arr[wordid]=classprob * rnnlm->neu2[wordid].ac;
            }
        }
        /*  special case : oos */
        if ( oosNodeClass < 0 )
            rnnlm->outP_arr[rnnlm->vocab_size-1] = rnnlm->neu2[rnnlm->layer2_size -1 ].ac ;
        else
            rnnlm->outP_arr[rnnlm->vocab_size-1] = rnnlm->neu2[rnnlm->vocab_size + oosNodeClass ].ac;


#if 1
        if (rnnlm->fulldict_size == 0)
        {
            HError(999,"In RNNLM: fulldict_size is not specified.");
        }
        if (rnnlm->fulldict_size < rnnlm->vocab_size)
        {
            HError(999, "In RNNLM, fulldict_size should not be less than vocab_size.");
        }
        rnnlm->outP_arr[rnnlm->vocab_size-1] = rnnlm->outP_arr[rnnlm->vocab_size-1] / (rnnlm->fulldict_size - rnnlm->vocab_size + 1);
#endif
#if 0
        int iwrd= 0;
        float probmass=0;
        for ( iwrd =0; iwrd < rnnlm->vocab_size; iwrd++)
            probmass+=rnnlm->outP_arr[iwrd] ;
        printf("prob mass: %.3f\n", probmass);
#endif
    }

    /*-----------------------------------------------------------------------------
     *  final, compute log10 probability
     *-----------------------------------------------------------------------------*/
    if ( curword!=-1)
    {
        classid =  rnnlm->vocab[curword].class_index;
        log10p = log10(rnnlm->neu2[curword].ac)
                    +
                 log10(rnnlm->neu2[classid+rnnlm->vocab_size].ac);
    }
    else
    {
        /*  do not handle oov on rnn its own !!! */
        log10p = 0;
    }

    /* cancel activation caused by lastword */
    if ( lastword != -1 )
        rnnlm->neu0[lastword].ac= 0.0 ;
    for (a=MAX_NGRAM_ORDER-1; a>0; a--)
       rnnlm->history[a]=rnnlm->history[a-1];

    /*  curword is the id maps by out_vocab
     *  need to convert to id in in_vocab,
     *  update rnnlm->histroy[0] is moved to where this function is called */

    copyHiddenLayerToInput(rnnlm);

#ifdef SAVEACTNOSIGMOID
    for (b =0; b<rnnlm->layer1_size; b++)
    {
        rnnlm->neu1[b].ac =  acwithnosigmoid[b];
    }
    free (acwithnosigmoid);
#endif
#ifdef SAVEACTWITHINPUTWEIGHTMATRIX
    for (b =0; b<rnnlm->layer1_size; b++)
    {
        rnnlm->neu1[b].ac =  actforhist[b];
    }
    free (actforhist);
#endif
    return log10p;
}



/*-----------------------------------------------------------------------------
 *  test functions
 *-----------------------------------------------------------------------------*/

/*  EXPORT -> RNNOutputLogProb(char* modelfn, char* textfn) */
void RNNOutputLogProb(char* modelfn, char* textfn)
{
    int wordcn=0;
    int sentcn=0;
    int lastword = 0;
    int curword  = 0;
    float logp   = 0;
    float logcurp= 0;
    FILE* fp = NULL;
    char buf[MAXSTRLEN];

    RNNLM* rnnlm=CreateRNNLM(&rnnInfoStack);
    LoadRNNLM(modelfn, rnnlm);

    fp = fopen(textfn, "rb");
    RNNLMStart(rnnlm);
    while (1)
    {
        readWord(buf, fp);
        curword=searchRNNVocab(rnnlm, buf);
        if (rnnlm->usecuedrnnlm)
        {
            if (fabs(rnnlm->cuedversion - 1.1) < 1e-6)
            {
                logcurp = CUEDRNNLMAcceptWord_v1_0 (rnnlm, lastword, curword);
            }
            else if (rnnlm->cuedversion == 1.0)
            {
                logcurp = CUEDRNNLMAcceptWord_v1_0 (rnnlm, lastword, curword);
            }
            else
            {
                logcurp = CUEDRNNLMAcceptWord (rnnlm, lastword, curword);
            }
        }
        else if (rnnlm->usefrnnlm)
        {
            logcurp = FRNNLMAcceptWord (rnnlm, lastword, curword);
        }
        else
        {
            logcurp=RNNLMAcceptWord(rnnlm, lastword, curword);
        }
        logp+=logcurp;

        lastword=curword;
        if ( rnnlm->independent && curword == 0 )
            RNNLMEnd(rnnlm);
        if ( curword == 0 )
            sentcn ++ ;

        if (curword != -1 )
            wordcn++;

        if ( trace & T_PRO )
        {
            if ( curword == -1 ) {
                printf("-1\t0\t\tOOV\n");
            }
            else
            {
                int classid =rnnlm->vocab[curword].class_index ;
                float f = rnnlm->neu2[curword].ac*rnnlm->neu2[classid+rnnlm->vocab_size].ac;
                printf("%d\t%.10f\t%s\n",curword, f , rnnlm->vocab[curword].word);
            }
        }
        if ( feof(fp) )
            break;
    }
    RNNLMEnd(rnnlm);

    printf("\nPPL net: %f\n", exp(-logp/(real)wordcn*M_LN10 ) );
    printf("\nTotal %d words in %d sentences\n", wordcn, sentcn) ;



}


/*  EXPORT -> RNNLMInVocabMap  */
int RNNLMInVocabMap(RNNLM* rnnlm, char* name)
{
    int* id=NULL;

    if ( !(id=FindInHashTable(&rnnlm->in_vocab, name )) ){
        return rnnlm->in_oos_nodeid;      /* oos return  */
    }

    return *id;

}
/*  EXPORT -> RNNLMOutVocabMap  */
int RNNLMOutVocabMap(RNNLM* rnnlm, char* name)
{
    int* id=NULL;

    if ( !(id=FindInHashTable(&rnnlm->out_vocab, name )) ){
        return rnnlm->out_oos_nodeid;      /* oos return  */
    }
    return *id;
}

/* Read topic matrix file */
void ReadTopicMatrixFile (RNNLM *rnnlm)
{
   int i, j, t;
   int dim_topic;
   int n_topic;
   float value;
   FILE *fin_topic = fopen (rnnlm->topicmatrixfile, "rb");
   if (fin_topic == NULL)
      {
         printf ("ERROR: Failed to open topic matrix file: %s\n", rnnlm->topicmatrixfile);
         exit (1);
      }
   fscanf (fin_topic, "%d %d", &n_topic, &dim_topic);
   rnnlm->num_topic = n_topic;
   if (dim_topic != rnnlm->layer0_topic_size)
      {
         printf ("the topic feature size from model file(%d) and matrix file(%d) is different\n", rnnlm->layer0_topic_size, dim_topic);
         exit (0);
      }
   rnnlm->topicmatrix = (float *)calloc(dim_topic*n_topic, sizeof(float));
   printf ("%d topic models (with %d dimensions) will be read from %s\n", n_topic, dim_topic, rnnlm->topicmatrixfile);
   i = 0;
   while (i < n_topic)
      {
         fscanf (fin_topic, "%d", &j);
         assert (j == i);
         for (t=0; t<dim_topic; t++)
            {
               fscanf (fin_topic, "%f", &value);
               rnnlm->topicmatrix[t+dim_topic*i] = value;
            }
         i ++;
      }
   fclose (fin_topic);
   printf ("%d topic models (with %d dimensions) is read from %s successfully\n", n_topic, dim_topic, rnnlm->topicmatrixfile);
}

int GetRNNLMHiddenVectorSize(RNNLM* lm)
{
    int totalhiddensize = 0;
    int a;
    if (lm->usecuedrnnlm)
    {
        if (fabs(lm->cuedversion - 1.1) < 1e-6)
        {
            for (a = 0; a<lm->num_layer; a++)
            {
                totalhiddensize += GetHiddenSize (&(lm->_layers[a]));
            }
            return totalhiddensize;
        }
        else if (lm->cuedversion == 1.0)
        {
            for (a = 0; a<lm->num_layer; a++)
            {
                totalhiddensize += GetHiddenSize (&(lm->_layers[a]));
            }
            return totalhiddensize;
        }
        else
        {
            return lm->layersizes[lm->num_layer-1];
        }
    }
    return lm->layer1_size;
}

int GetRNNLMHistorySize(RNNLM* lm)     /*  history is used in RNNLM for direct connection */
{
    return lm->direct_order;
}

/*
void GetRNNLMHistVector (RNNLM* lm, Vector v)
{
    int i =0 ;
    int size= VectorSize(v);
    int bias = lm->layer0_size - lm->layer1_size;

    if (lm->usecuedrnnlm)
    {
        if (lm->cuedversion == 1.0)
        {
            GetHiddenAc (&(lm->_layers[lm->num_layer-1]), v);
        }
        else
        {
        }
        for ( i = 1; i<= size; i++ )
            v[i] = lm->neu0_ac_hist[i-1];
    }
    else
    {
        for ( i = 1; i<= size; i++ )
            v[i]=lm->neu0[bias + i-1].ac;
    }
}
*/

/*
void AssignRNNLMHistVector (RNNLM *lm, Vector v)
{
    int i = 0;
    int size = VectorSize(v);
    int bias = lm->layer0_size - lm->layer1_size;
    if (lm->usecuedrnnlm)
    {
        for (i = 1; i <= size; i ++)
            lm->neu0_ac_hist[i-1] = v[i];
    }
    else
    {
        for (i = 1; i <= size; i ++)
            lm->neu0[bias + i-1].ac = v[i];
    }
}
*/

void GetRNNLMHiddenVector(RNNLM* lm, Vector v)
/*  before call this function v is a vector allocated with layer1_size
 *  dimension */
{
    int i =0 ;
    int size= VectorSize(v);

    if (lm->usecuedrnnlm)
    {
        if (fabs(lm->cuedversion - 1.1) < 1e-6)
        {
            GetRNNLMHiddenVector_v1_0 (lm, v);
        }
        else if (lm->cuedversion == 1.0)
        {
            GetRNNLMHiddenVector_v1_0 (lm, v);
        }
        else
        {
            for ( i = 1; i<= size; i++ )
                v[i] = lm->neu_ac[lm->num_layer-1][i-1];
        }
    }
    else
    {
        for ( i = 1; i<= size; i++ )
            v[i]=lm->neu1[i-1].ac;
    }
}

void AssignRNNLMHiddenVector (RNNLM *lm, Vector v)
{
    int i = 0;
    int size = VectorSize(v);
    if (lm->usecuedrnnlm &&  fabs(lm->cuedversion - 1.1) < 1e-6)
    {
        AssignRNNLMHistVector_v1_0 (lm, v);
    }
    else if (lm->usecuedrnnlm && lm->cuedversion == 1.0)
    {
        AssignRNNLMHistVector_v1_0 (lm, v);
    }
    else
    {
        for (i = 1; i <= size; i ++)
        {
            if (lm->usecuedrnnlm)
            {
                lm->neu_ac[lm->num_layer-1][i-1] = v[i];
            }
            else
            {
                lm->neu1[i-1].ac = v[i];
            }
        }
    }
}


void GetRNNLMOutputVector (RNNLM* lm, Vector v)
{
    int i = 0;
    int size = VectorSize (v);
    for (i = 1; i<= size; i++)
        v[i] = lm->outP_arr[i-1];
}

double CalcHistsDistance(Vector v1, Vector v2)
{
    int i, size= 0;
    double d = 0.0, dd = 0.0;

    size = VectorSize(v1);
    if (size != VectorSize(v2))
       HError(5270,"CalcHistsDistance: sizes differ %d vs %d",
              size,VectorSize(v2));

    for ( i = 1, size=VectorSize(v1) ; i<=size; i++){
        d= v1[i]-v2[i];
        dd+= d*d;
    }

#if 0
    fprintf(stdout, "HV Dist = %f dd = %f size = %d \n", dd/size, dd, size);
    fflush(stdout);
#endif

    return dd/size;
}

float CalcProbDistance(Vector v1, Vector v2)
{
    int i, size= 0;
    float d, dd = 0;

    size=VectorSize(v1);
    for ( i = 1; i<=size; i++){
        if ( v1[i] > 1e-6 && v2[i] > 1e-6 ) {
            d=v1[i]*log(v1[i]/v2[i]);
            d+=v2[i]*log(v2[i]/v1[i]);
            dd+=d/2;
        }
    }
    return dd/size;
}

static Vector hists, mean, pmean, pvar, var;
static int nsample = 0;
static int printnum = 20;

void UpdateMeanVariance (RNNLM* lm)
{
    int j = 1;
    GetRNNLMHiddenVector (lm, hists);
    for (j=1; j<=lm->layer1_size; j++)
    {
        mean[j] = (nsample * pmean[j] + hists[j]) / (nsample+1);
        var[j] = (nsample*pvar[j] + (hists[j]-pmean[j])*(hists[j]+pmean[j]-2*mean[j]))/(nsample+1) + (mean[j] - pmean[j]) * (mean[j] - pmean[j]);
    }
    nsample ++;
    memcpy (pmean, mean, sizeof(float)*(lm->layer1_size+1));
    memcpy (pvar, var, sizeof(float)*(lm->layer1_size+1));
    if (nsample % 10 == 0)
    {
        printf ("After %d samples are collected, mean and variance is:\n", nsample);
        printf ("Mean Vector[1..%d]\n", printnum);
        for (j=1; j<=printnum; j++)      printf ("%f ", mean[j]);
        printf ("\nVariance Vector[1..%d]\n", printnum);
        for (j=1; j<=printnum; j++)      printf ("%f ", var[j]);
        printf ("\n");
    }
}

float GetVariance(int i)
{
    return var[i];
}

float GetMean (int i)
{
    return mean[i];
}

void printMean(int ndim)
{
    int i = 0;
    printf ("Mean:\n");
    for (i = 1; i <= ndim; i ++)
    {
        if (i % 50 == 0)
        {
            printf ("\n");
        }
        printf ("%f ", mean[i]);
    }
    printf ("\n");
}

void printVariance (int ndim)
{
    int i;
    printf ("Variance: \n");
    for (i = 1; i <= ndim; i ++)
    {
        if (i % 50 == 0)
        {
            printf ("\n");
        }
        printf ("%f ", var[i]);
    }
    printf ("\n");
}


void CalOutputMatrix (Matrix outputmatrix, struct synapse *syn, int nrows, int scol, int ecol)
{
    int i, j, k;
    for (i = 1; i <= nrows; i ++)
    {
        for (j = 1; j <= nrows; j ++)
        {
            if (j < i)   outputmatrix[i][j] = outputmatrix[j][i];
            else
            {
                outputmatrix[i][j] = 0.0;
                for (k = scol; k < ecol; k ++)
                    outputmatrix[i][j]+=syn[nrows*k+ i-1].weight*syn[nrows*k+ j-1].weight;
            }
        }
    }
}

void InitMeanVariance (int ndim)
{
    hists = CreateVector (&gstack, ndim+1);
    mean = CreateVector (&gstack, ndim+1);
    pmean = CreateVector (&gstack, ndim+1);
    var = CreateVector (&gstack, ndim+1);
    pvar = CreateVector (&gstack, ndim+1);
    memset (mean,  0, sizeof(float) * (ndim+1));
    memset (pmean, 0, sizeof(float) * (ndim+1));
    memset (var,   0, sizeof(float) *(ndim+1));
    memset (pvar,  0, sizeof(float) * (ndim+1));
}
void FreeMeanVariance ()
{
    FreeVector(&gstack, mean);
    FreeVector(&gstack, pmean);
    FreeVector(&gstack, var);
    FreeVector(&gstack, pvar);
    FreeVector(&gstack, hists);
}

void GetRNNLMHistory(RNNLM* lm, IntVec v)
/*  before call this function v is a IntVec allocated with direct_size
 *  dimension
 *  v[1] is the most recent word!!! */
{
    int i =0 ;
    int direct_order=lm->direct_order;
    for ( i = 1 ; i<= direct_order ;  i++)
        v[i]= lm->history[i-1];

}

void CloneRNNLM(RNNLM* src, RNNLM* tgt)
/*  clone src rnnlm to target rnnlm
 *   -- copy src hidden vector to target
 *   -- copy history from src to target
 *   before calling this function, tgt must have been created
 *   tgt=CreateRNNLM(MemHeap) */
{
    IntVec hist;
    Vector hidd;
    int i,j =0;

    hist=CreateIntVec(&gstack, src->direct_order);
    hidd=CreateVector(&gstack, src->layer1_size);
    GetRNNLMHistory(src, hist);
    GetRNNLMHiddenVector(src, hidd);


    /*  TODO : some of variables/structure are redundant, need
     *  to be eliminated to save space if many instance of RNNLM is created */
    {
       tgt->version = src->version ;
       tgt->filetype = src->filetype ;
       tgt->layer0_size = src->layer0_size;
       tgt->layer1_size = src->layer1_size;
       tgt->layerc_size = src->layerc_size;
       tgt->layer2_size = src->layer2_size;
       tgt->direct_size = src->direct_size;
       tgt->direct_order= src->direct_order;
       tgt->bptt        = src->bptt;
       tgt->bptt_block  = src->bptt_block;
       tgt->vocab_size  = src->vocab_size;
       tgt->class_size  = src->class_size;
       tgt->old_classes = src->old_classes;
       tgt->independent = src->independent;
       tgt->vocab_size  = src->vocab_size;
       tgt->vocab_max_size = src->vocab_max_size;
       tgt->vocab_hash_size= src->vocab_hash_size;
       tgt->vocab=(struct vocab_word *)New(tgt->x, sizeof(struct vocab_word)* tgt->vocab_max_size );
       for (i=0; i<tgt->vocab_size ; i++) {
            strcpy(tgt->vocab[i].word,  src->vocab[i].word);
            tgt->vocab[i].cn = src->vocab[i].cn;
            tgt->vocab[i].prob = src->vocab[i].prob;
            tgt->vocab[i].class_index=src->vocab[i].class_index;
       }
       initNet(tgt);
       /*  copy weight 0->1 */
       for (j=0; j<src->layer1_size; j++) {
           for (i=0; i<src->layer0_size; i++) {
               tgt->syn0[i+j*src->layer0_size].weight=
                 src->syn0[i+j*src->layer0_size].weight;
           }
       }
       /*  copy weight 1->2 */
       if ( src->layerc_size ==  0 ) {
           for (j=0; j<src->layer2_size; j++) {
               for (i=0; i<src->layer1_size; i++) {
                   tgt->syn1[i+j*src->layer1_size].weight=
                       src->syn1[i+j*src->layer1_size].weight;
               }
           }
       }
       else
       {
           for (j=0; j<src->layerc_size; j++) {
               for (i=0; i<src->layer1_size; i++) {
                   tgt->syn1[i+j*src->layer1_size].weight=
                       src->syn1[i+j*src->layer1_size].weight;
               }
           }
           for (j=0; j<src->layer2_size; j++) {
               for (i=0; i<src->layerc_size; i++) {
                   tgt->sync[i+j*src->layerc_size].weight=
                       src->sync[i+j*src->layerc_size].weight;
               }
           }

       }


       /*  direct connection  */
       for ( i = 0 ; i<src->direct_size; i++)
       {
           tgt->syn_d[i]=src->syn_d[i];
       }

       saveWeights(tgt);
    }



    for (i=1; i<=src->direct_order ; i++)
        tgt->history[i-1] = hist[i];

    for (i=1; i<=src->layer1_size; i++)
        tgt->neu1[i-1].ac=hidd[i];

    copyHiddenLayerToInput(tgt);

    Dispose(&gstack, hidd);
    Dispose(&gstack, hist);
}




/* ----------------------- A generic hashtable  --------------- */

static unsigned int HashValue(HashTable* t, char* name)
{
   unsigned int hashval;

   for (hashval=0; *name != '\0'; name++)
      hashval = *name + 31*hashval;
   return hashval%(t->hashsize);
}
void InitalHashTable(unsigned int tablesize, unsigned int objsize, HashTable* t)
{
    int i =0 ;
    t->hashsize = tablesize;
    t->objectsize  = objsize;

    t->table=(HashSlot**)New(&rnnInfoStack, sizeof( HashSlot*)*t->hashsize);
    for (i =0 ; i<t->hashsize; i++)
        t->table[i]=NULL;
}
void InsertToHashTable(HashTable* t, char* key, void* obj)
{
    HashSlot* p = NULL;
    HashSlot* prep=NULL;
    unsigned int hashval=HashValue(t,key);
    if ( (p=t->table[hashval]) )
    {
        while (p)
        {
            if ( strcmp(p->key, key)==0  )
            {
                return ;        /*  do nothing, do not insert */
            }
            prep=p;
            p=p->next;
        }
        p=(HashSlot*) New(&rnnInfoStack, sizeof(HashSlot));
        p->next=NULL;
        p->key=CopyString(&rnnInfoStack, key);
        p->object=(void*)New(&rnnInfoStack, t->objectsize);
        memcpy(p->object, obj, t->objectsize);
        prep->next=p;
    }
    else
    {
        p= t->table[hashval] = (HashSlot*)New(&rnnInfoStack, sizeof(HashSlot));
        p->next=NULL;
        p->key=CopyString(&rnnInfoStack, key  );
        p->object = (void*)New(&rnnInfoStack, t->objectsize);
        memcpy(p->object, obj, t->objectsize );
    }
}
void* FindInHashTable(HashTable* t, char* key)
{
    unsigned int hashval=HashValue(t,key);
    HashSlot* p = t->table[hashval];

    while (p)
    {
        if (strcmp(key, p->key )==0)
        {
            return p->object;
        }
        else
            p=p->next;
    }
    return p;
}


/* for cuedrnnlm v1.0 */
void ReadFile (float *U, FILE *fptr, int nrows, int ncols)
{
    int i, j;
    float v;
    for (i=0; i<nrows; i++)
    {
        for (j=0; j<ncols; j++)
        {
            fscanf(fptr, "%f ", &v);
            U[i+j*nrows] = v;
        }
        fscanf (fptr, "\n");
    }
}

void sigmoidij (float *ac, int num)
{
    int i;
    for (i=0; i<num; i++)
    {
        ac[i] = 1.0/(1+exp(-ac[i]));
    }
}

void reluij (float *ac, float ratio, int num)
{
    int i;
    for (i=0; i<num; i++)
    {
        if (ac[i] > 0)
        {
            ac[i] *= ratio;
        }
        else
        {
            ac[i] = 0;
        }
    }
}

void tanhij (float *ac, int num)
{
    int i;
    for (i=0; i<num; i++)
    {
        if (ac[i] > 50) ac[i] = 50;
        ac[i] = (exp(2*ac[i])-1)/(exp(2*ac[i])+1);
    }
}

void dotmultiply (float *arr1, float *arr2, int num)
{
    int i;
    for (i=0; i<num; i++)
    {
        arr1[i] *= arr2[i];
    }
}

void calgruop (float *h_ac, float *h_, float *z, int num)
{
    int i;
    for (i=0; i<num; i++)
    {
        h_ac[i] = h_ac[i]*(1-z[i]) + h_[i]*z[i];
    }
}

void adddotmultiply (float *arr1, float *arr2, float *wgt, int num)
{
    int i;
    for (i=0; i<num; i++)
    {
        arr1[i] += arr2[i]*wgt[i];
    }
}

void add (float *arr1, float *arr2, int num)
{
    int i;
    for (i=0; i<num; i++)
    {
        arr1[i] += arr2[i];
    }
}

void softmaxij_sm (float *ac, int num, float smfactor)
{
    int a, maxi;
    float v, norm, maxv = 1e-8;
    maxv = -1e10;
    for (a=0; a<num; a++)
    {
        ac[a] = ac[a]*smfactor;
    }
    for (a=0; a<num; a++)
    {
        v = ac[a];
        if (v > maxv)
        {
            maxv = v;
            maxi = a;
        }
    }
    norm = 0;
    for (a=0; a<num; a++)
    {
        v = ac[a] - maxv;
        ac[a] = exp(v);
        norm += ac[a];
    }
    for (a=0; a<num; a++)
    {
        v = ac[a] / norm;
        ac[a] = v;
    }
    printf("SOFTMAX norm: %f\n",norm*exp(-maxv));
}

void softmaxij (float *ac, int num)
{
    int a, maxi;
    float v, norm, maxv = 1e-8;
    maxv = -1e10;
    for (a=0; a<num; a++)
    {
        v = ac[a];
        if (v > maxv)
        {
            maxv = v;
            maxi = a;
        }
    }
    norm = 0;
    for (a=0; a<num; a++)
    {
        v = ac[a] - maxv;
        ac[a] = exp(v);
        norm += ac[a];
    }
    for (a=0; a<num; a++)
    {
        v = ac[a] / norm;
        ac[a] = v;
    }
}

void forward_nosigm (Layer *self, int prevword, int curword, float *neu0_ac, float *neu1_ac)
{
    int i, j;
    float v;
    int ncols = self->ncols;
    int nrows = self->nrows;
    if (strcmp(self->type, "feedforward") == 0)
    {
        matrixXvector_v2 (neu0_ac, self->U, neu1_ac, nrows, ncols);
    }
    else
    {
        printf ("Error: it is a %s layer, should be feedforward layer!\n", self->type);
        exit (0);
    }
}

void forward (Layer *self, int prevword, int curword, float *neu0_ac, float *neu1_ac)
{
    int i, j;
    float v;
    int ncols = self->ncols;
    int nrows = self->nrows;
    memset (neu1_ac, 0, sizeof(float)*ncols);
    if (strcmp(self->type, "input") == 0)
    {
        for (i=0; i<ncols; i++)
        {
            neu1_ac[i] = self->U[prevword+i*nrows];
        }
        if (self->dim_fea > 0)
        {
            matrixXvector_v2 (self->ac_fea, self->U_fea, neu1_ac, self->dim_fea, ncols);
        }
    }
    else if (strcmp(self->type, "feedforward") == 0)
    {
        memset (neu1_ac, 0, ncols*sizeof(float));
        matrixXvector_v2 (neu0_ac, self->U, neu1_ac, nrows, ncols);
        if (self->nodetype == 0)
        {
            sigmoidij (neu1_ac, ncols);
        }
        else if (self->nodetype == 1)
        {
            reluij (neu1_ac, self->reluratio, ncols);
        }
    }
    else if (strcmp(self->type, "linear") == 0)
    {
        memset (neu1_ac, 0, ncols*sizeof(float));
        matrixXvector_v2 (neu0_ac, self->U, neu1_ac, nrows, ncols);
    }
    else if (strcmp(self->type, "recurrent") == 0)
    {
        memset (neu1_ac, 0, ncols*sizeof(float));
        matrixXvector_v2 (self->h_ac, self->W, neu1_ac, ncols, ncols);
        matrixXvector_v2 (neu0_ac, self->U, neu1_ac, nrows, ncols);
        if (self->nodetype == 0)
        {
            sigmoidij (neu1_ac, ncols);
        }
        else if (self->nodetype == 1)
        {
            reluij (neu1_ac, self->reluratio, ncols);
        }
        memcpy (self->h_ac, neu1_ac, ncols*sizeof(float));
    }
    else if (strcmp(self->type, "gru") == 0)
    {
        memset (self->z, 0, ncols*sizeof(float));
        matrixXvector_v2 (neu0_ac, self->Uz, self->z, nrows, ncols);
        matrixXvector_v2 (self->h_ac, self->Wz, self->z, ncols, ncols);
        sigmoidij (self->z, ncols);

        memset (self->r, 0, ncols*sizeof(float));
        matrixXvector_v2 (neu0_ac, self->Ur, self->r, nrows, ncols);
        matrixXvector_v2 (self->h_ac, self->Wr, self->r, ncols, ncols);
        sigmoidij (self->r, ncols);
        dotmultiply (self->r, self->h_ac, ncols);
	
        memset (self->h_, 0, ncols*sizeof(float));
        matrixXvector_v2 (neu0_ac, self->Uh, self->h_, nrows, ncols);
        matrixXvector_v2 (self->r, self->Wh, self->h_, ncols, ncols);
        tanhij (self->h_, ncols);

        memset (neu1_ac, 0, ncols*sizeof(float));
        calgruop (self->h_ac, self->h_, self->z, ncols);
        memcpy (neu1_ac, self->h_ac, ncols*sizeof(float));
    }
    else if (strcmp(self->type, "lstm") == 0)
    {
        memset (self->i, 0, ncols*sizeof(float));
        matrixXvector_v2 (neu0_ac, self->Ui, self->i, nrows, ncols);
        matrixXvector_v2 (self->h_ac, self->Wi, self->i, ncols, ncols);
        adddotmultiply (self->i, self->c, self->Pi, ncols);
        sigmoidij (self->i, ncols);

        memset (self->f, 0, ncols*sizeof(float));
        matrixXvector_v2 (neu0_ac, self->Uf, self->f, nrows, ncols);
        matrixXvector_v2 (self->h_ac, self->Wf, self->f, ncols, ncols);
        adddotmultiply (self->f, self->c, self->Pf, ncols);
        sigmoidij (self->f, ncols);
        dotmultiply (self->f, self->c, ncols);

        memset (self->z, 0, ncols*sizeof(float));
        matrixXvector_v2 (neu0_ac, self->Uz, self->z, nrows, ncols);
        matrixXvector_v2 (self->h_ac, self->Wz, self->z, ncols, ncols);
        tanhij (self->z, ncols);
        dotmultiply (self->z, self->i, ncols);

        memcpy (self->newc, self->z, ncols*sizeof(float));
        add (self->newc, self->f, ncols);

        memcpy (self->c, self->newc, ncols*sizeof(float));

        memset (self->o, 0, ncols*sizeof(float));
        matrixXvector_v2 (neu0_ac, self->Uo, self->o, nrows, ncols);
        matrixXvector_v2 (self->h_ac, self->Wo, self->o, ncols, ncols);
        adddotmultiply (self->o, self->newc, self->Po, ncols);
        sigmoidij (self->o, ncols);

        memcpy (self->h_ac, self->newc, ncols*sizeof(float));
        tanhij (self->h_ac, ncols);
        dotmultiply (self->h_ac, self->o, ncols);
        memcpy (neu1_ac, self->h_ac, ncols*sizeof(float));

    }
    else if (strcmp(self->type, "output") == 0)
    {
        memset (neu1_ac, 0, ncols*sizeof(float));
        matrixXvector_v2 (neu0_ac, self->U, neu1_ac, nrows, ncols);
        softmaxij_sm (neu1_ac, ncols, self->smfactor);
    }
    else
    {
        printf ("ERROR: unrecognized layer type in model file: %s\n", self->type);
        exit (0);
    }
}



void Read (Layer *self, FILE *fptr)
{
    int i, j;
    float v;
    int nr = self->nrows;
    int nc = self->ncols;
    if (strcmp(self->type, "input") == 0)
    {
        ReadFile (self->U, fptr, nr, nc);
        if (self->dim_fea > 0)
        {
            ReadFile (self->U_fea, fptr, self->dim_fea, nc);
        }
    }
    else if (strcmp(self->type, "feedforward") == 0)
    {
        fscanf (fptr, "nodetype: %d\n", &self->nodetype);
        fscanf (fptr, "reluratio: %f\n", &self->reluratio);
        ReadFile (self->U, fptr, nr, nc);
    }
    else if (strcmp(self->type, "linear") == 0)
    {
        ReadFile (self->U, fptr, nr, nc);
    }
    else if (strcmp(self->type, "recurrent") == 0)
    {
        fscanf (fptr, "nodetype: %d\n", &self->nodetype);
        fscanf (fptr, "reluratio: %f\n", &self->reluratio);
        ReadFile (self->U, fptr, nr, nc);
        ReadFile (self->W, fptr, nc, nc);
    }
    else if (strcmp(self->type, "gru") == 0)
    {
        ReadFile (self->Wh, fptr, nc, nc);
        ReadFile (self->Uh, fptr, nr, nc);
        ReadFile (self->Wr, fptr, nc, nc);
        ReadFile (self->Ur, fptr, nr, nc);
        ReadFile (self->Wz, fptr, nc, nc);
        ReadFile (self->Uz, fptr, nr, nc);
    }
    else if (strcmp(self->type, "lstm") == 0)
    {
        ReadFile (self->Wz, fptr, nc, nc);
        ReadFile (self->Uz, fptr, nr, nc);
        ReadFile (self->Wi, fptr, nc, nc);
        ReadFile (self->Ui, fptr, nr, nc);
        ReadFile (self->Wf, fptr, nc, nc);
        ReadFile (self->Uf, fptr, nr, nc);
        ReadFile (self->Wo, fptr, nc, nc);
        ReadFile (self->Uo, fptr, nr, nc);
        ReadFile (self->Pi, fptr, 1, nc);
        ReadFile (self->Pf, fptr, 1, nc);
        ReadFile (self->Po, fptr, 1, nc);
    }
    else if (strcmp(self->type, "output") == 0)
    {
        ReadFile (self->U, fptr, nr, nc);
    }
    else
    {
        printf ("ERROR: unrecognized layer type in model file: %s\n", self->type);
        exit (0);
    }
}


/* BEGIN modification */
void GetRNNLMHiddenVector_v1_0 (RNNLM *lm, Vector v)
{
    int i,l;
    int pos = 1;
    for (l=0; l<lm->num_layer; l++)
    {
        Layer *self = &(lm->_layers[l]);
        if (strcmp(self->type, "recurrent") == 0)
        {
            for (i=0; i< self->ncols; i++)
            {
                v[pos++] = self->h_ac[i];
            }
        }
        else if (strcmp (self->type, "gru") == 0)
        {
            for (i=0; i< self->ncols; i++)
            {
                v[pos++] = self->h_ac[i];
            }
        }
        else if (strcmp (self->type, "lstm") == 0)
        {
            for (i=0; i< self->ncols; i++)
            {
                v[pos++] = self->h_ac[i];
            }
            for (i=0; i< self->ncols; i++)
            {
                v[pos++] = self->c[i];
            }
        }
    }
}



void AssignRNNLMHistVector_v1_0 (RNNLM *lm, Vector v)
{
    int i,l;
    int pos = 1;
    for (l=0; l<lm->num_layer; l++)
    {
        Layer *self = &(lm->_layers[l]);
        if (strcmp(self->type, "recurrent") == 0)
        {
            for (i=0; i<self->ncols; i++)
            {
                self->h_ac[i] = v[pos++];
            }
        }
        else if (strcmp (self->type, "gru") == 0)
        {
            for (i=0; i<self->ncols; i++)
            {
                self->h_ac[i] = v[pos++];
            }
        }
        else if (strcmp (self->type, "lstm") == 0)
        {
            for (i=0; i<self->ncols; i++)
            {
                self->h_ac[i] = v[pos++];
            }
            for (i=0; i<self->ncols; i++)
            {
                self->c[i] = v[pos++];
            }
        }
    }
}
/* END modification */

void AssignHiddenAc (Layer *self, Vector v)
{
    int i;
    int size = VectorSize (v);
    if (strcmp(self->type, "recurrent") == 0)
    {
        for (i=1; i<= size; i++)
        {
            self->h_ac[i-1] = v[i];
        }
    }
    else if (strcmp(self->type, "gru") == 0)
    {
        for (i=1; i<= size; i++)
        {
            self->h_ac[i-1] = v[i];
        }
    }
    else if (strcmp(self->type, "lstm") == 0)
    {
        for (i=1; i<= size; i++)
        {
            self->h_ac[i-1] = v[i];
        }
    }
}

void GetHiddenAc (Layer *self, Vector v)
{
    int i;
    int size = VectorSize (v);
    if (strcmp(self->type, "recurrent") == 0)
    {
        for (i=1; i<= size; i++)
        {
            v[i] = self->h_ac[i-1];
        }
    }
    else if (strcmp(self->type, "gru") == 0)
    {
        for (i=1; i<= size; i++)
        {
            v[i] = self->h_ac[i-1];
        }
    }
    else if (strcmp(self->type, "lstm") == 0)
    {
        for (i=1; i<= size; i++)
        {
            v[i] = self->h_ac[i-1];
        }
    }
}

void CopyHiddenAc (Layer *self)
{
    return;
}

void Reset (Layer *self)
{
    int i;
    if (strcmp(self->type, "recurrent") == 0)
    {
        for (i=0; i<self->ncols; i++)
        {
            self->h_ac[i] = 0.1;
        }
    }
    else if (strcmp(self->type, "gru") == 0)
    {
        for (i=0; i<self->ncols; i++)
        {
            self->h_ac[i] = 0.1;
        }
    }
    else if (strcmp(self->type, "lstm") == 0)
    {
        for (i=0; i<self->ncols; i++)
        {
            self->h_ac[i] = 0.1;
            self->c[i]    = 0.1;
        }
    }
}

int GetHiddenSize (Layer *self)
{

    if (strcmp(self->type, "recurrent") == 0)
    {
        return self->ncols;
    }
    else if (strcmp(self->type, "gru") == 0)
    {
        return self->ncols;
    }
    else if (strcmp(self->type, "lstm") == 0)
    {
        return (self->ncols) * 2;
    }
    return 0;
}

void Init(RNNLM *rnnlmodel, Layer *self, char *str, int nr, int nc)
{
    strcpy (self->type, str);
    self->nrows = nr;
    self->ncols = nc;
    self->sizes = nr*nc;
#if 1
    self->U = (float *)New (rnnlmodel->x, nr*nc*sizeof(float));
#else
    self->U = (float *)malloc (nr*nc*sizeof(float));
#endif
    if (strcmp(self->type, "input") == 0)
    {
        if (self->dim_fea > 0)
        {
            self->U_fea = (float *) New (rnnlmodel->x, self->dim_fea*nc*sizeof(float));
        }
        else
        {
            self->U_fea = NULL;
        }
        return;
    }
    else if (strcmp(self->type, "feedforward") == 0)
    {
        return;
    }
    else if (strcmp(self->type, "linear") == 0)
    {
        return;
    }
    else if (strcmp(self->type, "recurrent") == 0)
    {
#if 1
        self->W = (float *)New(rnnlmodel->x, nc*nc*sizeof(float));
        self->h_ac = (float *)New(rnnlmodel->x, nc*sizeof(float));
#else
        self->W = (float *)malloc(nr*nc*sizeof(float));
        self->h_ac = (float *)malloc(nc*sizeof(float));
#endif
    }
    else if (strcmp(self->type, "gru") == 0)
    {
#if 1
        self->Uh = (float *)New(rnnlmodel->x, nr*nc*sizeof(float));
        self->Ur = (float *)New(rnnlmodel->x, nr*nc*sizeof(float));
        self->Uz = (float *)New(rnnlmodel->x, nr*nc*sizeof(float));

        self->Wh = (float *)New(rnnlmodel->x, nc*nc*sizeof(float));
        self->Wr = (float *)New(rnnlmodel->x, nc*nc*sizeof(float));
        self->Wz = (float *)New(rnnlmodel->x, nc*nc*sizeof(float));

        self->h_ac = (float *)New(rnnlmodel->x, nc*sizeof(float));

        self->r = (float *)New(rnnlmodel->x, nc*sizeof(float));
        self->z = (float *)New(rnnlmodel->x, nc*sizeof(float));
        self->h_= (float *)New(rnnlmodel->x, nc*sizeof(float));
        self->c = (float *)New(rnnlmodel->x, nc*sizeof(float));
#else
        self->Uh = (float *)malloc(nr*nc*sizeof(float));
        self->Ur = (float *)malloc(nr*nc*sizeof(float));
        self->Uz = (float *)malloc(nr*nc*sizeof(float));

        self->Wh = (float *)malloc(nc*nc*sizeof(float));
        self->Wr = (float *)malloc(nc*nc*sizeof(float));
        self->Wz = (float *)malloc(nc*nc*sizeof(float));

        self->h_ac = (float *)malloc(nc*sizeof(float));

        self->r = (float *)malloc(nc*sizeof(float));
        self->z = (float *)malloc(nc*sizeof(float));
        self->h_= (float *)malloc(nc*sizeof(float));
        self->c = (float *)malloc(nc*sizeof(float));
#endif
    }
    else if (strcmp(self->type, "lstm") == 0)
    {
#if 1
        self->Uz = (float *)New(rnnlmodel->x, nr*nc*sizeof(float));
        self->Uf = (float *)New(rnnlmodel->x, nr*nc*sizeof(float));
        self->Ui = (float *)New(rnnlmodel->x, nr*nc*sizeof(float));
        self->Uo = (float *)New(rnnlmodel->x, nr*nc*sizeof(float));

        self->Wz = (float *)New(rnnlmodel->x, nc*nc*sizeof(float));
        self->Wi = (float *)New(rnnlmodel->x, nc*nc*sizeof(float));
        self->Wf = (float *)New(rnnlmodel->x, nc*nc*sizeof(float));
        self->Wo = (float *)New(rnnlmodel->x, nc*nc*sizeof(float));

        self->Pi = (float *)New(rnnlmodel->x, nc*sizeof(float));
        self->Pf = (float *)New(rnnlmodel->x, nc*sizeof(float));
        self->Po = (float *)New(rnnlmodel->x, nc*sizeof(float));

        self->newc=(float *)New(rnnlmodel->x, nc*sizeof(float));
        self->zi = (float *)New(rnnlmodel->x, nc*sizeof(float));
        self->fc = (float *)New(rnnlmodel->x, nc*sizeof(float));

        self->c  = (float *)New(rnnlmodel->x, nc*sizeof(float));
        self->z  = (float *)New(rnnlmodel->x, nc*sizeof(float));
        self->i  = (float *)New(rnnlmodel->x, nc*sizeof(float));
        self->f  = (float *)New(rnnlmodel->x, nc*sizeof(float));
        self->o  = (float *)New(rnnlmodel->x, nc*sizeof(float));
        self->h_ac = (float *)New(rnnlmodel->x, nc*sizeof(float));
#else
        self->Uz = (float *)malloc(nr*nc*sizeof(float));
        self->Uf = (float *)malloc(nr*nc*sizeof(float));
        self->Ui = (float *)malloc(nr*nc*sizeof(float));
        self->Uo = (float *)malloc(nr*nc*sizeof(float));

        self->Wz = (float *)malloc(nc*nc*sizeof(float));
        self->Wi = (float *)malloc(nc*nc*sizeof(float));
        self->Wf = (float *)malloc(nc*nc*sizeof(float));
        self->Wo = (float *)malloc(nc*nc*sizeof(float));

        self->Pi = (float *)malloc(nc*sizeof(float));
        self->Pf = (float *)malloc(nc*sizeof(float));
        self->Po = (float *)malloc(nc*sizeof(float));

        self->newc=(float *)malloc(nc*sizeof(float));
        self->zi = (float *)malloc(nc*sizeof(float));
        self->fc = (float *)malloc(nc*sizeof(float));

        self->c  = (float *)malloc(nc*sizeof(float));
        self->z  = (float *)malloc(nc*sizeof(float));
        self->i  = (float *)malloc(nc*sizeof(float));
        self->f  = (float *)malloc(nc*sizeof(float));
        self->o  = (float *)malloc(nc*sizeof(float));
        self->h_ac = (float *)malloc(nc*sizeof(float));
#endif
    }
    else if (strcmp(self->type, "output") == 0)
    {
        return;
    }
    else
    {
        printf ("ERROR: unrecognized layer type in model file: %s\n", self->type);
        exit (0);
    }
}






/* ------------------------- End of HRNLM.c ------------------------- */


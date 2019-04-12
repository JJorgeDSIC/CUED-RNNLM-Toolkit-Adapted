/* ----------------------------------------------------------- */
/*                                                             */
/*                          ___                                */
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
/*      http://svr-www.eng.cam.ac.uk/                          */
/*                                                             */
/*      Entropic Cambridge Research Laboratory                 */
/*      (now part of Microsoft)                                */
/*                                                             */
/* ----------------------------------------------------------- */
/*         Copyright: Microsoft Corporation                    */
/*          1995-2000 Redmond, Washington USA                  */
/*                    http://www.microsoft.com                 */
/*                                                             */
/*          2001-2002 Cambridge University                     */
/*                    Engineering Department                   */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*         File: HLM.h language model handling                 */
/* ----------------------------------------------------------- */

/* !HVER!HLM:   3.4.1 [CUED 12/03/09] */

#ifndef _HLM_H_
#define _HLM_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { boNGram=1, matBigram, mlpLM, intpltLM, rnnLM, hlmModel } LMType;

/* Calculate word|class log probability value : taken from LModel.h */
#define LOG_NATURAL(x) (((x) < 1.0E-20) ? -99.9900 : log(x))

#define EMINARG -46.05
#define UNLOG_NATURAL(x) (((x) < EMINARG) ? 0.0 : exp(x))

#if 0
#define MLPLMPROBNORM 1         /* MLP output prob normalization
                                   for cases WITHOUT OOS node */
#endif

#if 0
#define MLPLMPROBNORM_OOS 1     /* MLP output prob normalization
                                   for cases WITH OOS node,
                                   MLPLMPROBNORM must also be 1 */
#define OOS_NODE 1
#endif

#ifndef MLPLMPROBNORM            /* if no prob norm at all for either
                                   case w/o OOS node */
#undef MLPLMPROBNORM_OOS
#endif

#ifndef OOS_NODE                 /* no OOS output node, then not doing
                                   associated normalization */
#undef MLPLMPROBNORM_OOS
#endif

#if 0                           /* for small vocab <= 64k */
#define MAX_LMID 65534          /* Max number of words */
typedef unsigned short lmId;    /* Type used by lm to id words  1..MAX_LMID */
typedef unsigned short lmCnt;   /* Type used by lm to count wds 0..MAX_LMID */
#else
#define MAX_LMID 4294967295     /* Max number of words */
typedef unsigned int lmId;      /* Type used by lm to id words  1..MAX_LMID */
typedef unsigned int lmCnt;     /* Type used by lm to count wds 0..MAX_LMID */
#endif

#define MAX_MATBILMID 65534     /* Max number of words */

/* #define NSIZE 4                 /\* Max length of ngram 2==bigram etc *\/ */
#define NSIZE 30                 /* Max length of ngram 2==bigram etc */

#define MAX_LMODEL 256          /* Max number of com LMs for interpolated LM */

typedef struct lmpplexacc {
   int nsent;                   /* accumumalte of number of sentences */
   int nword;                   /* accumumalte of number of words */
   int noov;                    /* accumumalte of number of OOV words */
   double loglike;              /* accumumalte of log-likelihood */
   FILE *oStreamFP;             /* output LM prob. stream file pointer */
} LMPPlexAcc;

typedef struct sentry {         /* HLM NGram probability */
   lmId word;                   /* word id */
   float prob;                  /* probability */
} SEntry;

typedef struct nentry {         /* HLM NGram history */
   lmId word[NSIZE-1];          /* Word history representing this entry */
   lmId fword[NSIZE];              /* Succeeding word */
   lmCnt nse;                   /* Number of ngrams for this entry */
   float bowt;                  /* Back-off weight */
   SEntry *se;                  /* Array[0..nse-1] of ngram probabilities */
   struct nentry *link;         /* Next entry in hash table */
   void *user;                  /* Accumulator or cache storage */
   Vector rnnlm_hist;           /* Vector representation v_{i-2, ..., 1} for partial CURRENT
                                   history in RNNLM to compute P(w_i | w_i-1, v_{i-2, ..., 1})
                                 */
   Vector rnnlm_fhist;          /* Vector representation v_{i-1, ..., 1} for partial FUTURE
                                   history in RNNLM to compute P(w_i+1 | w_i, v_{i-1, ..., 1})
                                   generated AFTER computing P(w_i | w_i-1, v_{i-2, ..., 1})
                                */
} NEntry;

typedef struct ngramlm {
   int nsize;                   /* Unigram==1, Bigram==2, Trigram==3 */
   unsigned int hashsize;       /* Size of hashtab (adjusted by lm counts) */
   NEntry **hashtab;            /* Hash table for finding NEntries */
   int counts[NSIZE+1];         /* Number of [n]grams */
   int vocSize;                 /* Core LM size */
   Vector unigrams;             /* Unigram probabilities */
   LabId *wdlist;               /* Lookup table for words from lmId */
   MemHeap *heap;               /* Pointer to heap */
} NGramLM;

typedef struct matbilm {
   lmCnt numWords;              /* Number of words for language model */
   Matrix bigMat;               /* Actual probs */
   LabId *wdlist;               /* Lookup table for words from lmId */
   MemHeap *heap;               /* Pointer to heap */
} MatBiLM;

typedef struct mlplm {
   Ptr *mlp;                    /* MLP LM */
   Ptr *cache;                  /* Full span N-gram cache model w.r.t. MLP vocab */
   Ptr *lmstate;                /* Full span N-gram cache for full vocab LM states */
#ifdef MLPLMPROBNORM
   Ptr *nglm;                   /* Back-off N-gram LM for MLP LM prob normalization */
#endif
   LabId *wdlist;               /* Lookup table for words from lmId */
   MemHeap *heap;               /* Pointer to heap */
} MLPLM;

typedef struct {
   Ptr rnnlm;                   /* similar to MLPLM->mlp, will hold RNNLM in HRNLM.c */
   Ptr *cache;                  /* Full span N-gram cache model w.r.t. RNN vocab */
   Ptr *lmstate;                /* Full span N-gram cache for full vocab LM states */
#ifdef MLPLMPROBNORM
   Ptr *nglm;                   /* Back-off N-gram LM for RNNLM prob normalization */
#endif
   LabId *wdlist;               /* Lookup table for words from lmId */
   NEntry *curlmste;            /* current LM state - updated after prob. computation */
   Vector rnnlm_hist;           /* Vector representation v_{i-2, ..., 1} for partial CURRENT
                                   history in RNNLM to compute P(w_i | w_i-1, v_{i-2, ..., 1})
                                 */
   Vector rnnlm_fhist;          /* Vector representation v_{i-1, ..., 1} for partial FUTURE
                                   history in RNNLM to compute P(w_i+1 | w_i, v_{i-1, ..., 1})
                                   generated AFTER computing P(w_i | w_i-1, v_{i-2, ..., 1})
                                 */
   MemHeap *heap ;
} RNLM;

typedef struct intpltlm
{
   int nModels;                 /* Number of component language models */
   Vector weight;               /* Interpolation weight */
   Ptr **lms;                   /* Language models to be interpolated */
   void *wsi;                   /* LM weight set */
   void *RePtrHT;               /* Recombination for interpolated models (trn) */
   MemHeap *heap;               /* Pointer to heap */
   float restprobweight         /* weight for LM prob from lattice */
} IntpltLM;

typedef struct lmodel {
    int id;
   char *name;                  /* Name used for identifying lm */
   LMType type;                 /* LM type */
   LogFloat pen;                /* Word insertion penalty */
   float scale;                 /* Language model scale */
   union {
      MatBiLM *matbi;
      NGramLM *ngram;
      MLPLM *mlplm;
      RNLM *rnlm;
      IntpltLM *ilang;
      void *hlmModel;
   }
   data;
   MemHeap *heap;               /* Heap for allocating lm structs */
   LMPPlexAcc *ppinfo;          /* LM perplexity information */
} LModel;


void InitLM(void);
/*
   Initialise the module
*/

/* ---------------- Lower Level Routines ----------------- */

NGramLM *CreateBoNGram(LModel *lm,int vocSize,int counts[NSIZE+1]);
/*
   Create backoff NGram language models with size defined by counts.
    vocSize=number of words in vocabulary
    counts[1]=number of unigrams
    counts[2]=approximate number of bigrams
    counts[3]=approximate number of trigrams
               (approximate sizes are used to determine hash table size)
*/
MatBiLM *CreateMatBigram(LModel *lm,int nw);
/*
   Create matrix bigram language models of specified size.
*/

NEntry *GetNEntry(NGramLM *nglm,lmId ndx[NSIZE],Boolean create);
/*
   Find [create] ngram entry for word histories ...ndx[1] ndx[0].
*/

/* --------------- Higher Level Routines ----------------- */

float GetLMProb(LModel *lm, LabId prid[NSIZE], LabId wdid, LabId succwdid);
/*
   Calculate probability for wdid following prid
*/

LModel *ReadLModel(MemHeap *heap,char *fn, int id);
void WriteLModel(LModel *lm,char *fn,int flags);
/*
   Read/write language model from/to specified file.
   Flags control format for writing.
*/

void ClearLModel(LModel *lm);
/*
   Clear LModel before deletion
*/

/* ResetMLPLMCache : reset ngram and LM state caches of MLP LM */
void ResetMLPLMCache(LModel *lm);

#ifndef NO_LAT_LM
typedef Ptr LMState;

LogFloat LMTrans (LModel *lm, LMState src, LabId wdid, LMState *dest, LMState *src_mix, LMState *dest_mix, LabId *succwordId, LogFloat lmlike);
#endif

#ifdef __cplusplus
}
#endif

#endif  /* _HLM_H_ */

/* ---------------------- End of HLM.h ----------------------- */

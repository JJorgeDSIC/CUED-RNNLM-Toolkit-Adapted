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
/*         File: HLM.c  language model handling                */
/* ----------------------------------------------------------- */

char *hlm_version = "!HVER!HLM:   3.4 [CUED 25/04/06]";
char *hlm_vc_id = "$Id: HLM.c,v 1.1.1.1 2006/10/11 09:54:57 jal58 Exp $";

#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HWave.h"
#include "HLabel.h"
#include "HLM.h"
#include "HNLM.h"
#include "HRNLM.h"

/* --------------------------- Trace Flags ------------------------- */

#define T_TIO 1  /* Progress tracing whilst performing IO */

static int trace=0;

/* --------------------------- Initialisation ---------------------- */

#define LN10 2.30258509299404568 /* Defined to save recalculating it */

static Boolean rawMITFormat = FALSE;    /* Don't use HTK quoting and escapes */
static Boolean useIntpltLM = FALSE;     /* Use interpolated LM */
static Boolean padStartWord = FALSE;    /* Padding start word for MLP LM */
static Boolean nnlmOrigFormat = FALSE;  /* Set to TRUE for backward compatability */
static Boolean rnnlmUseHVDist = FALSE;  /* RNNLM using history vector distance in lattice expansion */
static int origMLPNSize = 4;            /* Allow specification MLP SIZE on command line */
/* static int ACTUAL_NSIZE = NSIZE;        /\* Actual N-length of n-gram LM *\/ */
static int ACTUAL_MLP_NSIZE = MLP_NSIZE;/* Actual N-length of MLP LM */
static int rnnlmLMStateSize = 2;        /* RNNLM effective LM state size, used together
                                           with history vector distance threshold,
                                           default is 2, i.e., previous word only (to
                                           to combined with history vector distance) */
static double rnnlmMinHVDist = 0.00001; /* RNNLM history vector distance threshold */

static ConfParam *cParm[MAXGLOBS];      /* config parameters */
static int nParm = 0;

/* ---------------------- Global Variables ----------------------- */

static MemHeap mlpLMCacheInfoStack[10];          /* Local stack to for MLP cache info */
static MemHeap mlpLMStateInfoStack;          /* Local stack to for MLP LM info */

/* EXPORT->InitLM: initialise configuration parameters */
void InitLM(void)
{
   Boolean b;
   int i;
   double f = 0.0;

   Register(hlm_version,hlm_vc_id);
   nParm = GetConfig("HLM", TRUE, cParm, MAXGLOBS);
   if (nParm>0){
      if (GetConfInt(cParm,nParm,"TRACE",&i)) trace = i;
      if (GetConfBool(cParm,nParm,"RAWMITFORMAT",&b)) rawMITFormat = b;
      if (GetConfBool(cParm,nParm,"PADSTARTWORD",&b)) padStartWord = b;
      if (GetConfBool(cParm,nParm,"NNLMORIGFORMAT",&b)) nnlmOrigFormat = b;
      if (GetConfBool(cParm,nParm,"RNNLMUSEHVDIST",&b)) rnnlmUseHVDist = b;
      if (GetConfInt(cParm,nParm,"ORIGMLPNSIZE",&i)) origMLPNSize = i;
      if (GetConfInt(cParm,nParm,"RNNLMSTATESIZE",&i)) rnnlmLMStateSize = i;
      if (GetConfFlt(cParm,nParm,"RNNLMMINHVDIST",&f)) rnnlmMinHVDist = f;
   }

   /* setup the local memory management */
   for (i=0; i<10; i++)
   {
       CreateHeap(&mlpLMCacheInfoStack[i], "mlpLMCacheInfoStore", MSTAK, 1, 0.0, 4000, 4000);
   }
   CreateHeap(&mlpLMStateInfoStack, "mlpLMStateInfoStore", MSTAK, 1, 0.0, 4000, 4000);
}

/*------------------------- Input Scanner ---------------------------*/

static Source source;           /* input file */

static char *GetInLineSource(Source *src,char *buf)
{
  int  i, c;

  c = GetCh(src);
  if (c==EOF)
    return NULL;
  i = 0;
  while (c!='\n' && i<MAXSTRLEN) {
    buf[i++] = c;
    c = GetCh(src);
  }
  buf[i] = '\0';
  return buf;
}

/* GetInLine: read a complete line from source */
static char *GetInLine(char *buf)
{
   int  i, c;

   c = GetCh(&source);
   if (c==EOF)
      return NULL;
   i = 0;
   while (c!='\n' && i<MAXSTRLEN) {
      buf[i++] = c;
      c = GetCh(&source);
   }
   buf[i] = '\0';
   return buf;
}

/* SyncStr: read input until str found */
static void SyncStr(char *buf,char *str)
{
   while (strcmp(buf, str)!=0) {
      if (GetInLine(buf)==NULL)
         HError(8150,"SyncStr: EOF searching for %s", str);
   }
}

/* GetInt: read int from input stream */
static int GetInt(void)
{
   int x;
   char buf[100];

   if (!ReadInt(&source,&x,1,FALSE))
      HError(8150,"GetInt: Int Expected at %s",SrcPosition(source,buf));
   return x;
}

/* GetFLoat: read float from input stream */
static float GetFloat(Boolean bin)
{
   float x;
   char buf[100];

   if (!ReadFloat(&source,&x,1,bin))
      HError(8150,"GetFloat: Float Expected at %s",SrcPosition(source,buf));
   return x;
}

/* ReadLMWord: read a string from input stream */
static char *ReadLMWord(char *buf)
{
   int i, c;

   if (rawMITFormat) {
      while (isspace(c=GetCh(&source)));
      i=0;
      while (!isspace(c) && c!=EOF && i<MAXSTRLEN){
         buf[i++] = c; c=GetCh(&source);
      }
      buf[i] = '\0';
      UnGetCh(c,&source);
      if (i>0)
         return buf;
      else
         return NULL;
   }
   else {
      if (ReadString(&source,buf))
         return buf;
      else
         return NULL;
   }
}

/* RNNLM handling functions */
#include "HLM-RNNLM.c"


/*------------------------- NEntry handling ---------------------------*/

static int hvs[]= { 165902236, 220889002, 32510287, 117809592,
                    165902236, 220889002, 32510287, 117809592 };

/* EXPORT->GetNEntry: Access specific NGram entry indexed by ndx */
NEntry *GetNEntry(NGramLM *nglm,lmId ndx[NSIZE],Boolean create)
{
   NEntry *ne;
   unsigned int hash;
   int i;
   /* #define LM_HASH_CHECK */

   hash=0;
   for (i=0;i<NSIZE-1;i++)
      hash=hash+(ndx[i]*hvs[i]);
   hash=(hash>>7)&(nglm->hashsize-1);

   for (ne=nglm->hashtab[hash]; ne!=NULL; ne=ne->link) {
      if (ne->word[0]==ndx[0]
#if NSIZE > 2
          && ne->word[1]==ndx[1]
#endif
#if NSIZE > 3
          && ne->word[2]==ndx[2]
#endif
#if NSIZE > 4
          && ne->word[3]==ndx[3]
#endif
#if NSIZE > 5
          && ne->word[4]==ndx[4]
#endif
#if NSIZE > 6
          && ne->word[5]==ndx[5]
#endif
#if NSIZE > 7
          && ne->word[6]==ndx[6]
#endif
#if NSIZE > 8
          && ne->word[7]==ndx[7]
#endif
          )
         break;
   }

   if (ne==NULL && create) {
      ne=(NEntry *) New(nglm->heap,sizeof(NEntry));
      nglm->counts[0]++;

      for (i=0;i<NSIZE-1;i++)
         ne->word[i]=ndx[i];
      /* BEGIN: Modification */
      for (i=0; i<10; i++)
	{
          ne->fword[i] = 0; /* 0??? */
	}
      /* END: Modification */
      ne->user=0;
      ne->nse=0;
      ne->se=NULL;;
      ne->bowt=0.0;
      ne->link=nglm->hashtab[hash];
      nglm->hashtab[hash]=ne;
   }

   return(ne);
}

/* EXPORT->GetNEntry2: Access specific NGram entry indexed by ndx */
NEntry *GetNEntry2(NGramLM *nglm,lmId ndx[NSIZE],Boolean create, int ngsize)
{
   NEntry *ne;
   unsigned int hash;
   int i;

   hash=0;
   for (i=0;i<ngsize-1;i++)
      hash=hash+(ndx[i]*hvs[i]);
   hash=(hash>>7)&(nglm->hashsize-1);

   for (ne=nglm->hashtab[hash]; ne!=NULL; ne=ne->link) {
      if (ne->word[0]==ndx[0]
#if ngsize > 2
          && ne->word[1]==ndx[1]
#endif
#if ngsize > 3
          && ne->word[2]==ndx[2]
#endif
#if ngsize > 4
          && ne->word[3]==ndx[3]
#endif
#if ngsize > 5
          && ne->word[4]==ndx[4]
#endif
#if ngsize > 6
          && ne->word[5]==ndx[5]
#endif
#if ngsize > 7
          && ne->word[6]==ndx[6]
#endif
#if ngsize > 8
          && ne->word[7]==ndx[7]
#endif
          )
         break;
   }

   if (ne==NULL && create) {
      ne=(NEntry *) New(nglm->heap,sizeof(NEntry));
      nglm->counts[0]++;

      for (i=0;i<ngsize-1;i++)
         ne->word[i]=ndx[i];
      ne->user=0;
      ne->nse=0;
      ne->se=NULL;;
      ne->bowt=0.0;
      ne->link=nglm->hashtab[hash];
      nglm->hashtab[hash]=ne;
   }

   return(ne);
}

NEntry *GetNEntry2_RNNLM_Lmstate(NGramLM *nglm,lmId ndx[NSIZE],Boolean create, int ngsize, Vector v, Vector fv)
{
    ngsize = 14;
    NEntry *ne;
    unsigned int hash;
    int i;

    hash=0;
    for (i=0;i<ngsize-1;i++)
        hash=hash+(ndx[i]*hvs[i]);
    hash=(hash>>7)&(nglm->hashsize-1);

    for (ne=nglm->hashtab[hash]; ne!=NULL; ne=ne->link) {
        if (ne->word[0]==ndx[0]
            && ne->word[1]==ndx[1]
            && ne->word[2]==ndx[2]
            && ne->word[3]==ndx[3]
            && ne->word[4]==ndx[4]
            && ne->word[5]==ndx[5]
            && ne->word[6]==ndx[6]
            && ne->word[7]==ndx[7]
            && ne->word[8]==ndx[8]
            && ne->word[9]==ndx[9]
            && ne->word[10]==ndx[10]
            && ne->word[11]==ndx[11]
            && ne->word[12]==ndx[12]
            && ne->word[13]==ndx[13]
            )
            break;
    }

    if (ne==NULL && create) {
        ne=(NEntry *) New(nglm->heap,sizeof(NEntry));
        nglm->counts[0]++;

        for (i=0;i<NSIZE-1;i++)
            ne->word[i]=ndx[i];
        ne->user=0;
        ne->nse=0;
        ne->se=NULL;;
        ne->bowt=0.0;
        ne->link=nglm->hashtab[hash];
        nglm->hashtab[hash]=ne;
        /* RNNLM current history vector */
        ne->rnnlm_hist = NULL;
        /* RNNLM future history vector */
        ne->rnnlm_fhist = NULL;
    }

    return(ne);
}

NEntry *GetNEntry2_RNNLM_ProbCache(NGramLM *nglm,lmId ndx[NSIZE],Boolean create, int ngsize, Vector v, Vector fv)
{

    ngsize = 14;
    NEntry *ne;
    unsigned int hash;
    int i;

    hash=0;
    for (i=0;i<ngsize-1;i++)
        hash=hash+(ndx[i]*hvs[i]);
    hash=(hash>>7)&(nglm->hashsize-1);

    for (ne=nglm->hashtab[hash]; ne!=NULL; ne=ne->link) {
        if (ne->word[0]==ndx[0]
            && ne->word[1]==ndx[1]
            && ne->word[2]==ndx[2]
            && ne->word[3]==ndx[3]
            && ne->word[4]==ndx[4]
            && ne->word[5]==ndx[5]
            && ne->word[6]==ndx[6]
            && ne->word[7]==ndx[7]
            && ne->word[8]==ndx[8]
            && ne->word[9]==ndx[9]
            && ne->word[10]==ndx[10]
            && ne->word[11]==ndx[11]
            && ne->word[12]==ndx[12]
            && ne->word[13]==ndx[13]
            )
            break;
    }

    if (ne==NULL && create) {
        ne=(NEntry *) New(nglm->heap,sizeof(NEntry));
        nglm->counts[0]++;

        for (i=0;i<NSIZE-1;i++)
            ne->word[i]=ndx[i];
        ne->user=0;
        ne->nse=0;
        ne->se=NULL;;
        ne->bowt=0.0;
        ne->link=nglm->hashtab[hash];
        nglm->hashtab[hash]=ne;
        /* RNNLM current history vector */
        ne->rnnlm_hist = NULL;
        /* RNNLM future history vector */
        ne->rnnlm_fhist = NULL;
    }

    return(ne);
}

/* EXPORT->GetNEntry2_RNNLM: Access specific NGram entry indexed by ndx and stores
   an associated given history vector */
NEntry *GetNEntry2_RNNLM(NGramLM *nglm,lmId ndx[NSIZE], lmId fndx[NSIZE], Boolean use_prob_table, Boolean create, int ngsize, Vector v, Vector fv)
{

   ngsize = 20;
   NEntry *ne;
   unsigned int hash;
   int i;

   hash=0;
   for (i=0;i<ngsize-1;i++)
      hash=hash+(ndx[i]*hvs[i%8]);
   hash=(hash>>7)&(nglm->hashsize-1);

   for (ne=nglm->hashtab[hash]; ne!=NULL; ne=ne->link) {
      if (ne->word[0]==ndx[0] && (use_prob_table ? (ne->fword[0]==fndx[0] && ne->fword[1]==fndx[1] && ne->fword[2]==fndx[2]) : 1)
          && ne->word[1]==ndx[1]
          && ne->word[2]==ndx[2]
          && ne->word[3]==ndx[3]
          && ne->word[4]==ndx[4]
          && ne->word[5]==ndx[5]
          && ne->word[6]==ndx[6]
          && ne->word[7]==ndx[7]
          && ne->word[8]==ndx[8]
          && ne->word[9]==ndx[9]
          && ne->word[10]==ndx[10]
          && ne->word[11]==ndx[11]
          && ne->word[12]==ndx[12]
          && ne->word[13]==ndx[13]
          && ne->word[14]==ndx[14]
          && ne->word[15]==ndx[15]
          && ne->word[16]==ndx[16]
          && ne->word[17]==ndx[17]
          && ne->word[18]==ndx[18]
          && ne->word[19]==ndx[19]
          )
         break;
   }

   if (ne==NULL && create)
   {
      ne=(NEntry *) New(nglm->heap,sizeof(NEntry));
      nglm->counts[0]++;

      for (i=0;i<NSIZE-1;i++)
      {
          ne->word[i]=ndx[i];
      }
      for (i=0; i<10; i++)
      {
          ne->fword[i] = fndx[i];
      }
      ne->user=0;
      ne->nse=0;
      ne->se=NULL;;
      ne->bowt=0.0;
      ne->link=nglm->hashtab[hash];
      nglm->hashtab[hash]=ne;
      /* RNNLM current history vector */
      ne->rnnlm_hist = NULL;
      /* RNNLM future history vector */
      ne->rnnlm_fhist = NULL;
   }

   return(ne);
}


NEntry *GetNEntry2_RNNLM_HVDist(NGramLM *nglm,lmId ndx[NSIZE],Boolean create, int ngsize, Vector v, Vector fv)
{

   NEntry *ne;
   unsigned int hash;
   int i;

   hash=0;
   for (i=0;i<ngsize-1;i++)
      hash=hash+(ndx[i]*hvs[i]);
   hash=(hash>>7)&(nglm->hashsize-1);

   for (ne=nglm->hashtab[hash]; ne!=NULL; ne=ne->link) {
      if (ne->word[0]==ndx[0]
#if ngsize > 2
          && ne->word[1]==ndx[1]
#endif
#if ngsize > 3
          && ne->word[2]==ndx[2]
#endif
#if ngsize > 4
          && ne->word[3]==ndx[3]
#endif
#if ngsize > 5
          && ne->word[4]==ndx[4]
#endif
#if ngsize > 6
          && ne->word[5]==ndx[5]
#endif
#if ngsize > 7
          && ne->word[6]==ndx[6]
#endif
#if ngsize > 8
          && ne->word[7]==ndx[7]
#endif
          && CalcHistsDistance(ne->rnnlm_hist, v) <= rnnlmMinHVDist
          )
         break;
   }

   if (ne==NULL && create) {
      ne=(NEntry *) New(nglm->heap,sizeof(NEntry));
      nglm->counts[0]++;

      for (i=0;i<NSIZE-1;i++)
         ne->word[i]=ndx[i];
      ne->user=0;
      ne->nse=0;
      ne->se=NULL;;
      ne->bowt=0.0;
      ne->link=nglm->hashtab[hash];
      nglm->hashtab[hash]=ne;
      /* RNNLM current history vector */
      ne->rnnlm_hist = NULL;
      /* RNNLM future history vector */
      ne->rnnlm_fhist = NULL;
   }

   return(ne);
}

static int se_cmp(const void *v1,const void *v2)
{
   SEntry *s1,*s2;

   s1=(SEntry*)v1;s2=(SEntry*)v2;
   return((int)(s1->word-s2->word));
}

/*--------------------- ARPA-style NGrams ------------------------*/

static int nep_cmp(const void *v1,const void *v2)
{
   NEntry *n1,*n2;
   int res,i;

   res=0; n1=*((NEntry**)v1); n2=*((NEntry**)v2);
   for(i=NSIZE-2;i>=0;i--)
      if (n1->word[i]!=n2->word[i]) {
         res=(n1->word[i]-n2->word[i]);
         break;
      }
   return(res);
}


/* WriteNGram: Write n grams to file */
static int WriteNGrams(FILE *file,NGramLM *nglm,int n,float scale)
{
   NEntry *ne,*be,*ce,**neTab;
   SEntry *se;
   LogFloat prob;
   lmId ndx[NSIZE+1];
   int c,i,j,k,N,g=1,hash,neCnt,total;

   if (trace&T_TIO)
      printf("\nn%1d ",n),fflush(stdout);
   fprintf(file,"\n\\%d-grams:\n",n);
   N=VectorSize(nglm->unigrams);

   neTab=(NEntry **) New(&gstack,sizeof(NEntry*)*nglm->counts[0]);

   for (hash=neCnt=0;hash<nglm->hashsize;hash++)
      for (ne=nglm->hashtab[hash]; ne!=NULL; ne=ne->link) {
         for (i=1,ce=ne;i<n;i++)
            if (ne->word[i-1]==0) {
               ce=NULL;
               break;
            }
         if (ce!=NULL)
            for (i=n;i<NSIZE;i++)
               if (ne->word[i-1]!=0) {
                  ce=NULL;
                  break;
               }
         if (ce!=NULL && ce->nse>0)
            neTab[neCnt++]=ce;
      }
   qsort(neTab,neCnt,sizeof(NEntry*),nep_cmp);

   total=0;
   for (c=n;c<=NSIZE;c++) ndx[c]=0;
   for (j=0;j<neCnt;j++) {
      ne=neTab[j];
      for (c=1;c<n;c++) ndx[c]=ne->word[c-1];
      if (ne!=NULL && ne->nse>0) {
         for (i=0,se=ne->se;i<ne->nse;i++,se++) {
            if (trace&T_TIO) {
               if ((g%25000)==0)
                  printf(". "),fflush(stdout);
               if ((g%800000)==0)
                  printf("\n   "),fflush(stdout);
               g++;
            }
            ndx[0]=se->word;

            if (n<nglm->nsize) be=GetNEntry(nglm,ndx,FALSE);
            else be=NULL;
            if (be==NULL || be->nse==0) be=NULL;
            total++;
            if (n==1) prob=nglm->unigrams[se->word];
            else prob=se->prob;
            if (prob*scale<-99.999)
               fprintf(file,"%+6.3f",-99.999);
            else
               fprintf(file,"%+6.4f",prob*scale);
            c='\t';
            for (k=n-1;k>=0;k--)
               if (rawMITFormat)
                  fprintf(file,"%c%s",c,nglm->wdlist[ndx[k]]->name),c=' ';
               else
                  fprintf(file,"%c%s",c,
                          ReWriteString(nglm->wdlist[ndx[k]]->name,
                                        NULL,ESCAPE_CHAR)),c=' ';
            if (be!=NULL)
               fprintf(file,"\t%+6.4f\n",be->bowt*scale);
            else
               fprintf(file,"\n");
         }
      }

   }
   Dispose(&gstack,neTab);
   if (trace&T_TIO)
      printf("\n"),fflush(stdout);
   return(total);
}

#define PROGRESS(g) \
   if (trace&T_TIO) { \
      if ((g%25000)==0) \
         printf(". "),fflush(stdout); \
      if ((g%800000)==0) \
         printf("\n   "),fflush(stdout); \
   }

#if 0
#define NGHSIZE1 8192
#define NGHSIZE2 32768
#define NGHSIZE3 131072
#else
#define NGHSIZE1 65536
#define NGHSIZE2 NGHSIZE1*8
#define NGHSIZE3 NGHSIZE2*8
#define NGHSIZE4 NGHSIZE3*8
#endif

/* EXPORT->CreateBoNGram: Allocate and create basic NGram structures */
NGramLM *CreateBoNGram(LModel *lm,int vocSize, int counts[NSIZE])
{
   lmId ndx[NSIZE];
   int i,k;
   NGramLM *nglm;

   nglm = (NGramLM *) New(lm->heap, sizeof(NGramLM));
   lm->data.ngram = nglm;
   nglm->heap = lm->heap;

   for (i=0;i<NSIZE;i++) nglm->counts[i]=0;
   for (i=1;i<NSIZE;i++)
      if (counts[i]==0) break;
      else nglm->counts[i]=counts[i];
   nglm->nsize=i-1;

   /* Don't count final layer */
   for (k=0,i=1;i<nglm->nsize;i++)
      k+=nglm->counts[i];
   /* Then use total to guess NEntry hash size */
#if 0
   if (k<25000)
      nglm->hashsize=NGHSIZE1;
   else if (k<250000)
      nglm->hashsize=NGHSIZE2;
   else
      nglm->hashsize=NGHSIZE3;
#else
   if (k<(NGHSIZE1*8))
       nglm->hashsize=NGHSIZE1;
   else if (k<(NGHSIZE2*8))
       nglm->hashsize=NGHSIZE2;
   else if (k<(NGHSIZE3*8))
       nglm->hashsize=NGHSIZE3;
   else
       nglm->hashsize=NGHSIZE4;
#endif

   nglm->hashtab=(NEntry **) New(lm->heap,sizeof(NEntry*)*nglm->hashsize);
   for (i=0; i<nglm->hashsize; i++)
      nglm->hashtab[i]=NULL;

   nglm->vocSize = vocSize;
   nglm->unigrams = CreateVector(lm->heap,nglm->vocSize);
   nglm->wdlist = (LabId *) New(lm->heap,nglm->vocSize*sizeof(LabId)); nglm->wdlist--;
   for (i=1;i<=nglm->vocSize;i++) nglm->wdlist[i]=NULL;

   for (i=0;i<NSIZE;i++) ndx[i]=0;
   GetNEntry(nglm,ndx,TRUE);

   return(nglm);
}

#define BIN_ARPA_HAS_BOWT 1
#define BIN_ARPA_INT_LMID 2

/* ReadNGrams: read n grams list from file */
static int ReadNGrams(NGramLM *nglm,int n,int count, Boolean bin)
{
   float prob;
   LabId wdid;
   SEntry *cse;
   char wd[255];
   lmId ndx[NSIZE+1];
   NEntry *ne,*le=NULL;
   int i, g, idx, total;
   unsigned char size, flags=0;

   cse = (SEntry *) New(nglm->heap,count*sizeof(SEntry));
   for (i=1;i<=NSIZE;i++) ndx[i]=0;

   if (trace&T_TIO)
      printf("\nn%1d ",n),fflush(stdout);

   total=0;
   for (g=1; g<=count; g++){
      PROGRESS(g);

      if (bin) {
         size = GetCh (&source);
         flags = GetCh (&source);
      }

      prob = GetFloat(bin)*LN10;

      if (n==1) { /* unigram treated as special */
         ReadLMWord(wd);
         wdid = GetLabId(wd, FALSE);
         if (!wdid) {
            wdid = GetLabId(wd, TRUE);
         }
         /* Using IntpltLM can have the label ptr set multiple times */
         if (wdid->aux != NULL && !useIntpltLM)
            HError(8150,"ReadNGrams: Duplicate word (%s) in 1-gram list",
                   wdid->name);
         if (wdid->aux == NULL) wdid->aux = (Ptr)g;
         nglm->wdlist[g] = wdid;
         nglm->unigrams[g] = prob;
         ndx[0]=g;
      } else {    /* bigram, trigram, etc. */
         for (i=0;i<n;i++) {
            if (bin) {
               if (flags & BIN_ARPA_INT_LMID) {
                  unsigned int ui;
                  if (!ReadInt (&source, (int *) &ui, 1, bin))
                     HError (9999, "ReadNGrams: failed reading int lm word id");
                  idx = ui;
               }
               else {
                  unsigned short us;
                  if (!ReadShort (&source, (short *) &us, 1, bin))
                     HError (9999, "ReadNGrams: failed reading short lm word id at");
                  idx = us;
               }
            }
            else {
               ReadLMWord(wd);
               wdid = GetLabId(wd, FALSE);
               idx = (wdid==NULL?0:(int)wdid->aux);
            }
            if (idx<1 || idx>nglm->vocSize)
               HError(8150,"ReadNGrams: Unseen word (%s) in %dGram",wd,n);
            ndx[n-1-i]=idx;
         }
      }

      total++;
      ne = GetNEntry(nglm,ndx+1,FALSE);
      if (ne == NULL)
         HError(8150,"ReadNGrams: Backoff weight not seen for %dth %dGram",g,n);
      if (ne!=le) {
         if (le != NULL && ne->se != NULL)
            HError(8150,"ReadNGrams: %dth %dGrams out of order",g,n);
         if (le != NULL) {
            if (le->nse==0) {
               le->se=NULL;
            } else {
               qsort(le->se,le->nse,sizeof(SEntry),se_cmp);
            }
         }
         ne->se = cse;
         ne->nse = 0;
         le = ne;
      }
      cse->prob = prob;
      cse->word = ndx[0];
      ne->nse++; cse++;

      /* read back-off weight */
      if (bin) {
         if (flags & BIN_ARPA_HAS_BOWT) {
            ne = GetNEntry(nglm,ndx,TRUE);
            ne->bowt = GetFloat (TRUE)*LN10;
         }
      }
      else {
         SkipWhiteSpace(&source);
         if (!source.wasNewline) {
            ne=GetNEntry(nglm,ndx,TRUE);
            ne->bowt = GetFloat(FALSE)*LN10;
         }
      }
   }

   /* deal with the last accumulated set */
   if (le != NULL) {
      if (le->nse==0) {
         le->se=NULL;
      } else {
         qsort(le->se,le->nse,sizeof(SEntry),se_cmp);
      }
   }

   if (trace&T_TIO)
      printf("\n"),fflush(stdout);

   return(total);
}

/* ReadBoNGram: read and store WSJ/DP format ngram */
static void ReadBoNGram(LModel *lm,char *fn)
{
   NGramLM *nglm;
   int i,j,k,counts[NSIZE+1];
   Boolean ngBin[NSIZE+1];
   char buf[MAXSTRLEN+1],syc[64];
   char ngFmtCh;

   if (trace&T_TIO)
      printf("\nBOffB "),fflush(stdout);

   if(InitSource(fn,&source,LangModFilter)<SUCCESS)
      HError(8110,"ReadBoNGram: Can't open file %s", fn);
   GetInLine(buf);
   SyncStr(buf,"\\data\\");
   for (i=1;i<=NSIZE;i++) counts[i]=0;
   for (i=1;i<=NSIZE;i++) {
      GetInLine(buf);
      if (sscanf(buf, "ngram %d%c%d", &j, &ngFmtCh, &k)!=3 && i>1)
         break;
      if (i!=j || k==0)
         HError(8150,"ReadBoNGram: %dGram count missing (%s)",i,buf);

      switch (ngFmtCh) {
      case '=':
         ngBin[j] = FALSE;
         break;
      case '~':
         ngBin[j] = TRUE;
         break;
      default:
         HError (9999, "ReadARPALM: unknown ngram format type '%c'", ngFmtCh);
      }
      counts[j]=k;
   }

   if (ngBin[1])
      HError (8113, "ReadARPALM: unigram must be stored as text");

   nglm=CreateBoNGram(lm,counts[1],counts);
   for (i=1;i<=nglm->nsize;i++) {
      sprintf(syc,"\\%d-grams:",i);
      SyncStr(buf,syc);
      ReadNGrams(nglm,i,nglm->counts[i], ngBin[i]);
   }
   SyncStr(buf,"\\end\\");
   CloseSource(&source);

   if (trace&T_TIO) {
      printf("\n NEntry==%d ",nglm->counts[0]);
      for(i=1;i<=nglm->nsize;i++)
         printf(" %d-Grams==%d",i,nglm->counts[i]);
      printf("\n\n");
      fflush(stdout);
   }
}
/* WriteBoNGram: write out WSJ/DP format ngram */
static void WriteBoNGram(LModel *lm,char *fn,int flags)
{
   int i,k;
   FILE *file;
   NGramLM *nglm;
   Boolean isPipe;

   nglm = lm->data.ngram;
   file=FOpen(fn,LangModOFilter,&isPipe);
   fprintf(file,"\\data\\\n");

   for (i=1;i<=nglm->nsize;i++) {
      fprintf(file,"ngram %d=%d\n",i,nglm->counts[i]);
   }
   for (i=1;i<=nglm->nsize;i++) {
      k = WriteNGrams(file,nglm,i,1.0/LN10);
      if (k!=nglm->counts[i])
         HError(-8190,"WriteBoNGram: Counts disagree for %dgram (%d vs %d)",
                i, k, nglm->counts[i]);
   }
   fprintf(file,"\n\\end\\\n");
   FClose(file,isPipe);
}

void ClearBoNGram(LModel *lm)
{
   NGramLM *nglm = lm->data.ngram;
   int i;

   for(i=1;i<=nglm->vocSize;i++)
      if (nglm->wdlist[i]!=NULL) nglm->wdlist[i]->aux=0;
}

/* -------------- Matrix Bigram Handling Routines ----------- */

MatBiLM *CreateMatBigram(LModel *lm,int nw)
{
   MatBiLM *matbi;

   matbi = (MatBiLM *) New(lm->heap,sizeof(MatBiLM));
   lm->data.matbi = matbi;
   matbi->heap = lm->heap;

   matbi->numWords = nw;
   matbi->wdlist = (LabId *) New(lm->heap,sizeof(LabId)*(nw+1));
   matbi->bigMat = CreateMatrix(lm->heap,nw,nw);
   ZeroMatrix(matbi->bigMat);
   return(matbi);
}

/* ReadRow: read a row from bigram file f into v */
int ReadRow(Vector v)
{
   int i,j,N,cnt,c;
   float x;

   N = VectorSize(v);
   i=0;
   while(!source.wasNewline) {
      x = GetFloat(FALSE);
      c=GetCh(&source);
      if (c == '*')
         cnt=GetInt();
      else {
         UnGetCh(c,&source);
         cnt=1;
      }
      SkipWhiteSpace(&source);
      for (j=0;j<cnt;j++) {
         i++;
         if (i<=N) v[i] = x;
      }
   }
   return(i);
}

/* ReadBigram: load a bigram from given file */
static void ReadMatBigram(LModel *lm,char *fn)
{
   Vector vec;
   char buf[132];
   int P,p,j;
   float sum,x;
   LabId id;
   MatBiLM *matbi;

   if (trace&T_TIO)
      printf("\nMB "),fflush(stdout);

   if(InitSource(fn,&source,LangModFilter)<SUCCESS)
      HError(8110,"ReadMatBigram: Can't open file %s", fn);
   vec = CreateVector(&gcheap,MAX_MATBILMID);
   ReadLMWord(buf);SkipWhiteSpace(&source);
   id=GetLabId(buf,TRUE);
   P = ReadRow(vec);

   if (P<=0 || P >MAX_MATBILMID)
      HError(8151,"ReadMatBigram: First row invalid (%d entries)",P);

   matbi=CreateMatBigram(lm,P);

   matbi->wdlist[1] = id;
   for (p=1;p<=P;p++) matbi->bigMat[1][p]=vec[p];
   id->aux=(Ptr) 1;
   Dispose(&gcheap,vec);

   for (sum=0.0, j=1; j<=P; j++) {
      x = matbi->bigMat[1][j];
      if (x<0)
         HError(8151,"ReadMatBigram: In bigram, entry %d for %s is -ve (%e)",
                j,buf,x);
      sum += x;
      matbi->bigMat[1][j]=((x<MINLARG)?LZERO:log(x));
   }
   if (sum < 0.99 || sum > 1.01)
      HError(-8151,"ReadMatBigram: Row %d of bigram %s adds up to %f",1,fn,sum);

   for (p=2; ReadLMWord(buf); p++) {
      if (trace&T_TIO) {
         if ((p%25)==0)
            printf(". "),fflush(stdout);
         if ((p%800)==0)
            printf("\n   "),fflush(stdout);
      }
      if (p>P)
         HError(8150,"ReadMatBigram: More rows than columns in bigram %s",fn);
      id=GetLabId(buf,TRUE);
      if ((int)id->aux != 0)
         HError(8150,"ReadMatBigram: Duplicated name %s in bigram %s",buf,fn);
      id->aux = (Ptr) p;
      matbi->wdlist[p] = id;
      SkipWhiteSpace(&source);
      if (ReadRow(matbi->bigMat[p])!=P)
         HError(8150,"ReadMatBigram: Wrong number of items in row %d",p);
      for (sum=0.0, j=1; j<=P; j++) {
         x = matbi->bigMat[p][j];
         if (x<0)
            HError(8151,"ReadMatBigram: In bigram, entry %d for %s is -ve (%e)",
                   j,buf,x);
         sum += x;
         matbi->bigMat[p][j]=((x<MINLARG)?LZERO:log(x));
      }
      if (sum < 0.99 || sum > 1.01)
         HError(-8151,"ReadMatBigram: Row %d of bigram %s adds up to %f",p,fn,sum);
   }
   if (P>p)
      HError(8150,"ReadMatBigram: More columns than rows in bigram %s",fn);
   if (trace&T_TIO)
      printf("\n"),fflush(stdout);
   CloseSource(&source);
}

/* WriteMatBigram: write out old HVite format bigram */
static void WriteMatBigram(LModel *lm,char *fn,int flags)
{
   const float epsilon = 0.000001;
   MatBiLM *matbi;
   FILE *file;
   Boolean isPipe;
   Vector v;
   double x,y;
   int i,j,rep;

   if (trace&T_TIO)
      printf("\nMB "),fflush(stdout);

   matbi = lm->data.matbi;
   file=FOpen(fn,LangModOFilter,&isPipe);

   for (i=1;i<=matbi->numWords;i++) {
      if (trace&T_TIO) {
         if ((i%25)==0)
            printf(". "),fflush(stdout);
         if ((i%800)==0)
            printf("\n   "),fflush(stdout);
      }

      fprintf(file,"%-8s ",ReWriteString(matbi->wdlist[i]->name,
                                         NULL,ESCAPE_CHAR));

      v=matbi->bigMat[i];rep=0;x=-1.0;
      for (j=1;j<=matbi->numWords;j++){
         y = L2F(v[j]);
         if (fabs(y - x) <= epsilon) rep++;
         else {
            if (rep>0) {
               fprintf(file,"*%d",rep+1);
               rep=0;
            }
            x = y;
            if (x == 0.0)
               fprintf(file," 0");
            else if (x == 1.0)
               fprintf(file," 1");
            else
               fprintf(file," %e",x);
         }
      }
      if (rep>0)
         fprintf(file,"*%d",rep+1);
      fprintf(file,"\n");
   }
   FClose(file,isPipe);
   if (trace&T_TIO)
      printf("\n"),fflush(stdout);
}


/*--------------------- MLP LMs ------------------------*/

/* ReadMLPLM: read and store MLP LM */
static void ReadMLPLM(LModel *lm, char *fn)
{
   int i = 0, k = 0, inVocSize = 0, outVocSize = 0;
   int counts[NSIZE] = {0};
   char buf[MAXSTRLEN+1];
   char *wgtfn = NULL, *inmapfn = NULL, *outmapfn = NULL;
   MLPLM *mlplm = NULL;
   Source src;

   if (trace&T_TIO)
      printf("\nMLPLM \n"),fflush(stdout);

   if (InitSource(fn, &src, LangModFilter) < SUCCESS)
      HError(8110,"ReadMLPLM : Can't open file %s", fn);

   mlplm = (MLPLM *)New(lm->heap, sizeof(MLPLM));
   lm->data.mlplm = mlplm;
   /* for interpolated model, this can be later linked
      to other component ngram models' word list */
   mlplm->wdlist = NULL;
   mlplm->mlp = (Ptr *)New(lm->heap, sizeof(NNLM));
   mlplm->cache = mlplm->lmstate = NULL;
#ifdef MLPLMPROBNORM
   mlplm->nglm = NULL;
#endif
   mlplm->heap = lm->heap;

   if (!ReadString(&src, buf)) {
      HError(8110,"ReadMLPLM : Can't read header from MLPLM file %s", fn);
   }
   if (strcmp(buf, "!MLP") != 0) {
      HError(8110, "ReadMLPLM: expecting !MLP but got %s\n", buf);
   }
   if (nnlmOrigFormat) {
      /* original format MLPLM file */
      /* set order of NNLM */
      if (origMLPNSize > 0) {
         ACTUAL_MLP_NSIZE = origMLPNSize;
         if (ACTUAL_MLP_NSIZE > MLP_NSIZE) {
            HError(8110, "ReadMLPLM : MLP order %d beyond max size %d\n",
                   ACTUAL_MLP_NSIZE, MLP_NSIZE);
         }
         fprintf(stdout, "MLP order: %d\n", ACTUAL_MLP_NSIZE);
         fflush(stdout);
      } else {
	 HError(-8110,
                "ReadMLPLM : MLP order not specified - using default order %d\n",
                MLP_NSIZE);
      }

      if (!ReadString(&src, buf)) {
         HError(8110,"ReadMLPLM : Can't read weight file name from MLPLM file %s", fn);
      }
      wgtfn = (char *)New(&gstack, 4096);
      wgtfn = strcpy(wgtfn, buf);
      fprintf(stdout, "weight file: %s\n", wgtfn);
      fflush(stdout);

      if (!ReadString(&src, buf)) {
         HError(8110,"ReadMLPLM : Can't read input layer map file name from MLPLM file %s", fn);
      }
      inmapfn = (char *)New(&gstack, 4096);
      inmapfn = strcpy(inmapfn, buf);
      fprintf(stdout, "input layer vocab file: %s\n", inmapfn);
      fflush(stdout);

      if (!ReadString(&src, buf)) {
         HError(8110,"ReadMLPLM : Can't read output layer map file name from MLPLM file %s", fn);
      }
      outmapfn = (char *)New(&gstack, 4096);
      outmapfn = strcpy(outmapfn, buf);
      fprintf(stdout, "output layer vocab file: %s\n", outmapfn);
      fflush(stdout);

      if (!ReadInt(&src, &inVocSize, 1, FALSE)) {
      HError(8110, "ReadMLPLM: Expecting size of input layer vocab in %s\n", fn);
      }
      fprintf(stdout, "intput layer vocab size: %d\n", inVocSize);
      fflush(stdout);

      if (!ReadInt(&src, &outVocSize, 1, FALSE)) {
      HError(8110, "ReadMLPLM: Expecting size of output layer vocab in %s\n", fn);
      }
      fprintf(stdout, "output layer vocab size: %d\n", outVocSize);
      fflush(stdout);

   } else {
     /* v2 format MLPLM file */
      GetInLine(buf);
      if ((sscanf(buf, "order=%d", &k) != 1) || k==0) {
	HError(8110, "ReadMLPLM : MLP order missing (%s)", buf);
      }
      ACTUAL_MLP_NSIZE = k;
      if (ACTUAL_MLP_NSIZE > MLP_NSIZE) {
         HError(8110, "ReadMLPLM : MLP order %d beyond max size %d\n",
                ACTUAL_MLP_NSIZE, MLP_NSIZE);
      }
      fprintf(stdout, "MLP order: %d\n", ACTUAL_MLP_NSIZE);
      fflush(stdout);

      if (!ReadString(&src, buf)) {
         HError(8110,"ReadMLPLM : Can't read weight file name from MLPLM file %s", fn);
      }
      wgtfn = (char *)New(&gstack, 4096);
      wgtfn = strcpy(wgtfn, buf);
      fprintf(stdout, "weight file: %s\n", wgtfn);
      fflush(stdout);

      if (!ReadString(&src, buf)) {
         HError(8110,"ReadMLPLM : Can't read input layer map file name from MLPLM file %s", fn);
      }
      inmapfn = (char *)New(&gstack, 4096);
      inmapfn = strcpy(inmapfn, buf);
      fprintf(stdout, "input layer vocab file: %s\n", inmapfn);
      fflush(stdout);

      if (!ReadString(&src, buf)) {
         HError(8110,"ReadMLPLM : Can't read output layer map file name from MLPLM file %s", fn);
      }
      outmapfn = (char *)New(&gstack, 4096);
      outmapfn = strcpy(outmapfn, buf);
      fprintf(stdout, "output layer vocab file: %s\n", outmapfn);
      fflush(stdout);

      k = 0;
      GetInLine(buf);
      if ((sscanf(buf, "ninvoc=%d", &k) != 1) || k==0) {
	HError(8110, "ReadMLPLM : MLP input vocab size missing (%s)", buf);
      }
      inVocSize = k;
      fprintf(stdout, "intput layer vocab size: %d\n", inVocSize);
      fflush(stdout);

      k = 0;
      GetInLine(buf);
      if ((sscanf(buf, "noutvoc=%d", &k) != 1) || k==0) {
	HError(8110, "ReadMLPLM : MLP output vocab size missing (%s)", buf);
      }
      outVocSize = k;
      fprintf(stdout, "output layer vocab size: %d\n", outVocSize);
      fflush(stdout);
   }

   LoadNLMwgt(wgtfn, inmapfn, outmapfn, inVocSize, outVocSize, (NNLM *)mlplm->mlp);

   /* create ngram cache model */
   mlplm->cache = (Ptr *) New(lm->heap, sizeof(LModel));
   ((LModel*) mlplm->cache)->type = boNGram;
   ((LModel*) mlplm->cache)->heap = &mlpLMCacheInfoStack[lm->id];
   for (i=0; i<NSIZE; i++) {
      counts[i] = 250001;
   }
   fprintf(stdout, "Creating %d-gram cache for %d words ...\n", ACTUAL_MLP_NSIZE, outVocSize);
   fflush(stdout);
   ((LModel*) mlplm->cache)->data.ngram = CreateBoNGram(((LModel*) mlplm->cache), inVocSize, counts);

   /* create LM state cache */
   mlplm->lmstate = (Ptr *) New(lm->heap, sizeof(LModel));
   ((LModel*) mlplm->lmstate)->type = boNGram;
   ((LModel*) mlplm->lmstate)->heap = &mlpLMStateInfoStack;
   fprintf(stdout, "Creating LM state cache ...\n\n");
   fflush(stdout);
   ((LModel*) mlplm->lmstate)->data.ngram = CreateBoNGram(((LModel*) mlplm->lmstate), counts[0], counts);
}

/* UpdateMLPLMCache : query and update MLP LM ngram cache */
float UpdateMLPLMCache(LModel *lm, char **hist, NNLM *mlp, char *word, int N, Boolean useCache)
{
   int i = 0, hSize = 0;
   float prob = 0;
   LabId wdid = NULL;
   lmId ndx[MLP_NSIZE] = {0};
   NEntry *ne = NULL;
   SEntry *se = NULL, *cse = NULL;
   NGramLM *nglm = NULL;

#ifdef MLPLMPROBNORM
   int j = 0;
   LabId prid[MLP_NSIZE] = {0};
#endif

   if (!useCache) {
      prob = CalNLMProb(hist, mlp, word, N);
      return prob;
   }

   hSize = ACTUAL_MLP_NSIZE - 1;

   /* MLP LM history context */
   for (i=0; i<hSize; i++) {
      ndx[hSize - i] = String2Index(hist[i], mlp->inmap, mlp->in_num_word);
   }
   /* current word to predict */
   wdid = GetLabId(word, FALSE);
   if (!wdid) {
      HError(999, "UpdateMLPLMCache : Not expecting to find OOV word in cache : %s", word);
   }
   ndx[0] = String2Index(word, mlp->outmap, mlp->out_num_word);

#if 0
   fprintf(stdout, "Processing ngram : P(");
   for (j=0; j<hSize; j++) {
      fprintf(stdout, "%s %d ", hist[j], ndx[hSize - j]);
   }
   fprintf(stdout, "-> %s %d)\n", word, ndx[0]);
#endif

   /* query in cache ngram model */
   nglm = ((LModel *)lm->data.mlplm->cache)->data.ngram;
   ne = GetNEntry2(nglm, ndx+1, FALSE, ACTUAL_MLP_NSIZE);

   /* if ngrams of matching history found */
   if (ne && ne->nse == mlp->out_num_word) {
#if 0
      fprintf(stdout, "Found ngrams in cache : P(");
      for (j=0; j<hSize; j++) {
         fprintf(stdout, "%s ", hist[j]);
      }
      fprintf(stdout, "-> *)\n");
#endif
      se = ne->se + String2Index(word, mlp->outmap, mlp->out_num_word);
      prob = se->prob;
#ifdef MLPLMPROBNORM
/* MLP prob normalization for cases WITHOUT OOS output node */
#ifndef MLPLMPROBNORM_OOS
      prob *= exp(ne->bowt);
/* MLP prob normalization for cases WITH OOS output node */
#else
      if (String2Index(word, mlp->outmap, mlp->out_num_word) == mlp->out_num_word) {
         if (ne->bowt > LZERO) {
	    LabId prid[MLP_NSIZE] = {0};

            /* MLP LM history context */
            for (i=0; i<hSize; i++) {
               prid[hSize - i - 1] = GetLabId(hist[i], FALSE);
            }
            wdid = GetLabId(word, FALSE);

            prob *= exp(GetLMProb((LModel *)lm->data.mlplm->nglm, prid, wdid) - ne->bowt);
         }
      }
#endif
#endif
   }
   /* otherwise cache ngrams */
   else {
      prob = CalNLMProb(hist, mlp, word, N);

#if 0
      fprintf(stdout, "Inserting %d ngrams into cache : P(", mlp->out_num_word);
      for (j=0; j<hSize; j++) {
         fprintf(stdout, "%s ", hist[j]);
      }
      fprintf(stdout, "-> *)\n");
#endif

#if 1
      if (nglm->heap->totAlloc > 1073741824) {
         int counts[NSIZE] = {250001};
         ResetHeap(nglm->heap);
         nglm = CreateBoNGram((LModel *)lm->data.mlplm->cache, mlp->in_num_word, counts);
      }
#endif

      ne = GetNEntry2(nglm, ndx+1, TRUE, ACTUAL_MLP_NSIZE);
      ne->se = (SEntry *) New(nglm->heap, mlp->out_num_word * sizeof(SEntry));
      ne->nse = mlp->out_num_word;
      cse = ne->se;

      for (i=0; i<mlp->out_num_word; i++) {
         cse->prob = mlp->outP_arr[i];
         /* current word to predict */
         wdid = GetLabId(id2string_out(mlp, i), FALSE);
#if 0
         if (!wdid) {
            HError(-999, "UpdateMLPLMCache : Not expecting to cache OOV word : %s", word);
         }
#endif
         cse->word = (lmId) i;
         cse++;
      }

#ifdef MLPLMPROBNORM
      ne->bowt = LZERO;

      /* MLP LM history context */
      for (i=0; i<hSize; i++) {
         prid[hSize - i - 1] = GetLabId(hist[i], FALSE);
      }

/* MLP prob normalization for cases WITHOUT OOS output node */
#ifndef MLPLMPROBNORM_OOS
      for (i=0; i<mlp->out_num_word; i++) {
         /* current word to predict */
         wdid = GetLabId(id2string_out(mlp, i), FALSE);
         if (!wdid) {
            HError(999, "UpdateMLPLMCache : Not expecting to cache OOV word : %s", word);
         }
         ne->bowt = LAdd(ne->bowt, GetLMProb((LModel *)lm->data.mlplm->nglm, prid, wdid));
      }

#if 0
         fprintf(stdout, "short list word : %s prob normalization term: %e prob: %e %e\n",
                 word, ne->bowt, prob, prob*exp(ne->bowt));
         fflush(stdout);
#endif

      prob *= exp(ne->bowt);
/* MLP prob normalization for cases WITH OOS output node */
#else
      if (String2Index(word, mlp->outmap, mlp->out_num_word) == mlp->out_num_word) {
         for (i=1; i<=((LModel *)lm->data.mlplm->nglm)->data.ngram->vocSize; i++) {
            wdid = ((LModel *)lm->data.mlplm->nglm)->data.ngram->wdlist[i];
            /* only accumulate \sum_{w} P_ng(w|h) for OOS words */
            if (String2Index(wdid->name, mlp->outmap,
                             mlp->out_num_word) == mlp->out_num_word) {
               ne->bowt = LAdd(ne->bowt, GetLMProb((LModel *)lm->data.mlplm->nglm, prid, wdid));
            }
         }

         wdid = GetLabId(word, FALSE);
#if 0
         fprintf(stdout, "word : %s OOS prob normalization term: %e %e %e, prob: %e %e\n", word,
                 GetLMProb((LModel *)lm->data.mlplm->nglm, prid, wdid), ne->bowt,
                 exp(GetLMProb((LModel *)lm->data.mlplm->nglm, prid, wdid) - ne->bowt),
                 prob, prob * exp(GetLMProb((LModel *)lm->data.mlplm->nglm, prid, wdid) - ne->bowt));
         fflush(stdout);
#endif
         prob *= exp(GetLMProb((LModel *)lm->data.mlplm->nglm, prid, wdid) - ne->bowt);
      }
#endif
#endif

   }
   return prob;
}

/* GetMLPLMProb : computing MLP LM prob */
float GetMLPLMProb(LModel *lm, LabId prid[NSIZE], LabId wdid)
{
   int i = 0, hSize = 0;
   char **hist = NULL;
   NNLM *mlp = NULL;
   float prob = 0;

   mlp = (NNLM *)(lm->data.mlplm->mlp);

   hSize = ACTUAL_MLP_NSIZE - 1;
   hist = (char **) New(lm->heap, hSize * sizeof(char *));

   for (i=0; i<hSize; i++) {
      /* assigned zero prob to n-grams with truncated history contexts */
      if (!prid[hSize - i - 1]) {
         /* if need to pad <s> at the sentence start to make full span context */
         if (padStartWord) {
            prid[hSize - i - 1] = GetLabId("<s>", FALSE);
            if (!prid[hSize - i - 1]) {
               HError(999, "GetMLPLMProb : sentence start token <s> missing in word list");
            }
         }
         else {
#if 0
         fprintf(stdout, "\nassigning zero prob to n-grams with short history contexts : ");
#endif

         return LZERO;
         }
      }
      hist[i] = prid[hSize - i - 1]->name;
   }

   /*    CalNLMProb(char **in_seq, NNLM *NNlm, char *out, int Ngram); */

/* Defunct MLP prob output for OOV words unless performing
   MLP prob normalization for cases WITH OOS output node */
#ifndef MLPLMPROBNORM_OOS
   /* OOV words assigned zero prob */
   if (String2Index(wdid->name, mlp->outmap,
                    mlp->out_num_word) == mlp->out_num_word) {
#if 0
      fprintf(stdout, "\nassigning zero prob to OOV word %s : ", wdid->name);
#endif

      return LZERO;
   }
#endif

   /* query and update cache if necessary */
   prob = UpdateMLPLMCache(lm, hist, mlp, wdid->name, hSize+1, TRUE);

   Dispose(lm->heap, hist);

   return LOG_NATURAL(prob);
}

/* SetMLPLMVocab : setting the word list of MLP LM to that of
   other component ngram models */
void SetMLPLMVocab(LModel *lm)
{
   int i = 0;
   int nwords = 0;
   LabId *wdlist = NULL;
#ifdef MLPLMPROBNORM
   LModel *nglm = NULL;
#endif
   IntpltLM *ilang = NULL;
   RNLM *rnlm  = NULL;

   if (lm->type != intpltLM) {
      HError(999, "SetMLPLMVocab : Expecting intpltLM model of which MLP LM is a component");
   }

   ilang = lm->data.ilang;

#ifdef MLPLMPROBNORM
   if (ilang->nModels > 2) {
      HError(-999, "SetMLPLMVocab : Expecting 2 component intpltLM model of which MLP LM is a component");
   }
#else
#ifdef MLPLMPROBNORM_OOS
   HError(999, "SetMLPLMVocab : MLPLMPROBNORM and MLPLMPROBNORM_OOS must both be set to 1 for with OOS!!!");
#endif
#endif

   for (i=1; i<=ilang->nModels; i++) {
      if (((LModel *)ilang->lms[i])->type == boNGram) {
         wdlist = ((LModel *)ilang->lms[i])->data.ngram->wdlist;
         nwords = ((LModel *)ilang->lms[i])->data.ngram->vocSize;
#ifdef MLPLMPROBNORM
         nglm = (LModel *)ilang->lms[i];
#endif
         break;
      }
   }

   if (!wdlist) {
      HError(999, "SetMLPLMVocab : intpltLM model must contain at least one ngram model component");
   }

   if (nwords == 0)
   {
      HError(999, "SetMLPLMVocab : intpltLM model must contain at least one ngram model component, nwords should be larger than 0");
   }
   for (i=1; i<=ilang->nModels; i++) {
      if (((LModel *)ilang->lms[i])->type == mlpLM) {
         ((LModel *)ilang->lms[i])->data.mlplm->wdlist = wdlist;
#ifdef MLPLMPROBNORM
         ((LModel *)ilang->lms[i])->data.mlplm->nglm = (Ptr *)nglm;
         fprintf(stdout, "Normalizing MLP LM probs with back-off ngram statistics ...\n\n");
         fflush(stdout);
#ifdef MLPLMPROBNORM_OOS
         fprintf(stdout, "Assuming MLP LM has OOS output node ...\n\n");
         fflush(stdout);
#endif
#endif
      }
      if (((LModel *)ilang->lms[i])->type == rnnLM) {
         ((LModel *)ilang->lms[i])->data.rnlm->wdlist = wdlist;
         rnlm = ((LModel *)ilang->lms[i])->data.rnlm;
         ((RNNLM *)rnlm->rnnlm)->fulldict_size = nwords;
#ifdef MLPLMPROBNORM
         ((LModel *)ilang->lms[i])->data.rnlm->nglm = (Ptr *)nglm;
         fprintf(stdout, "Normalizing RNN LM probs with back-off ngram statistics ...\n\n");
         fflush(stdout);
#ifdef MLPLMPROBNORM_OOS
         fprintf(stdout, "Assuming RNN LM has OOS output node ...\n\n");
         fflush(stdout);
#endif
#endif
      }
   }
}

/* ResetMLPLMCache : reset ngram and LM state caches of MLP LM */
void ResetMLPLMCache(LModel *lm)
{
   int i = 0, counts[NSIZE] = {250001};
   MLPLM *mlplm = NULL;
   RNLM *rnlm = NULL;
   IntpltLM *ilang = NULL;

   if (lm->type != intpltLM && lm->type != mlpLM && lm->type != rnnLM) {
      return;
   }

   for (i=0; i<NSIZE; i++) {
      counts[i] = 250001;
   }

   if (lm->type == intpltLM) {
   ilang = lm->data.ilang;

   for (i=1; i<=ilang->nModels; i++) {
      if (((LModel *)ilang->lms[i])->type == mlpLM) {
         mlplm = ((LModel *)ilang->lms[i])->data.mlplm;
         /* ngram cache */
         ResetHeap( ((LModel *) mlplm->cache)->heap );
         ((LModel *) mlplm->cache)->data.ngram =
            CreateBoNGram(((LModel*) mlplm->cache), ((NNLM *)mlplm->mlp)->in_num_word, counts);
         /* LM state cache */
         ResetHeap( ((LModel *) mlplm->lmstate)->heap );
         ((LModel *) mlplm->lmstate)->data.ngram =
            CreateBoNGram(((LModel*) mlplm->lmstate), counts[0], counts);
      }
      if (((LModel *)ilang->lms[i])->type == rnnLM) {
         rnlm = ((LModel *)ilang->lms[i])->data.rnlm;
         /* ngram cache */
         ResetHeap( ((LModel *) rnlm->cache)->heap );
         ((LModel *) rnlm->cache)->data.ngram =
            CreateBoNGram(((LModel*) rnlm->cache), ((RNNLM *)rnlm->rnnlm)->in_num_word, counts);
         /* LM state cache */
         ResetHeap( ((LModel *) rnlm->lmstate)->heap );
         ((LModel *) rnlm->lmstate)->data.ngram =
            CreateBoNGram(((LModel*) rnlm->lmstate), counts[0], counts);
         /* reset RNNLM network */
         RNLMReset((LModel *) (LModel *)ilang->lms[i]);
      }
   }
   }

   if (lm->type == mlpLM) {
      mlplm = ((LModel *) lm)->data.mlplm;
      /* ngram cache */
      ResetHeap( ((LModel *) mlplm->cache)->heap );
      ((LModel *) mlplm->cache)->data.ngram =
         CreateBoNGram(((LModel*) mlplm->cache), ((NNLM *)mlplm->mlp)->in_num_word, counts);
      /* LM state cache */
      ResetHeap( ((LModel *) mlplm->lmstate)->heap );
      ((LModel *) mlplm->lmstate)->data.ngram =
         CreateBoNGram(((LModel*) mlplm->lmstate), counts[0], counts);
   }

   if(lm->type == rnnLM) {
      rnlm = ((LModel *) lm)->data.rnlm;
      /* ngram cache */
      ResetHeap( ((LModel *) rnlm->cache)->heap );
      ((LModel *) rnlm->cache)->data.ngram =
         CreateBoNGram(((LModel*) rnlm->cache), ((RNNLM *)rnlm->rnnlm)->in_num_word, counts);
      /* LM state cache */
      ResetHeap( ((LModel *) rnlm->lmstate)->heap );
      ((LModel *) rnlm->lmstate)->data.ngram =
         CreateBoNGram(((LModel*) rnlm->lmstate), counts[0], counts);
      /* reset RNNLM network */
      RNLMReset((LModel *) lm);
   }
}

/*--------------------- Interpolated LMs ------------------------*/

/* ReadIntpltLM: read and store interpolated LM */
static void ReadIntpltLM(LModel *lm, char *fn)
{
   int i = 0;
   char buf[MAXSTRLEN+1];
   IntpltLM *ilang = NULL;
   Source src;
   float lmprobmass = 0.0;

   if (trace&T_TIO)
      printf("\nIntpltLM : "),fflush(stdout);

   if (InitSource(fn, &src, LangModFilter) < SUCCESS)
      HError(8110,"ReadIntpltLM : Can't open file %s", fn);

   ilang = (IntpltLM *)New(lm->heap, sizeof(IntpltLM));
   lm->data.ilang = ilang;
   ilang->nModels = 0;
   ilang->weight = NULL; ilang->lms = NULL; ilang->wsi = NULL; ilang->RePtrHT = NULL;

   ilang->heap = lm->heap;

   if (!ReadString(&src, buf)) {
      HError(8110,"ReadIntpltLM : Can't read header from intpltLM file %s", fn);
   }
   if (strcmp(buf, "!INTERPOLATE") != 0) {
      HError(999, "ReadIntpltLM: expecting !INTERPOLATE but got %s\n", buf);
   }
   if (!ReadInt(&src, &ilang->nModels, 1, FALSE)) {
      HError(999, "ReadIntpltLM: Expecting number of component LMs in %s\n", fn);
   }

   if (trace & T_TIO) {
      fprintf (stdout, "%d component LMs in %s\n", ilang->nModels, fn);
      fflush(stdout);
   }

   if (ilang->nModels > MAX_LMODEL - 1) {
      HError(999, "number of component LMs %d exceeded MAX_LMODEL, recompile with a larger value");
   }

   ilang->weight = CreateVector(ilang->heap, ilang->nModels);
   ZeroVector(ilang->weight);

   ilang->lms = (Ptr **)New(ilang->heap, (ilang->nModels + 1) * sizeof(LModel *));

   for (i=1; i<=ilang->nModels; i++)
   {
      if (!ReadString(&src, buf)) {
         HError(8110,"ReadIntpltLM : Can't read %dth component LM type from intpltLM file %s", i, fn);
      }
      if (strcmp(buf, "!NGRAM") != 0 && strcmp(buf, "!MLP") != 0) {
         HError(999, "ReadIntpltLM: expecting !NGRAM or !MLP but got %s\n", buf);
      }
      if (!ReadFloat(&src, &ilang->weight[i], 1, FALSE))
      {
         HError(999, "ReadIntpltLM: Expecting %dth component LM weight in %s\n", i, fn);
      }
      if (!ReadString(&src, buf)) {
         HError(8110,"ReadIntpltLM: Can't read %dth component LM file name %s", i, fn);
      }
      if (trace & T_TIO) {
         fprintf (stdout, "\nReading component LM %d (weight %.10f) in %s :\n", i, ilang->weight[i], buf);
         fflush(stdout);
      }
      lmprobmass += ilang->weight[i];
      ilang->lms[i] = (Ptr *)ReadLModel(ilang->heap, buf, i);
   }

   float  restprobweight = 1 - lmprobmass;
   fprintf (stdout, "Weight assigned to prob from orig lattice: %.10f\n", restprobweight);

  /* setting the word list of MLP LM component models */
  SetMLPLMVocab(lm);
}


/* EXPORT GetLMProb: return probability of word wd_id following pr_id[] */
float GetLMProb(LModel *lm, LabId prid[NSIZE], LabId wdid, LabId succwdid);

float GetIntpltLMProb(LModel *lm, LabId prid[NSIZE], LabId wdid, LabId succwdid)
{
   int i = 0;
   LogFloat tmp_prob = 0, lmprob = 0;
   float wnorm = 0;
   Boolean defunc[MAX_LMODEL] = {0}, renorm = FALSE;
   IntpltLM *ilang = NULL;

   ilang = lm->data.ilang;

   /* sum over all component models and retain all destinations */
   for (i=1; i<=ilang->nModels; i++)
   {
      tmp_prob = exp( GetLMProb((LModel *)ilang->lms[i], prid, wdid, succwdid) );

      if ( ((LModel *)ilang->lms[i])->type == mlpLM || ((LModel *)ilang->lms[i])->type == rnnLM ) {
         /* defunct MLP component : OOV or truncated histrory contexts */
         if (tmp_prob <= exp(LZERO)) {
            defunc[i] = 1; renorm = TRUE;
         }
      }

      lmprob += ilang->weight[i] * tmp_prob;

      if (defunc[i] == 0) {
         wnorm += ilang->weight[i];
      }
   }

   /* re-normalize interpolated prob for cases when MLP components are defunct */
   /* for no OOS case this may need to be disabled when not being normalized */
   if (renorm) lmprob /= wnorm;

#if 1
   return LOG_NATURAL(lmprob);
#else
   return lmprob;
#endif
}


/*------------------------- User Interface --------------------*/

/* EXPORT GetLMProb: return probability of word wd_id following pr_id[] */
float GetLMProb(LModel *lm, LabId prid[NSIZE], LabId wdid, LabId succwdid)
{
   LabId cpid[NSIZE];
   NEntry *ne;
   SEntry *se;
   lmId p, q, word, ndx[NSIZE];
   LogFloat bowt,prob;
   int i, s;

   switch (lm->type) {
   case boNGram:
      if (!wdid) {
#if 0
         HError (-9999, "GetLMProb (ngram) : word %d not in LM wordlist", wdid);
#endif
         return (LZERO);
      }
      word = (int)wdid->aux;
      if (word==0 || word>lm->data.ngram->vocSize)
         return(LZERO);
      for (s=-1,i=0;i<NSIZE;i++)
         if (prid[i]!=NULL)
            ndx[i]=(int)prid[i]->aux, cpid[i]=prid[i], s=i;
         else
            ndx[i]=0, cpid[i]=NULL;

      /* If no answer back-off to unigram */
      if (s<0) {
         if (word!=0)
            return(lm->data.ngram->unigrams[word]);
         else
            return(log(1.0/lm->data.ngram->vocSize));
      }

      cpid[s]=0;
      ne = GetNEntry(lm->data.ngram,ndx,FALSE);
      if (ne) {
         /* Replace with bsearch equivalent */
         for (i=0, se=ne->se; i<ne->nse; i++,se++)
            if (se->word==word)
               return(se->prob); /* Ngram found */
         bowt=ne->bowt;
      }
      else {
         bowt=0.0;
      }

      if (s==0)
         return(lm->data.ngram->unigrams[word]+bowt); /* Backoff to unigram */
      else
         return(bowt+GetLMProb(lm,cpid,wdid, NULL)); /* else recurse */
      break;
   case matBigram:
      p=(int) prid[0]->aux;
      q=(int) wdid->aux;
      return(lm->data.matbi->bigMat[p][q]);
   case mlpLM:
      return GetMLPLMProb(lm, prid, wdid);
   case rnnLM:
      return GetRNNLMProb(lm, prid, wdid, succwdid);
   case intpltLM:
      return GetIntpltLMProb(lm, prid, wdid, succwdid);
   default:
      prob=LZERO;
   }
   return(prob);
}

/* EXPORT ReadLModel: Determine LM type and then read-in */
LModel *ReadLModel(MemHeap *heap,char *fn, int iditer)
{
   LModel *lm;
   LMType type;
   char buf[MAXSTRLEN+1];
   int i;
   Boolean usefrnnlm = FALSE;
   Boolean usecuedrnnlm = FALSE;
   float cuedversion = 1.0;

   lm=(LModel*)New(heap,sizeof(LModel));
   lm->heap=heap;
   lm->name=CopyString(heap,fn);
   lm->ppinfo = NULL;
   lm->id = iditer;

   if(InitSource(fn,&source,LangModFilter)<SUCCESS)
      HError(8110,"ReadLModel: Can't open file %s", fn);
   type=boNGram;i=0;
   do {
      if (i++==1000) {
         type=matBigram;
         break;
      }
      GetInLine(buf);
      if (strcmp(buf, "!INTERPOLATE") == 0) {
         type = intpltLM;
         useIntpltLM = TRUE;
         break;
      }
      if (strcmp(buf, "!MLP") == 0) {
         type = mlpLM;
         break;
      }
      if (strcmp(buf, "!RNN") == 0) {    /* default RNNLM is C-RNNLM */
         type = rnnLM;
         break;
      }
      else if (strcmp(buf, "!FRNN") == 0) {   /* F-RNNLM is also supported */
         usefrnnlm = TRUE;
         type = rnnLM;
         break;
      }
      else if (strcmp(buf, "!CUEDRNN") == 0) { /* CUEDRNNLM is supported */
          usecuedrnnlm = TRUE;
          cuedversion  = 0.1;
          type = rnnLM;
          break;
      }
      else if (strcmp(buf, "!CUEDRNNv1.0") == 0) { /* CUEDRNNLM is supported */
          usecuedrnnlm = TRUE;
          cuedversion = 1.0;
          type = rnnLM;
          break;
      }
      else if (strcmp(buf, "!CUEDRNNv1.1") == 0) { /* CUEDRNNLM is supported */
          usecuedrnnlm = TRUE;
          cuedversion = 1.1;
          type = rnnLM;
          break;
      }

   }
   while (strcmp(buf, "\\data\\")!=0);
   CloseSource(&source);

   lm->type=type;
   switch(type) {
   case boNGram:
      ReadBoNGram(lm,fn);
      break;
   case matBigram:
      ReadMatBigram(lm,fn);
      break;
   case mlpLM:
      ReadMLPLM(lm,fn);
      break;
   case rnnLM:
      ReadRNLM(lm, fn, usefrnnlm, usecuedrnnlm, cuedversion);
      break;
   case intpltLM:
      ReadIntpltLM(lm,fn);
      break;
   }
   return(lm);
}


/* EXPORT WriteLModel: Determine LM type and then write-out */
void WriteLModel(LModel *lm,char *fn,int flags)
{
   switch(lm->type) {
   case boNGram:
      WriteBoNGram(lm,fn,flags);
      break;
   case matBigram:
      WriteMatBigram(lm,fn,flags);
      break;
   }
}

void ClearLModel(LModel *lm)
{
   switch(lm->type) {
   case boNGram:
      ClearBoNGram(lm);
      break;
   case matBigram:
   case intpltLM:
      break;
   }
}

/*----------------------------------------------------------------------*/

#ifndef NO_LAT_LM
/* FindSEntry

     find SEntry for wordId in array using binary search
*/
static SEntry *FindSEntry (SEntry *se, lmId pronId, int l, int h)
{
   /*#### here l,h,c must be signed */
   int c;

   while (l <= h) {
      c = (l + h) / 2;
      if (se[c].word == pronId)
         return &se[c];
      else if (se[c].word < pronId)
         l = c + 1;
      else
         h = c - 1;
   }

   return NULL;
}

/* LMTransProb_ngram

     return logprob of transition from src labelled word. Also return dest state.
     ngram case
*/
LogFloat LMTrans_NGram (LModel *lm, LMState src, LabId wdid, LMState *dest)
{
   NGramLM *nglm;
   LogFloat lmprob;
   lmId hist[NSIZE] = {0};      /* initialise whole array to zero! */
   int i, l;
   NEntry *ne;
   SEntry *se;
   lmId word;

   assert (lm->type == boNGram);
   nglm = lm->data.ngram;

   word = (int) wdid->aux;

   if (word==0 || word>lm->data.ngram->vocSize) {
      HError (-9999, "word %d (%s) not in LM wordlist", word, wdid->name);
      *dest = NULL;
      return (LZERO);
   }

   ne = src;

   if (!src) {          /* unigram case */
      lmprob = nglm->unigrams[word];
   }
   else {
      /* lookup prob p(word | src) */
      /* try to find pronid in SEntry array */
      se = FindSEntry (ne->se, word, 0, ne->nse - 1);

      assert (!se || (se->word == word));

      if (se)        /* found */
         lmprob = se->prob;
      else {             /* not found */
         lmprob = 0.0;
         l = 0;
         hist[NSIZE-1] = 0;
         for (i = 0; i < NSIZE-1; ++i) {
            hist[i] = ne->word[i];
            if (hist[i] != 0)
               l = i;
         } /* l is now the index of the last (oldest) non zero element */

         for ( ; l >= 0; --l) {
            if (ne)
               lmprob += ne->bowt;
            hist[l] = 0;   /* back-off: discard oldest word */
            ne = GetNEntry (nglm, hist, FALSE);
            if (ne) {   /* skip over non existing hists. fix for weird LMs */
               /* try to find pronid in SEntry array */
               se = FindSEntry (ne->se, word, 0, ne->nse - 1);
               assert (!se || (se->word == word));
               if (se) { /* found it */
                  lmprob += se->prob;
                  l = -1;
                  break;
               }
            }
         }
         if (l == 0) {          /* backed-off all the way to unigram */
            assert (!se);
            lmprob += ne->bowt;
            lmprob += nglm->unigrams[word];
         }
      }
   }


   /* now determine dest state */
   if (src) {
      ne = (NEntry *) src;

      l = 0;
      hist[NSIZE-1] = 0;
      for (i = 1; i < NSIZE-1; ++i) {
         hist[i] = ne->word[i-1];
         if (hist[i] != 0)
            l = i;
      } /* l is now the index of the last (oldest) non zero element */
   }
   else {
      for (i = 1; i < NSIZE-1; ++i)
         hist[i] = 0;
      l = 1;
   }

   hist[0] = word;

   ne = (LMState) GetNEntry (nglm, hist, FALSE);
   for ( ; !ne && (l > 0); --l) {
      hist[l] = 0;              /* back off */
      ne = (LMState) GetNEntry (nglm, hist, FALSE);
   }
   /* if we left the loop because l=0, then ne is still NULL, which is what we want */

   *dest = ne;

#if 0
   printf ("lmprob = %f  dest %p\n", lmprob, *dest);
#endif

   return (lmprob);
}


/* LMTransProb_mlplm

     return logprob of transition from src labelled word. Also return dest state.
     mlplm case
*/
LogFloat LMTrans_MLPLM (LModel *lm, LMState src, LabId wdid, LMState *dest)
{
   int i = 0, hSize = 0;
   lmId ndx[MLP_NSIZE] = {0};
   LabId prid[MLP_NSIZE] = {0};
   NEntry *ne = NULL;
   NGramLM *nglm = NULL;
   float prob = 0;

   hSize = ACTUAL_MLP_NSIZE - 1;

   /* truncated previous MLP LM history context - remove last word */
   if (src) {
      for (i=0; i<hSize; i++) {
         if (((NEntry *)src)->word[i] != 0) {
            ndx[i+1] = ((NEntry *)src)->word[i];
         }
      }
   }
   /* current word to predict, now becomes first word in history */
   ndx[0] = (int) wdid->aux;

   /* query in LM state cache  */
   nglm = ((LModel *)lm->data.mlplm->lmstate)->data.ngram;
   ne = GetNEntry2(nglm, ndx, FALSE, ACTUAL_MLP_NSIZE);

   /* if no LM state of matching context found then add to cache */
   if (!ne) {

#if 1
      if (nglm->heap->totAlloc > 268435456) {
         int counts[NSIZE] = {250001};
         ResetHeap(nglm->heap);
         nglm = CreateBoNGram((LModel *)lm->data.mlplm->lmstate, counts[0], counts);
      }
#endif

      ne = GetNEntry2(nglm, ndx, TRUE, ACTUAL_MLP_NSIZE);
   }
   /* MLP LM state is always full context span */
   *dest = (LMState) ne;

   if (src) {
      for (i=0; i<ACTUAL_MLP_NSIZE-1; i++) {
         if (((NEntry *)src)->word[i] != 0) {
            prid[i] = lm->data.mlplm->wdlist[((NEntry *)src)->word[i]];
         }
      }
   }
   prob = GetMLPLMProb(lm, prid, wdid);

   return prob;
}


/* LMTrans_IntpltLM

     return logprob of transition from src labelled word. Also return dest state.
     IntpltLM case
*/
LogFloat LMTrans_IntpltLM (LModel *lm, LMState src, LabId wdid, LMState *dest, LMState *src_mix, LMState *dest_mix, LabId *succwordId, LogFloat lmlike)
{
   int i = 0, j = 0, l = 0, max = 0;
   LogFloat tmp_prob = 0, lmprob = 0;
   float wnorm = 0;
   Boolean defunc[MAX_LMODEL] = {0}, renorm = FALSE;
   float prob_mix[MAX_LMODEL] = {0};
   IntpltLM *ilang = NULL;
   NEntry **com_dest = NULL;

   ilang = lm->data.ilang;

   /* destination for each component model */
   com_dest = (NEntry **) New(lm->heap, (ilang->nModels + 1) * sizeof(NEntry *));

   for (i=1; i<=ilang->nModels; i++)
   {
       tmp_prob += ilang->weight[i];
   }
   /* BEGIN modification */
   wnorm= 1.0-tmp_prob;
   /* END modification */
   lmprob = (1 - tmp_prob)*exp(lmlike);
   tmp_prob = 0;

   Boolean usesurnnlm = FALSE;


   /* sum over all component models and retain all destinations */
   for (i=1; i<=ilang->nModels; i++) {
      if ( ((LModel *)ilang->lms[i])->type == boNGram ) {
         tmp_prob = exp( LMTrans_NGram ((LModel *)ilang->lms[i], src_mix[i], wdid, (LMState *) &dest_mix[i]) );
      }
      else {
         if ( ((LModel *)ilang->lms[i])->type == mlpLM ) {
            tmp_prob = exp( LMTrans_MLPLM ((LModel *)ilang->lms[i], src_mix[i], wdid, (LMState *) &dest_mix[i]) );
         }
         if ( ((LModel *)ilang->lms[i])->type == rnnLM ) {
            tmp_prob = exp( LMTrans_RNNLM ((LModel *)ilang->lms[i], src_mix[i], wdid, (LMState *) &dest_mix[i], succwordId));
            RNNLM *rnnlm = (RNNLM *)(((LModel *)ilang->lms[i])->data.rnlm->rnnlm);
            if (rnnlm->usecuedsurnnlm)
            {
                usesurnnlm = TRUE;
            }
         }
         /* defunct MLP/RNN component : OOV or truncated histrory contexts */
         if (tmp_prob <= exp(LZERO)) {
            defunc[i] = 1; renorm = TRUE;
         }
         
        
      }
      /* lmprob += ilang->weight[i] * tmp_prob; */
      prob_mix[i] = tmp_prob;

      if (defunc[i] == 0) {
         wnorm += ilang->weight[i];
      }

      com_dest[i] = (NEntry *) dest_mix[i];
#if 0
      fprintf(stdout, "P(src -> %s) : Com %d wgt = %e prob = %e logprob = %e dest = %p\n",
              wdid->name, i, ilang->weight[i], tmp_prob, log(tmp_prob), com_dest[i]);
#endif
   }


   if (usesurnnlm)
   {
       /* the su-RNNLM should always be the first component
        * 3 models in total
        * linear interpolation used for uni-LMs
        * log-linear interpolation used between su-RNNLMs and uni-LMs
        */
       tmp_prob = ilang->weight[2]*prob_mix[2] + ilang->weight[3]*prob_mix[3] + lmprob;
       tmp_prob = (1-ilang->weight[1])*LOG_NATURAL(tmp_prob) + LOG_NATURAL(prob_mix[1])*ilang->weight[1];
       lmprob = exp(tmp_prob);
   }
   else     /* linear interpolation used */
   {
       tmp_prob = lmprob;
       for (i=1; i<=ilang->nModels; i++)
       {
           tmp_prob += prob_mix[i] * ilang->weight[i];
       }
       lmprob = tmp_prob;

   }


   /* re-normalize interpolated prob for cases when MLP components are defunct */
   if (renorm) lmprob /= wnorm;
   /* BEGIN modification */
   if( lmprob <= 0.0 ) lmprob= 1.0; /* I think this only happens in
				       very special cases like
				       transitions to <s>. */
   /* END modification */
   
   /* find longest destination for linear model interpolation (union) */
   for (i=1, l=1, max=0; i<=ilang->nModels; i++)
   {
      if (com_dest[i] && defunc[i] == 0)
      {
          for (j=1; j<=NSIZE-2; j++)
          {
            if (com_dest[i]->word[j] == 0) break;
         }
      }
      else j = 1;
      if (j > max)
      {
         max = j; l = i;
      }
#if 0
      fprintf(stdout, "P(src -> %s) : Com %d destlen = %d\n", wdid->name, i, j);
#endif
   }

#if 0
   fprintf(stdout, "P(src -> %s) : intprob = %e intlogprob = %e max = %d l = %d dest = %p\n",
           wdid->name, lmprob, log(lmprob), max, l, com_dest[l]);
#endif

   *dest = com_dest[l];

   Dispose(lm->heap, com_dest);

/*    return log(lmprob); */
   return LOG_NATURAL(lmprob);
}

/* LMTrans

     return logprob of transition from src labelled word. Also return dest state.
     general case as a wrap-up of different LM types
*/
LogFloat LMTrans (LModel *lm, LMState src, LabId wdid, LMState *dest, LMState *src_mix, LMState *dest_mix, LabId *succwordId, LogFloat lmlike)
{
   switch (lm->type) {
   case boNGram:
      return LMTrans_NGram (lm, src, wdid, dest);
   case matBigram:
      HError(999, "LMTrans: matBigram not supported in LMTrans() !");
   case mlpLM:
      return LMTrans_MLPLM (lm, src, wdid, dest);
   case rnnLM:
      return LMTrans_RNNLM (lm, src, wdid, dest, succwordId);
   case intpltLM:
      return LMTrans_IntpltLM (lm, src, wdid, dest, src_mix, dest_mix, succwordId, lmlike);
   default:
      HError(-999, "LMTrans: unknown LM type !");
      return LZERO;
   }
}

#endif


/* ------------------------- End of HLM.c ------------------------- */

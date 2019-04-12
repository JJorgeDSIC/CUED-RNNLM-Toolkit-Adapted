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


#define L10ZERO       -99.9900
#define L10MINARG     +1.0E-20
#define EXP10MINARG   -20.0

#define LOG10_TO_FLT(x) \
   (((x) < EXP10MINARG) ? 0.0 : exp(LN10*(x)))

#define FLT_TO_LOG10(x) \
   (((x) < L10MINARG) ? L10ZERO : log10(x))

/* EXPORT->GetNEntry2_RNNLM: Access specific NGram entry indexed by ndx and stores
   an associated given current and future history vector */
NEntry *GetNEntry2_RNNLM(NGramLM *nglm,lmId ndx[NSIZE], lmId fndx[NSIZE], Boolean use_prob_table, Boolean create, int ngsize, Vector v, Vector fv);
NEntry *GetNEntry2_RNNLM_ProbCache(NGramLM *nglm,lmId ndx[NSIZE],Boolean create, int ngsize, Vector v, Vector fv);
NEntry *GetNEntry2_RNNLM_Lmstate(NGramLM *nglm,lmId ndx[NSIZE],Boolean create, int ngsize, Vector v, Vector fv);

NEntry *GetNEntry2_RNNLM_HVDist(NGramLM *nglm,lmId ndx[NSIZE],Boolean create, int ngsize, Vector v, Vector fv);


/*--------------------- RNN LMs ------------------------*/

void RNLMInit (LModel *lm)
{
   if (lm->type == rnnLM) {
      RNNLM* rnnlm = (RNNLM*)(lm->data.rnlm->rnnlm);
      RNNLMStart (rnnlm);
      lm->data.rnlm->curlmste = NULL;
   }
}
void RNLMReset (LModel *lm)
{
   if (lm->type == rnnLM) {
      RNNLM* rnnlm = (RNNLM*)(lm->data.rnlm->rnnlm);
      RNNLMEnd (rnnlm);
      lm->data.rnlm->curlmste = NULL;
      ZeroVector(lm->data.rnlm->rnnlm_hist);
      ZeroVector(lm->data.rnlm->rnnlm_fhist);
   }
}

static void ReadRNLM(LModel* lm, char* fn, Boolean usefrnnlm, Boolean usecuedrnnlm, float cuedversion)
{
   int i = 0, k = 0, inVocSize = 0, outVocSize = 0;
   int counts[NSIZE] = {0};
   Source src;
   char buf[MAXSTRLEN+1];
   char *rnnlmfn = NULL;   /*  file in Mikolov's original format   */
   char* inmapfn = NULL;   /*  file used to map word (input vocab) to lmId */
   char* outmapfn = NULL;  /*  file used to map word (output vocab) to lmId */
   RNLM* rnlm= NULL;       /*  defined in HLM.h */

   if (InitSource(fn, &src, LangModFilter) < SUCCESS )
      HError(8110, "ReadRNLM: Cannot open file %s", fn);

   if (trace&T_TIO)
      printf("\nRNLM \n"),fflush(stdout);

   rnlm = (RNLM*) New(lm->heap, sizeof(RNLM));
   lm->data.rnlm = rnlm;

   rnlm->wdlist = NULL;
   rnlm->cache = NULL;
   rnlm->lmstate = NULL;
   rnlm->curlmste = NULL;
   rnlm->heap = lm->heap;

   if (!ReadString(&src, buf)) {
      HError(8110, "ReadRNLM: Can't read header from RNNLM file %s", fn);
   }
   if (strcmp(buf, "!RNN") != 0 && strcmp(buf, "!FRNN") != 0 && strcmp(buf, "!CUEDRNN")!= 0 && strcmp(buf, "!CUEDRNNv1.0")!=0  && strcmp(buf, "!CUEDRNNv1.1") != 0){
      HError(8150, "ReadRNLM: expecting !RNN or !FRNN or !CUEDRNN or !CUEDRNNv1.0 or !CUEDRNNv1.1 but got %s\n", buf);
   }

   /*-----------------------------------------------------------------------------
    *  read in rnnlmfn
    *-----------------------------------------------------------------------------*/
   if ( ReadString(&src, buf) != TRUE )
      {
         HError(8150, "ReadRNLM: Cann't read original RNNLM-format file name from RNLM %s", fn);
      }
   rnnlmfn = (char*) New(&gstack, MAXSTRLEN+1);
   rnnlmfn = strcpy(rnnlmfn, buf);
   if (trace  & T_TIO )
      {
         fprintf(stdout, "RNN model in original format: %s\n", rnnlmfn);
         fflush(stdout);
      }


   /*-----------------------------------------------------------------------------
    *  read in inmapfn
    *-----------------------------------------------------------------------------*/
   if ( !ReadString(&src, buf)) {
      HError(8150, "ReadRNLM: Cann't read input map file name from  RNLM %s", fn);
   }
   inmapfn = (char* )New(&gstack, MAXSTRLEN+1);
   inmapfn = strcpy(inmapfn, buf);
   if (trace  & T_TIO ) {
      fprintf(stdout, "input vocab file: %s\n", inmapfn);
      fflush(stdout);
   }

   /*-----------------------------------------------------------------------------
    *  read in outmapfn
    *-----------------------------------------------------------------------------*/
   if ( !ReadString(&src, buf)) {
      HError(8150, "ReadRNLM: Cann't read output map file name from  RNLM %s", fn);
   }
   outmapfn = (char* )New(&gstack, MAXSTRLEN+1);
   outmapfn = strcpy(outmapfn, buf);
   if (trace  & T_TIO ) {
      fprintf(stdout, "output vocab file: %s\n", outmapfn);
      fflush(stdout);
   }


   /*-----------------------------------------------------------------------------
    *  read in ninvoc and noutvoc
    *-----------------------------------------------------------------------------*/
   k = 0;
   GetInLine(buf);
   if ((sscanf(buf, "%d", &k) != 1) || k==0) {
      HError(8150, "ReadRNLM : RNLM input vocab size missing (%s)", buf);
   }
   inVocSize = k;
   if ( trace & T_TIO) {
      fprintf(stdout, "intput layer vocab size: %d\n", inVocSize);
      fflush(stdout);
   }

   k = 0;
   GetInLine(buf);
   if ((sscanf(buf, "%d", &k) != 1) || k==0) {
      HError(8150, "ReadRNLM : RNLM output vocab size missing (%s)", buf);
   }
   outVocSize = k;
   if ( trace  & T_TIO )  {
      fprintf(stdout, "output layer vocab size: %d\n", outVocSize);
      fflush(stdout);
   }

   rnlm->rnnlm = (Ptr) CreateRNNLM(rnlm->heap);
   if (usecuedrnnlm)
   {
       if (fabs(cuedversion-1.1) < 1e-6)
       {
           LoadCUEDRNLMwgt_v1_1 (rnnlmfn, inmapfn, outmapfn, inVocSize, outVocSize, (RNNLM *)rnlm->rnnlm);
       }
       else if (cuedversion == 1.0)
       {
           LoadCUEDRNLMwgt_v1_0 (rnnlmfn, inmapfn, outmapfn, inVocSize, outVocSize, (RNNLM *)rnlm->rnnlm);
       }
       else         /* default version: 0.1 */
       {
           LoadCUEDRNLMwgt (rnnlmfn, inmapfn, outmapfn, inVocSize, outVocSize, (RNNLM *)rnlm->rnnlm);
       }
   }
   else if (usefrnnlm) {
      LoadFRNLMwgt (rnnlmfn, inmapfn, outmapfn, inVocSize, outVocSize, (RNNLM *)rnlm->rnnlm);
   }
   else {
      LoadRNLMwgt(rnnlmfn, inmapfn, outmapfn, inVocSize, outVocSize, (RNNLM*) rnlm->rnnlm);
   }

   /* various error checking */
   if (((RNNLM*) rnlm->rnnlm)->independent == 0) {
      HError(8150, "HLM: rnnlm->independent = 0, only supporting RNNLMs trained under sentence independent mode !!!");
   }
   if (padStartWord == FALSE) {
      padStartWord = TRUE;
      HError(-8150, "HLM: padStartWord == FALSE, forced to TRUE for RNNLMs !!!");
   }
   if (origMLPNSize > 0) {
      ACTUAL_MLP_NSIZE = origMLPNSize;
      if (ACTUAL_MLP_NSIZE > MLP_NSIZE)
      {
         HError(8150, "ReadRNNLM : MLP/RNN order %d beyond max size %d\n",
                ACTUAL_MLP_NSIZE, MLP_NSIZE);
      }
      /* fprintf(stdout, "MLP/RNN order: %d\n", ACTUAL_MLP_NSIZE); */
      fflush(stdout);
   }

   /* intialize RNNLM network */
   RNLMInit (lm);

   /* create ngram cache model */
   rnlm->cache = (Ptr *) New(lm->heap, sizeof(LModel));
   ((LModel*) rnlm->cache)->type = boNGram;
   /* Must use only heap to avoid hash table corruption !!! */
   ((LModel*) rnlm->cache)->heap = &mlpLMCacheInfoStack[lm->id];
   for (i=0; i<NSIZE; i++) {
      counts[i] = 250001;
   }
   fprintf(stdout, "Creating %d-gram cache for %d words ...\n", ACTUAL_MLP_NSIZE, outVocSize);
   fflush(stdout);
   ResetHeap(&mlpLMCacheInfoStack[lm->id]);
   ((LModel*) rnlm->cache)->data.ngram = CreateBoNGram(((LModel*) rnlm->cache), inVocSize, counts);

   /* create LM state cache */
   rnlm->lmstate = (Ptr *) New(lm->heap, sizeof(LModel));
   ((LModel*) rnlm->lmstate)->type = boNGram;
   /* Must use only heap to avoid hash table corruption !!! */
   ((LModel*) rnlm->lmstate)->heap = &mlpLMStateInfoStack;
   fprintf(stdout, "Creating LM state cache ...\n\n");
   fflush(stdout);
   ((LModel*) rnlm->lmstate)->data.ngram = CreateBoNGram(((LModel*) rnlm->lmstate), counts[0], counts);

   /* model level tracking of current RNNLM state */
   rnlm->rnnlm_hist = CreateVector(lm->heap, GetRNNLMHiddenVectorSize((RNNLM*) rnlm->rnnlm));
   rnlm->rnnlm_fhist = CreateVector(lm->heap, GetRNNLMHiddenVectorSize((RNNLM*) rnlm->rnnlm));
   ZeroVector(rnlm->rnnlm_hist);
   ZeroVector(rnlm->rnnlm_fhist);

   if (rnnlmUseHVDist == TRUE) {
      fprintf(stdout, "Effective RNN LM state size: %d-gram, hidden vector distance cut-off: %f\n",
              rnnlmLMStateSize, rnnlmMinHVDist);
      fflush(stdout);
   }

   Dispose(&gstack, outmapfn);
   Dispose(&gstack, inmapfn);
   Dispose(&gstack, rnnlmfn);
}

/* UpdateRNNLMCache : query and update RNN LM ngram cache */
float UpdateRNNLMCache(LModel *lm, char **hist, RNNLM *rnnlm, int wdID, char *word, int N, int shift, Boolean useCache, char succword[NSIZE][100], int succwdID[NSIZE])
{
    int succwdID_new[NSIZE];
   int i = 0, hSize = 0, hSize2 = 0, prID = 0, wdID_prev = 0;
   float prob = 0, prob_prev = 0;
   LabId wdid = NULL;
   lmId ndx[MLP_NSIZE] = {0};
   lmId fndx[NSIZE] = {0};
   Vector v = NULL, fv = NULL;
   NEntry *ne = NULL, *ne_prev = NULL;
   SEntry *se = NULL, *cse = NULL;
   NGramLM *nglm = NULL;
   char succword_new[NSIZE][100];

#ifdef MLPLMPROBNORM_OOS
   int j = 0; float tmp_norm = LZERO; LabId prid[MLP_NSIZE] = {0};
#endif

   hSize = ACTUAL_MLP_NSIZE - 1;
/*    hSize2 = ACTUAL_MLP_NSIZE; */

   if (MLP_NSIZE != NSIZE) {
      HError(999, "NSIZE (%d) and MLP_NSIZE (%d) need to be set equal !\n", NSIZE, MLP_NSIZE);
   }

   if (MLP_NSIZE - shift <= 0) {
      HError(999, "Actual history length reduced to zero, NSIZE/MLP_NSIZE = %d shift = %d, recompile with larger NSIZE !!!", MLP_NSIZE, shift);
   }
   hSize2 = MLP_NSIZE - shift;

   /* current word to predict */
   wdid = GetLabId(word, FALSE);
   if (!wdid) {
      HError(999, "UpdateRNNLMCache : Not expecting to find OOV word in cache : %s", word);
   }

   if (!useCache) {
      prID = RNNLMInVocabMap(rnnlm, hist[hSize2 - 1]);
      if (rnnlm->usecuedrnnlm)
      {
          if (rnnlm->cuedversion == 1.1)
          {
              if (rnnlm->usecuedsurnnlm)
              {
                  prob = CUEDRNNLMAcceptWord_v1_1_SU (rnnlm, prID, wdID, succwdID);
              }
              else
              {
                  prob = CUEDRNNLMAcceptWord_v1_1 (rnnlm, prID, wdID);
              }
          }
          if (rnnlm->cuedversion == 1.0)
          {
              prob = CUEDRNNLMAcceptWord_v1_0 (rnnlm, prID, wdID);
          }
          else
          {
              prob = CUEDRNNLMAcceptWord (rnnlm, prID, wdID);
          }
      }
      else if (rnnlm->usefrnnlm) {
         prob = FRNNLMAcceptWord (rnnlm, prID, wdID);
      }
      else {
         prob = RNNLMAcceptWord(rnnlm, prID, wdID);
      }
      /* prob is LOG10 based !!! */
      prob = LOG10_TO_FLT(prob);
#ifdef MLPLMPROBNORM_OOS
      tmp_norm = LZERO;
      if (wdID == rnnlm->out_oos_nodeid) {
         for (i=1; i<=((LModel *)lm->data.rnlm->nglm)->data.ngram->vocSize; i++) {
            wdid = ((LModel *)lm->data.rnlm->nglm)->data.ngram->wdlist[i];
            /* only accumulate \sum_{w} P_ng(w|h) for OOS words */
            if (RNNLMOutVocabMap(rnnlm, wdid->name) == rnnlm->out_oos_nodeid) {
               tmp_norm = LAdd(tmp_norm, GetLMProb((LModel *)lm->data.rnlm->nglm, prid, wdid));
            }
         }
         wdid = GetLabId(word, FALSE);
         prob *= exp(GetLMProb((LModel *)lm->data.rnlm->nglm, prid, wdid) - tmp_norm);
      }
#endif
      /* current word becomes part of future history */
      rnnlm->history[0] = RNNLMInVocabMap(rnnlm, wdid->name);
      return prob;
   }

   /* RNN LM history context */
   for (i=0; i<hSize2; i++) {
      if (hist[i] != NULL) {
         ndx[hSize2 - i] = RNNLMInVocabMap(rnnlm, hist[i]);
      }
   }
   ndx[0] = wdID;
   for (i=1; i<=3; i++)
   {
       fndx[i] = succwdID[i-1];
   }

#if 0
   fprintf(stdout, "Processing ngram : P(");
   for (j=0; j<1; j++) {
      fprintf(stdout, "(%s %d) ", hist[j], ndx[hSize2 - j]);
   }
   for (j=1; j<hSize2; j++) {
      fprintf(stdout, "%s %d ", hist[j], ndx[hSize2 - j]);
   }
   fprintf(stdout, "-> %s %d)\n", word, ndx[0]);
#endif



   v = CreateVector(&gstack, GetRNNLMHiddenVectorSize(rnnlm));
   fv = CreateVector(&gstack, GetRNNLMHiddenVectorSize(rnnlm));

   /* query in cache ngram model */
   nglm = ((LModel *)lm->data.rnlm->cache)->data.ngram;
#ifdef MLPLMPROBNORM_OOS
   if (rnnlmUseHVDist == FALSE)
   {
       ne = GetNEntry2_RNNLM(nglm, ndx+1, fndx+1, TRUE, FALSE, ACTUAL_MLP_NSIZE, v, fv);
       /* RNN LM previous history context - shift backwards by one word */
       ne_prev = GetNEntry2_RNNLM(nglm, ndx+2, -1, FALSE, FALSE, ACTUAL_MLP_NSIZE, v, fv);
   }
   else
   {
      /* initial node reset RNNLM */
      if (strcmp(hist[hSize2 - 1], "<s>") == 0)
      {
         RNLMReset(lm);
      }
      if (lm->data.rnlm->curlmste != NULL)
      {
         /* using weakened truncated contexts and history vector distance */
         /* RNNLM future history vector - now becomes current history for LM state */
         CopyVector(lm->data.rnlm->rnnlm_fhist, v);
         ne = GetNEntry2_RNNLM_HVDist(nglm, ndx+1, FALSE, rnnlmLMStateSize, v, fv);
         /* previous ngram - efficient for PPlex on non-lattices single sequence */
         ne_prev = lm->data.rnlm->curlmste;
      }
  }
#else
   ne = GetNEntry2_RNNLM(nglm, ndx+1, fndx+1, TRUE, FALSE, ACTUAL_MLP_NSIZE, v, fv);
   /* RNN LM previous history context - shift backwards by one word */
   ne_prev = GetNEntry2_RNNLM(nglm, ndx+2, -1, FALSE, FALSE, ACTUAL_MLP_NSIZE, v, fv);
   if (strcmp(hist[hSize2 - 1], "<s>") == 0)
   {
       ne_prev = NULL;
   }

#endif
   /* if ngrams of matching history found */
   if (ne && ne->nse == rnnlm->out_num_word) {
#if 0
      fprintf(stdout, "Found ngrams in cache : P(");
      for (j=1; j<hSize2; j++) {
         fprintf(stdout, "%s ", hist[j]);
      }
      fprintf(stdout, "-> *)\n");
#endif
      se = ne->se + wdID;
      prob = se->prob;
      /* current word becomes part of future history */
      rnnlm->history[0] = RNNLMInVocabMap(rnnlm, wdid->name);
      /* re-assign current RNNLM recurrent history */
      AssignRNNLMHiddenVector(rnnlm, ne->rnnlm_hist);
      copyHiddenLayerToInput(rnnlm);
      /* RNNLM model level tracking of current LM state */
      lm->data.rnlm->curlmste = ne;
      CopyVector(ne->rnnlm_hist, lm->data.rnlm->rnnlm_hist);
      CopyVector(ne->rnnlm_fhist, lm->data.rnlm->rnnlm_fhist);
#ifdef MLPLMPROBNORM
/* MLP/RNN prob normalization for cases WITHOUT OOS output node */
#ifndef MLPLMPROBNORM_OOS
      HError(999, "RNNLM prob normalization for cases WITHOUT OOS output node not supported !!!");
/* MLP/RNN prob normalization for cases WITH OOS output node */
#else
      if (wdID == rnnlm->out_oos_nodeid) {
         if (ne->bowt > LZERO) {
	    LabId prid[MLP_NSIZE] = {0};
            /* MLP/RNN LM history context */
            for (i=0; i<hSize; i++) {
               prid[hSize - i - 1] = GetLabId(hist[i], FALSE);
            }
            wdid = GetLabId(word, FALSE);
            prob *= exp(GetLMProb((LModel *)lm->data.rnlm->nglm, prid, wdid) - ne->bowt);
         }
      }
#endif
#endif
   }
   /* otherwise cache ngrams */
   else {
      /* if not at the start/initial node of the lattice */
      if ((LModel *)lm->data.rnlm->curlmste != NULL)
      {
#if 0
         fprintf(stdout, "Looking up preceding ngram : P(");
         for (j=0; j<hSize2-1; j++) {
            fprintf(stdout, "%s %d ", hist[j], ndx[hSize2 - j]);
         }
         fprintf(stdout, "-> %s)\n", hist[hSize2 - 1]);
#endif
         /* found preceding ngram and set RNNLM */
         if (ne_prev && ne_prev->nse == rnnlm->out_num_word)
         {
            /* re-assign current RNNLM history - history's future becomes present */
            AssignRNNLMHiddenVector(rnnlm, ne_prev->rnnlm_fhist);
         }
        /* no preceding ngram found */
         else
         {
            /* initial node reset RNNLM */
            if (strcmp(hist[hSize2 - 1], "<s>") == 0)
            {
               RNLMReset(lm);
            }
            /* recursively building preceding ngrams */
            else
            {
                wdID_prev = RNNLMOutVocabMap(rnnlm, hist[hSize2 - 1]);
                fndx[0] = RNNLMInVocabMap (rnnlm, word);
                strcpy(succword_new[1], succword[0]);
                strcpy(succword_new[2], succword[1]);
                strcpy(succword_new[0], word);

                prob_prev = UpdateRNNLMCache(lm, hist, rnnlm, wdID_prev, hist[hSize2 - 1], N, shift+1, useCache, succword_new, fndx);
                /* RNN LM previous history context - shift backwards by one word */
                ne_prev = GetNEntry2_RNNLM(nglm, ndx+2, fndx, TRUE, FALSE, ACTUAL_MLP_NSIZE, v, fv);
                /* re-assign current RNNLM history - history's future becomes present */
                AssignRNNLMHiddenVector(rnnlm, ne_prev->rnnlm_fhist);
            }
         }
         /* always copy to input layer and ready */
         copyHiddenLayerToInput(rnnlm);
      }

      GetRNNLMHiddenVector(rnnlm, v);

      prID = RNNLMInVocabMap(rnnlm, hist[hSize2 - 1]);
      if (rnnlm->usecuedrnnlm)
      {
          if (rnnlm->usecuedsurnnlm)
          {
              for (i=0; i<3; i++)
              {
                  if (strcmp(succword[i], "<s>") == 0)
                  {
                      succwdID_new[i] = INVALID_INT;
                  }
                  else if(strcmp(succword[i], "</s>") == 0)
                  {
                      succwdID_new[i] = 0;
                  }
                  else
                  {
                      succwdID_new[i] = succwdID[i];
                  }
              }
              prob = CUEDRNNLMAcceptWord_v1_1_SU (rnnlm, prID, wdID, succwdID_new);
          }
          else if (fabs(rnnlm->cuedversion - 1.1) < 1e-6)
          {
              prob = CUEDRNNLMAcceptWord_v1_1 (rnnlm, prID, wdID);
          }
          else if (rnnlm->cuedversion == 1.0)
          {
              prob = CUEDRNNLMAcceptWord_v1_0 (rnnlm, prID, wdID);
          }
          else
          {
              prob = CUEDRNNLMAcceptWord (rnnlm, prID, wdID);
          }
      }
      else if (rnnlm->usefrnnlm)
      {
         prob = FRNNLMAcceptWord(rnnlm, prID, wdID);
      }
      else {
         prob = RNNLMAcceptWord(rnnlm, prID, wdID);
      }
      /* prob is LOG10 based !!! */
      prob = LOG10_TO_FLT(prob);
      /* current word becomes part of future history */
      rnnlm->history[0] = RNNLMInVocabMap(rnnlm, wdid->name);
      GetRNNLMHiddenVector(rnnlm, fv);

#if 0
      fprintf(stdout, "Inserting %d ngrams into cache : P(", rnnlm->out_num_word);
      for (j=1; j<hSize2; j++) {
         fprintf(stdout, "%s ", hist[j]);
      }
      fprintf(stdout, "-> *)\n");
#endif

#ifndef MLPLMPROBNORM_OOS
      /* only do this when not in recursion on preceding ngrams */
#if 1
      if (nglm->heap->totAlloc > 1073741824 && shift == 0)
      {
#endif
#if 0
      if (nglm->heap->totAlloc > 2147483648 && shift == 0) {
#endif
         int counts[NSIZE] = {250001};
         ResetHeap(nglm->heap);
         nglm = CreateBoNGram((LModel *)lm->data.rnlm->cache, rnnlm->in_num_word, counts);
      }
#endif

      ne = GetNEntry2_RNNLM(nglm, ndx+1, fndx+1, TRUE, TRUE, ACTUAL_MLP_NSIZE, v, fv);
      ne->se = (SEntry *) New(nglm->heap, rnnlm->out_num_word * sizeof(SEntry));
      ne->nse = rnnlm->out_num_word;
      cse = ne->se;

      for (i=0; i<rnnlm->out_num_word; i++) {
         /* prob table contain probs */
         cse->prob = rnnlm->outP_arr[i];
         cse->word = (lmId) i;
         cse++;
      }

      /* RNNLM current history vector */
      ne->rnnlm_hist = CreateVector(nglm->heap, VectorSize(v));
      CopyVector(v, ne->rnnlm_hist);
      /* RNNLM future history vector */
      ne->rnnlm_fhist = CreateVector(nglm->heap, VectorSize(fv));
      CopyVector(fv, ne->rnnlm_fhist);
      /* RNNLM model level tracking of current LM state */
      lm->data.rnlm->curlmste = ne;
      CopyVector(v, lm->data.rnlm->rnnlm_hist);
      CopyVector(fv, lm->data.rnlm->rnnlm_fhist);
#if 0
      fprintf(stdout, "Current RNNLM history vector : \n");
      ShowVector("ne->rnnlm_hist: ", ne->rnnlm_hist, 50);
      fprintf(stdout, "Future RNNLM history vector : \n");
      ShowVector("ne->rnnlm_fhist: ", ne->rnnlm_fhist, 50);
      fflush(stdout);
#endif
      Dispose(&gstack, v);

#ifdef MLPLMPROBNORM
/*       LabId prid[MLP_NSIZE] = {0}; */
      ne->bowt = LZERO;

      /* MLP/RNN LM history context */
      for (i=0; i<hSize2; i++)
      {
         prid[hSize2 - i - 1] = GetLabId(hist[i], FALSE);
      }

/* MLP/RNN prob normalization for cases WITHOUT OOS output node */
#ifndef MLPLMPROBNORM_OOS
      HError(999, "RNNLM prob normalization for cases WITHOUT OOS output node not supported !!!");
/* MLP/RNN prob normalization for cases WITH OOS output node */
#else
      if (wdID == rnnlm->out_oos_nodeid)
      {
         for (i=1; i<=((LModel *)lm->data.rnlm->nglm)->data.ngram->vocSize; i++) {
            wdid = ((LModel *)lm->data.rnlm->nglm)->data.ngram->wdlist[i];
            /* only accumulate \sum_{w} P_ng(w|h) for OOS words */
            if (RNNLMOutVocabMap(rnnlm, wdid->name) == rnnlm->out_oos_nodeid) {
               ne->bowt = LAdd(ne->bowt, GetLMProb((LModel *)lm->data.rnlm->nglm, prid, wdid));
            }
         }

         wdid = GetLabId(word, FALSE);
#if 0
         fprintf(stdout, "word : %s OOS prob normalization term: %e %e %e, prob: %e %e\n", word,
                 GetLMProb((LModel *)lm->data.rnlm->nglm, prid, wdid), ne->bowt,
                 exp(GetLMProb((LModel *)lm->data.rnlm->nglm, prid, wdid) - ne->bowt),
                 prob, prob * exp(GetLMProb((LModel *)lm->data.rnlm->nglm, prid, wdid) - ne->bowt));
         fflush(stdout);
#endif
         prob *= exp(GetLMProb((LModel *)lm->data.rnlm->nglm, prid, wdid) - ne->bowt);
      }
#endif
#endif

   }

   /* Defunct MLP/RNN prob output for OOV words unless performing
      MLP/RNN prob normalization for cases WITH OOS output node */
#ifndef MLPLMPROBNORM_OOS
   /* OOV words assigned zero prob */
/*    if (wdID == rnnlm->out_oos_nodeid && rnnlm->usefrnnlm) { */
   if (wdID == rnnlm->out_oos_nodeid && wdID >= rnnlm->out_num_word - 1)
   {
      return LZERO;
   }
   if (wdID == rnnlm->out_oos_nodeid && strcmp(wdid->name, "</s>") != 0 && strcmp(wdid->name, "!SENT_END") != 0)
   {
      assert(wdID == 0);
      return LZERO;
   }
#endif

   return prob;
}


/* GetRNNLMProb: computing RNN LM prob */
float GetRNNLMProb(LModel* lm , LabId prid[NSIZE], LabId wdid, LabId *succwordId)
/* Given   word prid, return prob p(wdid | prid ... <s> )
   if updhistory == TRUE, will put prid into history and update
   hidden history vector */
/* return value: ln p(wdid | prid ... <s> ) */
{
   int i = 0, hSize = 0, hSize2 = 0, wdID = 0;
   char **hist = NULL;
   float prob = 0;
   RNNLM* rnnlm = NULL;

   rnnlm = (RNNLM*)(lm->data.rnlm->rnnlm);

   hSize = ACTUAL_MLP_NSIZE - 1;
/*    hSize2 = ACTUAL_MLP_NSIZE; */
   hSize2 = MLP_NSIZE;
   hist = (char **) New(lm->heap, hSize2 * sizeof(char *));

   for (i=0; i<hSize2; i++)
   {
      /* assigned zero prob to n-grams with truncated history contexts */
      if (!prid[hSize2 - i - 1])
      {
         /* if need to pad <s> at the sentence start to make full span context */
         if (padStartWord)
         {
            prid[hSize2 - i - 1] = GetLabId("<s>", FALSE);
            if (!prid[hSize2 - i - 1]) {
               HError(999, "GetRNNLMProb : sentence start token <s> missing in word list");
            }
         }
         else
         {
            return LZERO;
         }
      }
      if (prid[hSize2 - i - 1] != NULL && prid[hSize2 - i - 1]->name != NULL)
      {
         hist[i] = prid[hSize2 - i - 1]->name;
      }
      else
         hist[i] = NULL;
   }

   wdID = RNNLMOutVocabMap(rnnlm, wdid->name);
   int succword[NSIZE];
   char succwordname[NSIZE][100];
   for (i=0; i<3; i++)
   {
       succword[i] = RNNLMInVocabMap (rnnlm, succwordId[i]->name);
       strcpy (succwordname[i], succwordId[i]->name);
   }

   /* query and update cache if necessary */
   prob = UpdateRNNLMCache(lm, hist, rnnlm, wdID, wdid->name, hSize+1, 0, TRUE, succwordname, succword);
   Dispose(lm->heap, hist);

   return LOG_NATURAL(prob);
}


#ifndef NO_LAT_LM
/* LMTransProb_rnnlm

     return logprob of transition from src labelled word. Also return dest state.
     rnnlm case
*/
LogFloat LMTrans_RNNLM (LModel *lm, LMState src, LabId wdid, LMState *dest, LabId *succwordId)
{
   int i = 0, hSize = 0, hSize2 = 0;
   lmId ndx[MLP_NSIZE] = {0};
   lmId fndx[MLP_NSIZE] = {0};
   LabId prid[MLP_NSIZE] = {0};
   Vector v = NULL, fv = NULL;
   NEntry *ne = NULL;
   NGramLM *nglm = NULL;
   RNNLM* rnnlm = NULL;
   float prob = 0;

   hSize = ACTUAL_MLP_NSIZE - 1;
/*    hSize2 = ACTUAL_MLP_NSIZE; */
   hSize2 = MLP_NSIZE;

   /* truncated previous MLP LM history context - remove last word */
   if (src) {
      for (i=0; i<hSize2-1; i++) {
         if (((NEntry *)src)->word[i] != 0) {
            ndx[i+1] = ((NEntry *)src)->word[i];
         }
      }
   }
   /* current word to predict, now becomes first word in history */
   ndx[0] = (int) wdid->aux;

   rnnlm = (RNNLM*)((LModel *)lm->data.rnlm->rnnlm);

   for (i=0; i<3; i++)
   {
       fndx[i] = (int) succwordId[i]->aux;
   }

   /* RNNLM recurrent history vector */
   v = CreateVector(&gstack, GetRNNLMHiddenVectorSize(rnnlm));
   fv = CreateVector(&gstack, GetRNNLMHiddenVectorSize(rnnlm));
   ZeroVector(v); ZeroVector(fv);

   if (src) {
      for (i=0; i<hSize2-1; i++) {
         if (((NEntry *)src)->word[i] != 0) {
            prid[i] = lm->data.rnlm->wdlist[((NEntry *)src)->word[i]];
         }
      }
   }
   prob = GetRNNLMProb(lm, prid, wdid, succwordId);

   /* query in LM state cache  */
   nglm = ((LModel *)lm->data.rnlm->lmstate)->data.ngram;
   if (rnnlmUseHVDist == FALSE)
   {
       ne = GetNEntry2_RNNLM(nglm, ndx, fndx, TRUE, FALSE, ACTUAL_MLP_NSIZE, v, fv);
   }
   else
   {
      /* using weakened truncated contexts and history vector distance */
      /* RNNLM future history vector - now becomes current history for LM state */
      CopyVector(lm->data.rnlm->rnnlm_fhist, v);
      ne = GetNEntry2_RNNLM_HVDist(nglm, ndx, FALSE, rnnlmLMStateSize, v, fv);
   }

   /* if no LM state of matching context found then add to cache */
   if (!ne)
   {
#if 1
      if (nglm->heap->totAlloc > 268435456) {
         int counts[NSIZE] = {250001};
         ResetHeap(nglm->heap);
         nglm = CreateBoNGram((LModel *)lm->data.rnlm->lmstate, counts[0], counts);
      }
#endif
      if (rnnlmUseHVDist == FALSE)
      {
          ne = GetNEntry2_RNNLM(nglm, ndx, fndx, TRUE, TRUE, ACTUAL_MLP_NSIZE, v, fv);
      }
      else
      {
         ne = GetNEntry2_RNNLM_HVDist(nglm, ndx, TRUE, rnnlmLMStateSize, v, fv);
      }

      if (rnnlmUseHVDist == TRUE) {
         /* RNNLM current history vector */
         ne->rnnlm_hist = CreateVector(nglm->heap, VectorSize(v));
         ZeroVector(ne->rnnlm_hist);
         CopyVector(v, ne->rnnlm_hist);
      }

   }
   /* RNN LM state */
   *dest = (LMState) ne;

   Dispose(&gstack, v);

   return prob;
}

#endif


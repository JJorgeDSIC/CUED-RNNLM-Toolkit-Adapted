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
/*     File: HNLM.h neural network language model handling     */
/* ----------------------------------------------------------- */

/* !HVER!HLM:   3.4.1 [CUED 12/03/09] */

#ifndef _HNLM_H_
#define _HNLM_H_

#ifdef __cplusplus
extern "C" {
#endif

#define QN_MAX_LAYERS  5
#define QN_MAX_WEIGHTMATS QN_MAX_LAYERS-1
#define QN_MAX_SECTIONS   (QN_MAX_LAYERS-1 + QN_MAX_WEIGHTMATS)
#define QN_SIZET_BAD 0xffffffffffffffffu
#define QN_WEIGHTS_UNKNOWN 29

#define QN_MAT_TYPE_IEEELITTLE  0
#define QN_MAT_TYPE_IEEEBIG  1000

#define QN_MAT_TYPE_DOUBLE  0
#define QN_MAT_TYPE_FLOAT  10
#define QN_MAT_TYPE_INT32  20
#define QN_MAT_TYPE_INT16  30
#define QN_MAT_TYPE_UINT16  40
#define QN_MAT_TYPE_UINT8  50

#define QN_MAT_TYPE_FULL  0
#define QN_MAT_TYPE_TEXT  2
#define QN_MAT_TYPE_SPARSE  3

#define QN_MAT_NAMLEN_MAX  32

#define MAX_STRING_LENGTH 256
/* #define MLP_NSIZE 4 */
#define MLP_NSIZE NSIZE

typedef struct matheader
{
    int32 type;
    int32 mrows;
    int32 mcols;
    int32 imagf;
    int32 namlen;
} MatHeader;

typedef struct matinfo {
    fpos_t pos;             /* The location of the matrix in the file.*/
    int isbigendian;        /* Big endian?*/
    int isdouble;           /* Double precision?*/
    int rows;            /* Number of rows.*/
    int cols;            /* Number of columns.*/
} MatInfo;

typedef struct mlpweightfile {
    int a_layers;
    int a_layer_units[QN_MAX_LAYERS];
    float *weights12;
    float *bias2;
    float *weights23;
    float *bias3;
    float *weights34;
    float *bias4;
    float *weights45;
    float *bias5;
    MatInfo minfo[QN_MAX_SECTIONS];
} MLPWeightFile;

typedef struct string2indexmap {
    int *index;
    char **string;
} String2IndexMap;

typedef struct nnlm {
   MLPWeightFile weight;	/* MLP weight parameters */
   int in_num_word;             /* Input layer vocab size */
   int out_num_word;            /* Output layer vocab size */
   String2IndexMap inmap;	/* Input String to Index Map */
   String2IndexMap outmap;	/* Output String to Index Map */
   float *outP_arr;             /* Output prob array cache - only for most recent call */
} NNLM;

void InitNLM(void);
void LoadNLMwgt(char *wgt_file, char *inmap_file, char *outmap_file, 
                int in_num_word, int out_num_word, NNLM *NNlm);
/* char *id2string(String2IndexMap *map, int id); */
char *id2string_out(NNLM *NNlm, int id);
int String2Index(char *string, String2IndexMap map, int num_word);
float CalNLMProb(char **in_seq, NNLM *NNlm, char *out, int Ngram);

#ifdef __cplusplus
}
#endif

#endif  /* _HNLM_H_ */

/* ---------------------- End of HNLM.h ----------------------- */

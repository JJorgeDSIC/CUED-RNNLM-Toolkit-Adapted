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
/*     File: HNLM.c  neural network language model handling    */
/* ----------------------------------------------------------- */

char *hnlm_version = "!HVER!HNLM:   3.4 [CUED 25/04/06]";
char *hnlm_vc_id = "$Id: HNLM.c,v 1.1.1.1 2006/10/11 09:54:57 jhp33 Exp $";

#include <math.h>
#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HNLM.h"

/* --------------------------- Trace Flags ------------------------- */

#define T_TIO 1  /* Progress tracing whilst performing IO */

#if 1
#define OOS_NODE 1 /* Switch for with or without OOS node at output layer */
#endif

static int trace=0;

/* ---------------- Configuration Parameters --------------------- */

/*  

  TO DO: 

  1. Add config variable controlling projection matrix tying as option

  2. Use weight file specified layer size/number of units, overiding mlp_layer_size

*/

static ConfParam *cParm[MAXGLOBS];
static int nParm = 0;            /* total num params */

static int a_layers = 4;          /* number of layers and network size */

#ifdef OOS_NODE  
static int mlp_layer_size[4] = {300006,600,400,20002};
/*static int mlp_layer_size[4] = {60003,90,300,20001};*/
#else
static int mlp_layer_size[4] = {300006,600,400,20001};
#endif

static Boolean useBinSearch = TRUE;

/* ---------------------- Global Variables ----------------------- */

static MemHeap mlpInfoStack;       /* Local stack to for MLP info */

static FILE *weight_fp;
static FILE *fp_map_in;
static FILE *fp_map_out;

/* --------------------------- Initialisation ---------------------- */

#define LN10 2.30258509299404568 /* Defined to save recalculating it */
#define MINOUT 0.00000001

const char *m_sectname[] =
{
    "weights12",
    "bias2",
    "weights23",
    "bias3",
    "weights34",
    "bias4",
    "weights45",
    "bias5"
};

void read_all_hdrs(FILE *stream, int a_layers, int *mlp_layer_size, MLPWeightFile *weight);
void FreeNLMwgt(NNLM *NNlm);
void GenMap(FILE *fp_map, String2IndexMap *map, int num_word);
void FreeMap(String2IndexMap *map, int num_word);
static int binsearch(char *str[], int max, char *value);
void copy_vf_vf(int vec_len, const float* vec, float* o_vec);
void mulntacc_mfmf_mf(int Sm,int Sk,int Sn, const float *A,const float *B,float *C);
void tanh_vf_vf(int n, const float* in_vec, float* out_vec);
void maxmin_vf_ff(size_t n, const float *vec, float *maxp, float *minp);
void softmax_vf_vf(int n, const float* in_vec, float* out_vec);


/* EXPORT->InitLM: initialise configuration parameters */
void InitNLM(void)
{
   int i;
   Boolean b;
/*    char buf[MAXSTRLEN]; */

   Register(hnlm_version,hnlm_vc_id);
   nParm = GetConfig("HNLM", TRUE, cParm, MAXGLOBS);

   /* setup the local memory management */
   CreateHeap(&mlpInfoStack, "mlpInfoStore", MSTAK, 1, 0.0, 100000, ULONG_MAX);

   if (nParm>0){
      if (GetConfInt(cParm,nParm,"TRACE",&i)) trace = i;
      if (GetConfBool(cParm,nParm, "USEBINSEARCH", &b)) useBinSearch = b;   
   }
}

/* EXPORT->LoadNLM: load weight parameters from the type of matlab -V4 */
void LoadNLMwgt(char *wgt_file, char *inmap_file, char *outmap_file, 
                int in_num_word, int out_num_word, NNLM *NNlm)
{
   int i;
/*   MLPWeightFile weight;*/
   MatInfo *minfo; /* Info on the matrices on disk.*/

   /* Work out how many layers the file seems to have.*/
   int file_n_layers;   /* How many layers the file seems to have */
   int file_layer_units[QN_MAX_LAYERS]; /* The apparent size of each layer.*/

   int file_n_sections;
   int sect, lay;
   String2IndexMap map_in, map_out;

   InitNLM();
   
   weight_fp = fopen(wgt_file,"r");
   fp_map_in = fopen(inmap_file,"r");
   fp_map_out = fopen(outmap_file,"r");

   NNlm->in_num_word = in_num_word;
   NNlm->out_num_word = out_num_word;
   minfo = NNlm->weight.minfo;
   read_all_hdrs(weight_fp, a_layers, mlp_layer_size, &NNlm->weight);

   /* Use the weights to work out the number of layers. */
   file_n_layers = 0;
   for (i=0; i<QN_MAX_SECTIONS; i+=2)
   {
       if (minfo[i].rows>0)
           file_n_layers = i/2 + 2;
   }
   printf("Weight file appears to have %lu layers.\n", (unsigned long) file_n_layers);

   /* Use the weights again to work out the size of each layer.*/
   file_layer_units[0] = minfo[0].cols;
   for (i=1; i<file_n_layers; i++)
   {
       file_layer_units[i] = minfo[(i-1)*2].rows;
   }
   for (i=0; i<file_n_layers; i++)
   {
       printf("Layer %lu has %lu units.\n", (unsigned long) i+1, (unsigned long) file_layer_units[i]);
   }


   /* Now check the size of each matrix.*/
   file_n_sections = (file_n_layers-1) * 2;
   for (sect = 0, lay = 0; sect<file_n_sections; sect+=2, lay++)
   {
       if ( (minfo[sect].cols!=file_layer_units[lay]) ||
            (minfo[sect].rows!=file_layer_units[lay+1]) ||
            (minfo[sect+1].cols!=file_layer_units[lay+1]) ||
            (minfo[sect+1].rows!=1) )
       {
           printf("LOG : Weight parameters between %d and %d are tied-up.\n", lay, lay+1); /* jhp33 - 20091113 */
       }
    }

    /* If constructor specifies number of layers, check file agrees.*/
    if (a_layers!=0 && a_layers != file_n_layers)
    {
        HError(-1,"Matlab Weight file constructor requested %lu layers, file only had %lu layers.", a_layers, file_n_layers);
    }

    /* If constructor specifies size of layer, check file agrees.*/
    for (i=0; i<file_n_layers; i++)
    {
        if (mlp_layer_size!=NULL && mlp_layer_size[i]!=0
                && mlp_layer_size[i]!=file_layer_units[i])
        {
/*             printf("LOG : Matlab Weight file constructor requested " */
           printf("LOG : Default MLP network size requested " 
                       "layer %lu had %lu units, file had "
                       "%lu in the layer.\n",
                       (unsigned long) i+1,
                       (unsigned long) mlp_layer_size[i],
                       (unsigned long) file_layer_units[i]); /* jhp33 - 20091113 */
        }
        NNlm->weight.a_layer_units[i] = file_layer_units[i];
/*         /\* this would over-writes MLPLM file info with default size settings *\/ */
/*         NNlm->weight.a_layer_units[i] = mlp_layer_size[i]; /\* jhp33 - 20091113 *\/ */
        if (i == file_n_layers -1) {
           NNlm->weight.a_layer_units[i] = file_layer_units[i];
        }
    }

    /* Generate map */
    GenMap(fp_map_in, &map_in, NNlm->in_num_word);
    GenMap(fp_map_out, &map_out, NNlm->out_num_word);
    fclose(fp_map_in); fclose(fp_map_out); fclose(weight_fp);

    for (i=0; i<=5; i++) {
       fprintf(stdout, "%d : %d X %d\n", i, NNlm->weight.minfo[i].rows, NNlm->weight.minfo[i].cols);
    }
    fprintf(stdout, "\n");
    fflush(stdout);

    NNlm->inmap = map_in;
    NNlm->outmap = map_out;
    NNlm->outP_arr = (float*)malloc(sizeof(float)*NNlm->out_num_word);
}


void read_all_hdrs(FILE *stream, int a_layers, int *mlp_layer_size, MLPWeightFile *weight)
{
    int ec;
    int i,j;
    int ret;
    fpos_t startpos;            /* Where we are in file at start.*/
    int matno = 0;           /* The number of the current matrix.*/

    int isbigendian = 0;        /* IS this matrix bigendian?*/
    int isdouble = 0;           /* Is this matrix double (c.f. float)?*/

    MatHeader hdr;          /* A local copy of the header.*/
    int namlen;              /* The length of the matrix name.*/
    char name[QN_MAT_NAMLEN_MAX];       /* The name of the matrix.*/
    int elesize = 0;         /* The size of a single element.*/

    int skip;               /* Skip this matrix if non-zero.*/
    fpos_t pos;             /* Position for this matrix.*/
    int index=0;           /* INdex of the current section.*/

    float weight_tmp_float;
    double weight_tmp_double;

    weight->a_layers = a_layers;
    for (i = 0; i<a_layers; i++)
    {
        if (mlp_layer_size!=NULL)
            weight->a_layer_units[i] = mlp_layer_size[i];
        else
            weight->a_layer_units[i] = 0;
    }

    for (i=0; i<QN_MAX_SECTIONS; i++)
    {
        weight->minfo[i].rows = 0;
        weight->minfo[i].cols = 0;
    }



    ec = fgetpos(stream, &startpos);
    if (ec!=0)
    {
        HError(-1,"Failed go get current position in matlab weights file '%s' - %s.");
    }
    while (1)
    {
        ec = fgetpos(stream, &pos);
        if (ec!=0)
        {
            HError(-1,"Failed go get current position in matlab weights file '%s' - %s.");
        }
        /* Read in the matrix header, does endian swapping too.*/
        ret = fread(&hdr, sizeof(MatHeader), 1, stream);
        if (ret !=1)
        {
            if (feof(stream))
                break;
            else
            {
                HError(-1,"Failed to read matrix header");
            }
        }
        namlen = hdr.namlen;
        if (namlen>QN_MAT_NAMLEN_MAX)
        {
            HError(-1,"Matrix name too long in matrix number %lu from matlab weights file.", (unsigned long) matno);
        }
        ret = fread(name, 1, namlen, stream);
        if (ret!=namlen)
        {
            HError(-1,"Failed to read name of matrix number %lu from matlab weights file.", (unsigned long) matno);
        }
        if (name[namlen-1]!='\0')
        {
            HError(-1,"Missing terminating null in name of matrix number %lu from matlab weights file.", (unsigned long) matno);
        }
        /* Work out matrices we cannot handle.*/
        skip = 0;
        switch (((hdr.type/1000) % 10) * 1000)
        {
        case QN_MAT_TYPE_IEEELITTLE:
            isbigendian = 0;
            break;
        case QN_MAT_TYPE_IEEEBIG:
            isbigendian = 1;
            break;
        default:
            skip = 1;
        }
        switch (((hdr.type / 10) % 10) * 10)
        {
        case QN_MAT_TYPE_DOUBLE:
            isdouble = 1;
            elesize = sizeof(double);
            break;
        case QN_MAT_TYPE_FLOAT:
            isdouble = 0;
            elesize = sizeof(float );
            break;
        default:
            skip = 1;
        }
        switch (hdr.type % 10)
        {
        case QN_MAT_TYPE_FULL:
            break;
        default:
            skip = 1;
        }
        if (hdr.mrows==0 || hdr.mcols==0 || hdr.imagf!=0)
            skip = 1;
/*        index = QN_SIZET_BAD;*/ /* An index value indicating unknown sect.*/
        for (i=0; i<QN_MAX_SECTIONS; i++)
        {
            if (!strcmp(name, m_sectname[i]))
            {
                index = i;
            }
        }
/*        if (index==QN_SIZET_BAD)
            skip |= 1;*/
        if (skip)
        {
            printf("Skipping matrix number %lu, name '%s'.",
                         (unsigned long) matno, name);
        }
        else
        {
            /* Use non-zero rows to check for previous matrix of same name.*/
            if (weight->minfo[index].rows>0)
            {
                printf("Duplicate matrix '%s' in Matlab weights file.", m_sectname[index]);

            }
            weight->minfo[index].pos = pos;
            weight->minfo[index].isbigendian = isbigendian;
            weight->minfo[index].isdouble = isdouble;
            weight->minfo[index].rows = hdr.mrows;
            weight->minfo[index].cols = hdr.mcols;
        }

        /* Load the data.*/
	switch(index){
	    case 0:
		weight->weights12 = (float*)malloc((hdr.mrows*hdr.mcols)*sizeof(float));
		if (isdouble){
		        for(i=0;i<hdr.mcols;i++)
			    for(j=0;j<hdr.mrows;j++){
				ec = fread(&weight_tmp_double, sizeof(double), 1 , stream);
				weight->weights12[j*hdr.mcols + i] = (float)weight_tmp_double;
			    }
		}
		else {
                        for(i=0;i<hdr.mcols;i++)
                            for(j=0;j<hdr.mrows;j++){
                                ec = fread(&weight_tmp_float, sizeof(float), 1 , stream);
                                weight->weights12[j*hdr.mcols + i] = weight_tmp_float;
                            }
		}
		break;
            case 1:
		weight->bias2 = (float*)malloc((hdr.mrows*hdr.mcols)*sizeof(float));
                if (isdouble){
		        for(i=0;i<hdr.mrows*hdr.mcols;i++){
                        	ec = fread(&weight_tmp_double, sizeof(double), 1, stream);
				weight->bias2[i] = (float)weight_tmp_double;
			}
		}
                else {
			for(i=0;i<hdr.mrows*hdr.mcols;i++){
                                ec = fread(&weight_tmp_float, sizeof(float), 1, stream);
                                weight->bias2[i] = weight_tmp_float;
			}
		}
		break;
            case 2:
                weight->weights23 = (float*)malloc((hdr.mrows*hdr.mcols)*sizeof(float));
                if (isdouble){
                        for(i=0;i<hdr.mcols;i++)
                            for(j=0;j<hdr.mrows;j++){
                                ec = fread(&weight_tmp_double, sizeof(double), 1 , stream);
                                weight->weights23[j*hdr.mcols + i] = (float)weight_tmp_double;
                            }
                }
                else {
                        for(i=0;i<hdr.mcols;i++)
                            for(j=0;j<hdr.mrows;j++){
                                ec = fread(&weight_tmp_float, sizeof(float), 1 , stream);
                                weight->weights23[j*hdr.mcols + i] = weight_tmp_float;
                            }
                }
                break;
            case 3:
                weight->bias3 = (float*)malloc((hdr.mrows*hdr.mcols)*sizeof(float));
                if (isdouble){
                        for(i=0;i<hdr.mrows*hdr.mcols;i++){
                                ec = fread(&weight_tmp_double, sizeof(double), 1, stream);
                                weight->bias3[i] = (float)weight_tmp_double;
			}
                }
                else {
                        for(i=0;i<hdr.mrows*hdr.mcols;i++){
                                ec = fread(&weight_tmp_float, sizeof(float), 1, stream);
                                weight->bias3[i] = weight_tmp_float;
			}
                }
                break;
            case 4:
                weight->weights34 = (float*)malloc((hdr.mrows*hdr.mcols)*sizeof(float));
                if (isdouble){
                        for(i=0;i<hdr.mcols;i++)
                            for(j=0;j<hdr.mrows;j++){
                                ec = fread(&weight_tmp_double, sizeof(double), 1 , stream);
                                weight->weights34[j*hdr.mcols + i] = (float)weight_tmp_double;
                            }
                }
                else {
                        for(i=0;i<hdr.mcols;i++)
                            for(j=0;j<hdr.mrows;j++){
                                ec = fread(&weight_tmp_float, sizeof(float), 1 , stream);
                                weight->weights34[j*hdr.mcols + i] = weight_tmp_float;
                            }
                }
                break;
            case 5:
                weight->bias4 = (float*)malloc((hdr.mrows*hdr.mcols)*sizeof(float));
                if (isdouble){
                        for(i=0;i<hdr.mrows*hdr.mcols;i++){
                                ec = fread(&weight_tmp_double, sizeof(double), 1, stream);
                                weight->bias4[i] = (float)weight_tmp_double;
			}
                }
                else {
                        for(i=0;i<hdr.mrows*hdr.mcols;i++){
                                ec = fread(&weight_tmp_float, sizeof(float), 1, stream);
                                weight->bias4[i] = weight_tmp_float;
			}
                }
                break;
            case 6:
		if (a_layers > 4){
                	weight->weights45 = (float*)malloc((hdr.mrows*hdr.mcols)*sizeof(float));
	                if (isdouble){
	                        for(i=0;i<hdr.mcols;i++)
	                            for(j=0;j<hdr.mrows;j++){
	                                ec = fread(&weight_tmp_double, sizeof(double), 1 , stream);
	                                weight->weights45[j*hdr.mcols + i] = (float)weight_tmp_double;
	                            }
        	        }
	                else {
	                        for(i=0;i<hdr.mcols;i++)
	                            for(j=0;j<hdr.mrows;j++){
                	                ec = fread(&weight_tmp_float, sizeof(float), 1 , stream);
        	                        weight->weights45[j*hdr.mcols + i] = weight_tmp_float;
	                            }
	                }
		}
		break;
            case 7:
		if (a_layers > 4){
	                weight->bias5 = (float*)malloc((hdr.mrows*hdr.mcols)*sizeof(float));
	                if (isdouble){
	                        for(i=0;i<hdr.mrows*hdr.mcols;i++){
	                                ec = fread(&weight_tmp_double, sizeof(double), 1, stream);
	                                weight->bias5[i] = (float)weight_tmp_double;
				}
	                }
	                else {
	                        for(i=0;i<hdr.mrows*hdr.mcols;i++){
	                                ec = fread(&weight_tmp_float, sizeof(float), 1, stream);
	                                weight->bias5[i] = weight_tmp_float;
				}
	                }
		}
		break;

	}
        if (!ec)
        {
            HError(-1,"Failed to load over matrix number %lu in matlab weight file.", (unsigned long) matno);
        }
        matno++;                /* On to next matrix in file.*/
    }

    /* Seek back to start of file.*/
    ec = fsetpos(stream, &startpos);
    if (ec!=0)
    {
        HError(-1,"Failed go seek to start in matlab weights file.");
    }

}


void FreeNLMwgt(NNLM *NNlm)
{
        if (NNlm->weight.weights12 != NULL) free(NNlm->weight.weights12);
        if (NNlm->weight.bias2 != NULL ) free(NNlm->weight.bias2);
        if (NNlm->weight.weights23 != NULL) free(NNlm->weight.weights23);
        if (NNlm->weight.bias3 != NULL ) free(NNlm->weight.bias3);
        if (NNlm->weight.weights34 != NULL) free(NNlm->weight.weights34);
        if (NNlm->weight.bias4 != NULL ) free(NNlm->weight.bias4);
        if (NNlm->weight.weights45 != NULL) free(NNlm->weight.weights45);
        if (NNlm->weight.bias5 != NULL ) free(NNlm->weight.bias5);
	if (NNlm->outP_arr != NULL) free(NNlm->outP_arr);

	FreeMap(&NNlm->inmap, NNlm->in_num_word);
	FreeMap(&NNlm->outmap, NNlm->out_num_word);	
	
}


void GenMap(FILE *fp_map, String2IndexMap *map, int num_word)
{
    int i;

    map->index = (int*)malloc(sizeof(int)*num_word);
    map->string = (char**)malloc(sizeof(char*)*num_word);
    for(i=0;i<num_word;i++) map->string[i] = (char*)malloc(sizeof(char)*MAX_STRING_LENGTH);

    for(i=0;i<num_word;i++){
	fscanf(fp_map, "%d", &map->index[i]);
	fscanf(fp_map, "%s", map->string[i]);
    }
}

void FreeMap(String2IndexMap *map, int num_word)
{
    int i;

    free(map->index);
    for(i=0;i<num_word;i++) free(map->string[i]);
    free(map->string);
}

char *id2string_in(NNLM *NNlm, int id)
{
    return NNlm->inmap.string[id]; 	
}

char *id2string_out(NNLM *NNlm, int id)
{
    return NNlm->outmap.string[id];
}


/*-------------------------------------------------------------------------------------------------*/

static int binsearch(char *str[], int max, char *value) {
 int position;
 int begin = 0; 
 int end = max - 1;
 int cond = 0;

 while(begin <= end) {
  position = (begin + end) / 2;
  if((cond = strcmp(str[position], value)) == 0)
   return position;
  else if(cond < 0)
   begin = position + 1;
  else
   end = position - 1;
 }

 return max;
}

int String2Index(char *string, String2IndexMap map, int num_word)
{
  if (!useBinSearch) {
    int i;

    for(i=0;i<num_word;i++)
        if(strcmp(string,map.string[i]) == 0) return i;
    return num_word;
  }
  else {
    return binsearch(map.string, num_word, string);
  }
}

float CalNLMProb(char **in_seq, NNLM *NNlm, char *out, int Ngram)
{
    int i;
    int cur_layer;           /* The index of the current layer. */
    int prev_layer;          /* The index of the previous layer. */
    int cur_layer_units;     /* The number of units in the current layer. */
    int prev_layer_units;    /* The number of units in the previous layer. */
    int cur_layer_size;      /* The size of the current layer. */
    float *cur_layer_x = NULL;         /* Input to the current layer non-linearity. */
    float *cur_layer_y = NULL;         /* Output from the current layer */
                                /* non-linearity. */
    const float *prev_layer_y;  /* Output from the previous non-linearity. */
    float *cur_layer_bias;      /* Biases for the current layer. */
    float *cur_weights;         /* Weights inputing to the current layer. */

    int m,n;
    float *l_weight[QN_MAX_LAYERS];
    float *layer_bias[QN_MAX_LAYERS];
    float **layer_x, **layer_y;

    float l_output;
    int g_window_extent = Ngram - 1;
    unsigned int in_index[Ngram]; /* jhp33 - 20091113 */
    int out_index;
    MLPWeightFile weight;

    for(i=0;i<Ngram-1;i++) in_index[i] = String2Index(in_seq[i], NNlm->inmap, NNlm->in_num_word);

    out_index = String2Index(out, NNlm->outmap, NNlm->out_num_word);
#ifdef OOS_NODE
    if (out_index > NNlm->out_num_word) return 0;
#else
    if (out_index >= NNlm->out_num_word) return 0;
#endif

    weight = NNlm->weight;

    for(i=1;i<QN_MAX_LAYERS;i++){
        switch(i){
            case 1:
                l_weight[i]=weight.weights12; layer_bias[i]=weight.bias2;
                break;
            case 2:
                l_weight[i]=weight.weights23; layer_bias[i]=weight.bias3;
                break;
            case 3:
                l_weight[i]=weight.weights34; layer_bias[i]=weight.bias4;
                break;
            case 4:
                l_weight[i]=weight.weights45; layer_bias[i]=weight.bias5;
                break;
        }
    }

    layer_x = (float**)malloc(sizeof(float*)*weight.a_layers);
    layer_y = (float**)malloc(sizeof(float*)*weight.a_layers);
    for(cur_layer=1; cur_layer<weight.a_layers; cur_layer++){
        layer_x[cur_layer] = (float*)malloc(sizeof(float)*weight.a_layer_units[cur_layer]);
        layer_y[cur_layer] = (float*)malloc(sizeof(float)*weight.a_layer_units[cur_layer]);
    }

    for (cur_layer=1; cur_layer<weight.a_layers; cur_layer++)
    {
        prev_layer = cur_layer - 1;
        cur_layer_units = weight.a_layer_units[cur_layer];
        prev_layer_units = weight.a_layer_units[prev_layer];
        cur_layer_size = cur_layer_units;
        cur_layer_x = layer_x[cur_layer];
        cur_layer_y = layer_y[cur_layer];

        if (cur_layer==1)
           prev_layer_y = (float*) in_index;
        else
            prev_layer_y = layer_y[prev_layer];

        cur_layer_bias = layer_bias[cur_layer];
        cur_weights = l_weight[cur_layer];

        if (cur_layer==1)
                for(m=0;m<g_window_extent;m++)
                        for(n=0;n<(int)(cur_layer_units/g_window_extent);n++)
                                cur_layer_x[m*(int)(cur_layer_units/g_window_extent)+n]
					= cur_weights[n + (int)(cur_layer_units/g_window_extent*in_index[m])]; /* jhp33 - 20091113 */
        else{
                copy_vf_vf(cur_layer_units, cur_layer_bias, cur_layer_x);
                mulntacc_mfmf_mf(1, prev_layer_units, cur_layer_units,
                                    prev_layer_y, cur_weights,
                                    cur_layer_x);
        }

        if (cur_layer!=weight.a_layers - 1)
        {
            /* This is the intermediate layer non-linearity.*/
            if (cur_layer == 1)
                        copy_vf_vf(cur_layer_size, cur_layer_x,cur_layer_y); /*jhp33 (linear in projection layer)*/
            else tanh_vf_vf(cur_layer_size, cur_layer_x, cur_layer_y);
        }
        else softmax_vf_vf(cur_layer_units, cur_layer_x, cur_layer_y);
    }

    l_output =  (cur_layer_y[out_index] > MINOUT ) ? cur_layer_y[out_index]:MINOUT;
    for(i=0; i<NNlm->out_num_word; i++) NNlm->outP_arr[i] = (cur_layer_y[i] > MINOUT ) ? cur_layer_y[i]:MINOUT;

    for(cur_layer=1; cur_layer<weight.a_layers; cur_layer++){
        free(layer_x[cur_layer]);
        free(layer_y[cur_layer]);
    }
    free(layer_x); free(layer_y);

    return l_output;
}


void copy_vf_vf(int vec_len, const float* vec, float* o_vec)
{
    int i;
    for(i=0;i<vec_len;i++) o_vec[i]=vec[i];
}

void mulntacc_mfmf_mf(int Sm,int Sk,int Sn, const float *A,const float *B,float *C)
{
  int i,j,k;
  for (i=0;i<Sm;i++)
    for (j=0;j<Sn;j++)
      for (k=0;k<Sk;k++)
        C[i*Sn+j] += A[i*Sk+k]*B[j*Sk+k];
}

void tanh_vf_vf(int n, const float* in_vec, float* out_vec)
{
    int i;

    for (i=n; i!=0; i--)
    {
        *out_vec++ = tanh(*in_vec++);
    }
}

void maxmin_vf_ff(size_t n, const float *vec, float *maxp, float *minp)
{
    float max;
    float min;
    int i;
    float elem;

    if (n>0)
    {
        max = *vec;
        min = *vec++;

        for (i=n-1; i!=0; i--)
        {
            elem = *vec++;
            if (elem > max)
                max = elem;
            else if (elem < min)
                min = elem;
        }

        *maxp = max;
        *minp = min;
    }
}


void softmax_vf_vf(int n, const float* in_vec, float* out_vec)
{
    float max;
    float min;
    float sumexp = 0.0f;        /* Sum of exponents */
    float scale;                /* 1/sum of exponents */
    int i;

    float f;                /* Input value. */
    float e;                /* Exponent of current value. */


    maxmin_vf_ff(n, in_vec, &max, &min);     /* Find constant bias. */
    for (i=0; i<n; i++)
    {
        f = in_vec[i];
        e = exp(f - max);
        out_vec[i] = e;
        sumexp += e;
    }
    scale = 1.0f/sumexp;
    for (i=0; i<n; i++)
    {
        out_vec[i] = out_vec[i] * scale;
    }
}

/* ------------------------- End of HNLM.c ------------------------- */


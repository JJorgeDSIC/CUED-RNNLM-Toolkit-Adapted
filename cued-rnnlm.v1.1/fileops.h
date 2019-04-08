#include "head.h"
#include "Mathops.h"

#include <vector>

class FILEPTR
{
 private:
  std::vector<char> _word;
  
protected:
    FILE *fptr;
    int i;
    string filename;
public:
    FILEPTR()
    {
        fptr = NULL;
    }
    ~FILEPTR()
    {
        if (fptr)       fclose(fptr);
        fptr = NULL;
    }
    void open (string fn)
    {
        filename = fn;
        fptr = fopen (filename.c_str(), "rt");
        if (fptr == NULL)
        {
            printf ("ERROR: Failed to open file: %s\n", filename.c_str());
            exit (0);
        }
    }
    void close()
    {
        if (fptr)
        {
            fclose(fptr);
            fptr = NULL;
        }
    }
    bool eof()
    {
        return feof(fptr);
    }
    int readint ()
    {
        if (!feof(fptr))
        {
            if(fscanf (fptr, "%d", &i) != 1)
            {
                if (!feof(fptr))
                {
                    printf ("Warning: failed to read feature index from text file (%s)\n", filename.c_str());
                    exit (0);
                }
            }
            return i;
        }
        else
        {
            return INVALID_INT;
        }
    }
    void readline (vector<string> &linevec, int &cnt);
};

class ReadFileBuf
{
protected:
    int linecnt, wordcnt, cachesize, minibatch, lineperstream,
        mbcnt, mbcnter, randint;
    char line[1024][100];
    string filename, inputfilename, outputfilename, inputindexfilename, outputindexfilename;
    FILEPTR fileptr;
    FILE *fptr_in, *fptr_out;
    int *Indata, *Outdata, *featureindices;
    double *unigram, *accprob;
    float *logunigram;
    Matrix *inputbufptr, *outputbufptr;
    WORDMAP &inputmap, &outputmap;
public:
    ReadFileBuf(string txtfile, WORDMAP &inmap, WORDMAP &outmap, int mbsize, int csize, int num_fea=0, int rint=-1);
    void Indexword(string word, int &Inindex, int &Outindex);
    void FillBuffer();
    void DeleteIndexfile();
    void Init();
    void GetData(int index, int *indata, int *outdata)
    {
        Matrix &inputbuf = *inputbufptr;
        Matrix &outputbuf = *outputbufptr;
        memcpy (indata, inputbuf[index], sizeof(int)*minibatch);
        memcpy (outdata, outputbuf[index], sizeof(int)*minibatch);
    }

    void GetInputData (int origindex, int index, int *indata, int *prev_indata)
    {
        Matrix &inputbuf = *inputbufptr;
        Matrix &outputbuf = *outputbufptr;
        if (index >= mbcnt)
        {
            // set the succeeding words exceeding EOS as INPUT_PAD_INT
            for (int i=0; i<minibatch; i++)
            {
                indata[i] = INVALID_INT;
            }
        }
        else
        {
            for (int i=0; i<minibatch; i++)
            {
                if (outputbuf[origindex][i] == 0)
                {
                    indata[i] = INPUT_PAD_INT;
                }
                else
                {
                    indata[i] = inputbuf[index][i];
                }

                /*
                if (indata[i] == 0)         // reach sentence end
                {
                    indata[i] = INPUT_PAD_INT;
                }
                else*/
                if (prev_indata)       // already reach sentence end
                {
                    if (prev_indata[i] == INPUT_PAD_INT || prev_indata[i] == 0)
                    {
                        indata[i] = INPUT_PAD_INT;
                    }
                }
            }
        }
    }
    ~ReadFileBuf()
    {
        if (inputbufptr)       delete inputbufptr;
        if (outputbufptr)      delete outputbufptr;
        if (Indata)         delete Indata;
        if (Outdata)        delete Outdata;
        if (unigram)        delete [] unigram;
        if (accprob)        delete [] accprob;
        if (logunigram)     delete [] logunigram;
        if (featureindices) delete [] featureindices;
        if (fptr_in)        fclose(fptr_in);
        if (fptr_out)       fclose(fptr_out);
    }

    int getWordcnt ()
    {
        return wordcnt;
    }
    int getLinecnt ()
    {
        return linecnt;
    }
    int getMBcnt ()
    {
        return mbcnt;
    }
    int getRandint()
    {
        return randint;
    }
    void setMBcnter (int n)
    {
        mbcnter = n;
    }
    double* getUnigram ()
    {
        return unigram;
    }
    double* getAccprob ()
    {
        return accprob;
    }
    float* getLogUnigram ()
    {
        return logunigram;
    }
    int* getfeaptr()
    {
        return featureindices;
    }
};

#include "layer.h"

void DeleteMat (matrix *ptr)
{
    if (ptr)
    {
        delete ptr;
        ptr = NULL;
    }
}
void DeleteMatVec (vector<matrix *> matvec)
{
    for (int i=0; i<matvec.size(); i++)
    {
        if (matvec[i])
        {
            delete matvec[i];
            matvec[i] = NULL;
        }
    }
}

void matrixXvector (float *src, float *wgt, float *dst, int nr, int nc)
{
    int i, j;
    #pragma omp parallel for private(j)
    for (i=0; i<nc; i++)
    {
        for (j=0; j<nr; j++)
        {
            dst[i] += src[j]*wgt[j+i*nr];
        }
    }
    return;
}

layer::~layer()
{
    if (cunorm2ptr)
    {
        cufree (cunorm2ptr);
        cunorm2ptr = NULL;
    }
    DeleteMat (U);
    DeleteMat (dU);
    DeleteMatVec (dropoutmaskMat_vec);
}

layer::layer (int nr, int nc, int mbsize, int cksize)
{
    type = "base layer";
    momentum = 0.0;             // no momentum by default
    l2reg    = 0.0;
    gradient_cutoff = 5.0;
    gamma = 0.9995;               // TODO: make it a configuration later
    nrows = nr;
    ncols = nc;
    size = nr*nc;
    U = new matrix (nrows, ncols);
    dU= new matrix (nrows, ncols);
    U->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    dU->initmatrix();
    cunorm2ptr = (float *)cucalloc (sizeof(float));
    dropoutrate = 0.0;
    evalmode = false;

    chunksize = cksize;
    minibatch = mbsize;
    dropoutmaskMat_vec.clear();
}

void layer::allocDropoutMask ()
{
    dropoutmaskMat_vec.resize (chunksize);
    for (int i=0; i<chunksize; i++)
    {
        dropoutmaskMat_vec[i] = new matrix (nrows, minibatch);
        dropoutmaskMat_vec[i]->assignmatvalue (1.0);
    }
}

void layer::printGrad ()
{
    int i, j;
    printf ("Printing the gradient information...\n");
    dU->fetch();
    for (i=0; i<nrows; i++)
    {
        for (j=0; j<ncols; j++)
        {
            printf ("%.5f ", dU->fetchhostvalue(i, j));
        }
        printf("\n");
    }
}

float layer::fetchweightvalue (int i, int j)
{
    return U->fetchvalue(i,j);
}
float layer::fetchgradweightvalue (int i, int j)
{
    return dU->fetchvalue(i,j);
}

void inputlayer::Write (FILE *fptr)
{
    U->fetch ();
    U->Write (fptr);
    if (dim_fea > 0)
    {
        U_fea->fetch();
        U_fea->Write (fptr);
    }
}

void inputlayer::Read (FILE *fptr)
{
    U->Read (fptr);
    U->assign ();
    if (dim_fea > 0)
    {
        U_fea->Read(fptr);
        U_fea->assign();
    }
}

void outputlayer::Write (FILE *fptr)
{
    U->fetch ();
    U->Write (fptr);

    // U_succ->fetch ();
    // U_succ->Write (fptr);
}
void outputlayer::Read (FILE *fptr)
{
    U->Read (fptr);
    U->assign ();

    // U_succ->Read (fptr);
    // U_succ->assign ();
}

void recurrentlayer::Write (FILE *fptr)
{
    fprintf (fptr, "nodetype: %d\n", nodetype);
    fprintf (fptr, "reluratio: %f\n", reluratio);
    U->fetch ();
    W->fetch ();
    U->Write (fptr);
    W->Write (fptr);
}
void grulayer::Write (FILE *fptr)
{
    Wh->fetch();
    Wh->Write(fptr);
    Uh->fetch();
    Uh->Write(fptr);
    Wr->fetch();
    Wr->Write(fptr);
    Ur->fetch();
    Ur->Write(fptr);
    Wz->fetch();
    Wz->Write(fptr);
    Uz->fetch();
    Uz->Write(fptr);
}
void gruhighwaylayer::Write (FILE *fptr)
{
    Wh->fetch();
    Wh->Write(fptr);
    Uh->fetch();
    Uh->Write(fptr);
    Wr->fetch();
    Wr->Write(fptr);
    Ur->fetch();
    Ur->Write(fptr);
    Wz->fetch();
    Wz->Write(fptr);
    Uz->fetch();
    Uz->Write(fptr);
    // highway part
    Uhw->fetch();
    Uhw->Write(fptr);
    Whw->fetch();
    Whw->Write(fptr);
}
void lstmlayer::Write (FILE *fptr)
{
    Wz->fetch();
    Wz->Write (fptr);
    Uz->fetch();
    Uz->Write (fptr);
    Wi->fetch();
    Wi->Write (fptr);
    Ui->fetch();
    Ui->Write (fptr);
    Wf->fetch();
    Wf->Write (fptr);
    Uf->fetch();
    Uf->Write (fptr);
    Wo->fetch();
    Wo->Write (fptr);
    Uo->fetch();
    Uo->Write (fptr);
    Pi->fetch();
    Pi->Write(fptr);
    Pf->fetch();
    Pf->Write(fptr);
    Po->fetch();
    Po->Write(fptr);
}
void lstmhighwaylayer::Write (FILE *fptr)
{
    Wz->fetch();
    Wz->Write (fptr);
    Uz->fetch();
    Uz->Write (fptr);
    Wi->fetch();
    Wi->Write (fptr);
    Ui->fetch();
    Ui->Write (fptr);
    Wf->fetch();
    Wf->Write (fptr);
    Uf->fetch();
    Uf->Write (fptr);
    Wo->fetch();
    Wo->Write (fptr);
    Uo->fetch();
    Uo->Write (fptr);
    Pi->fetch();
    Pi->Write(fptr);
    Pf->fetch();
    Pf->Write(fptr);
    Po->fetch();
    Po->Write(fptr);
    // highway part
    Uhw->fetch();
    Uhw->Write(fptr);
    Phw->fetch();
    Phw->Write(fptr);
    Rhw->fetch();
    Rhw->Write(fptr);
}


void recurrentlayer::Read (FILE *fptr)
{
    int err;
    err = fscanf (fptr, "nodetype: %d\n", &nodetype);
    err = fscanf (fptr, "reluratio: %f\n", &reluratio);
    U->Read (fptr);
    W->Read (fptr);
    U->assign ();
    W->assign ();
}
void grulayer::Read (FILE *fptr)
{
    Wh->Read(fptr);
    Wh->assign();
    Uh->Read(fptr);
    Uh->assign();
    Wr->Read(fptr);
    Wr->assign();
    Ur->Read(fptr);
    Ur->assign();
    Wz->Read(fptr);
    Wz->assign();
    Uz->Read(fptr);
    Uz->assign();
}
void gruhighwaylayer::Read (FILE *fptr)
{
    Wh->Read(fptr);
    Wh->assign();
    Uh->Read(fptr);
    Uh->assign();
    Wr->Read(fptr);
    Wr->assign();
    Ur->Read(fptr);
    Ur->assign();
    Wz->Read(fptr);
    Wz->assign();
    Uz->Read(fptr);
    Uz->assign();
    // highway part
    Uhw->Read(fptr);
    Uhw->assign();
    Whw->Read(fptr);
    Whw->assign();
}
void lstmlayer::Read (FILE *fptr)
{
    Wz->Read (fptr);
    Wz->assign ();
    Uz->Read (fptr);
    Uz->assign ();
    Wi->Read (fptr);
    Wi->assign ();
    Ui->Read (fptr);
    Ui->assign ();
    Wf->Read (fptr);
    Wf->assign ();
    Uf->Read (fptr);
    Uf->assign ();
    Wo->Read (fptr);
    Wo->assign ();
    Uo->Read (fptr);
    Uo->assign ();
    Pi->Read (fptr);
    Pi->assign ();
    Pf->Read (fptr);
    Pf->assign ();
    Po->Read (fptr);
    Po->assign ();
}
void lstmhighwaylayer::Read (FILE *fptr)
{
    Wz->Read (fptr);
    Wz->assign ();
    Uz->Read (fptr);
    Uz->assign ();
    Wi->Read (fptr);
    Wi->assign ();
    Ui->Read (fptr);
    Ui->assign ();
    Wf->Read (fptr);
    Wf->assign ();
    Uf->Read (fptr);
    Uf->assign ();
    Wo->Read (fptr);
    Wo->assign ();
    Uo->Read (fptr);
    Uo->assign ();
    Pi->Read (fptr);
    Pi->assign ();
    Pf->Read (fptr);
    Pf->assign ();
    Po->Read (fptr);
    Po->assign ();
    // highway part
    Uhw->Read(fptr);
    Uhw->assign ();
    Phw->Read(fptr);
    Phw->assign ();
    Rhw->Read(fptr);
    Rhw->assign ();
}

void feedforwardlayer::Write (FILE *fptr)
{
    fprintf (fptr, "nodetype: %d\n", nodetype);
    fprintf (fptr, "reluratio: %f\n", reluratio);
    U->fetch ();
    U->Write (fptr);
}
void feedforwardlayer::Read (FILE *fptr)
{
    int err;
    err = fscanf (fptr, "nodetype: %d\n", &nodetype);
    err = fscanf (fptr, "reluratio: %f\n", &reluratio);
    U->Read (fptr);
    U->assign ();
}


void linearlayer::Write (FILE *fptr)
{
    U->fetch ();
    U->Write (fptr);
}
void linearlayer::Read (FILE *fptr)
{
    U->Read (fptr);
    U->assign ();
}

void recurrentlayer::host_resetHiddenac ()
{
    for (int i=0; i<ncols; i++)
    {
        hidden_ac->assignhostvalue (i, 0, RESETVALUE);
    }
}
void grulayer::host_resetHiddenac ()
{
    for (int i=0; i<ncols; i++)
    {
        hidden_ac->assignhostvalue (i, 0, RESETVALUE);
    }
}
void lstmlayer::host_resetHiddenac ()
{
    for (int i=0; i<ncols; i++)
    {
        hidden_ac->assignhostvalue (i, 0, RESETVALUE);
        c->assignhostvalue (i, 0, RESETVALUE);
    }
}

void outputlayer::host_forward (matrix *neu0_ac, matrix *neu1_ac)
{
    float *srcac = neu0_ac->gethostdataptr();
    float *dstac = neu1_ac->gethostdataptr();
    float *wgts  = U->gethostdataptr();
    float *succac = neu0_ac_succ->gethostdataptr ();
    memset (dstac, 0, ncols*sizeof(float));
    matrixXvector (srcac, wgts, dstac, nrows, ncols);
    neu1_ac->hostsoftmax ();
}

void recurrentlayer::host_forward (matrix *neu0_ac, matrix *neu1_ac)
{
    float *srcac = neu0_ac->gethostdataptr();
    float *dstac = neu1_ac->gethostdataptr();
    float *wgts  = U->gethostdataptr();
    memset (dstac, 0, ncols*sizeof(float));
    matrixXvector (srcac, wgts, dstac, nrows, ncols);
    float *hiddensrcac = hidden_ac->gethostdataptr();
    float *recwgts = W->gethostdataptr();
    matrixXvector (hiddensrcac, recwgts, dstac, ncols, ncols);
    if (nodetype == 0) // sigmoid
    {
        neu1_ac->hostsigmoid();
    }
    else if (nodetype == 1) // relue
    {
        neu1_ac->hostrelu (reluratio);
    }
    hidden_ac->hostassign (neu1_ac);
}

void grulayer::host_forward (matrix *neu0_ac, matrix *neu1_ac)
{
    float *host_Ur = Ur->gethostdataptr();
    float *host_Wr = Wr->gethostdataptr();
    float *host_Uz = Uz->gethostdataptr();
    float *host_Wz = Wz->gethostdataptr();
    float *host_Uh = Uh->gethostdataptr();
    float *host_Wh = Wh->gethostdataptr();

    float *host_srcac = neu0_ac->gethostdataptr();
    float *host_dstac = neu1_ac->gethostdataptr();
    float *host_hiddenac=hidden_ac->gethostdataptr();
    float *host_r = r->gethostdataptr();
    float *host_z = z->gethostdataptr();
    float *host_h_ = h_->gethostdataptr();

    memset (host_z, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uz, host_z, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wz, host_z, ncols, ncols);
    z->hostsigmoid();

    memset (host_r, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Ur, host_r, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wr, host_r, ncols, ncols);
    r->hostsigmoid();
    r->hostdotMultiply (hidden_ac);

    memset (host_h_, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uh, host_h_, nrows, ncols);
    matrixXvector (host_r, host_Wh, host_h_, ncols, ncols);
    h_->hosttanh();

    memset (host_dstac, 0, ncols*sizeof(float));
    hidden_ac->hostcalHiddenacGRU (h_, z);
    neu1_ac->hostassign (hidden_ac);
}


void gruhighwaylayer::host_forward (matrix *neu0_ac, matrix *neu1_ac)
{
    float *host_Ur = Ur->gethostdataptr();
    float *host_Wr = Wr->gethostdataptr();
    float *host_Uz = Uz->gethostdataptr();
    float *host_Wz = Wz->gethostdataptr();
    float *host_Uh = Uh->gethostdataptr();
    float *host_Wh = Wh->gethostdataptr();

    float *host_srcac = neu0_ac->gethostdataptr();
    float *host_dstac = neu1_ac->gethostdataptr();
    float *host_hiddenac=hidden_ac->gethostdataptr();
    float *host_r = r->gethostdataptr();
    float *host_z = z->gethostdataptr();
    float *host_h_ = h_->gethostdataptr();

    memset (host_z, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uz, host_z, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wz, host_z, ncols, ncols);
    z->hostsigmoid();

    memset (host_r, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Ur, host_r, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wr, host_r, ncols, ncols);
    r->hostsigmoid();
    r->hostdotMultiply (hidden_ac);

    memset (host_h_, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uh, host_h_, nrows, ncols);
    matrixXvector (host_r, host_Wh, host_h_, ncols, ncols);
    h_->hosttanh();

    memset (host_dstac, 0, ncols*sizeof(float));
    v->hostassign (hidden_ac);
    v->hostcalHiddenacGRU (h_, z);

    // highway part
    float *host_g = g->gethostdataptr ();
    float *host_Uhw = Uhw->gethostdataptr ();
    float *host_Whw = Whw->gethostdataptr ();
    memset (host_g, 0, ncols*sizeof(float));

    matrixXvector (host_srcac, host_Uhw, host_g, nrows, ncols);
    matrixXvector (host_hiddenac, host_Whw, host_g, nrows, ncols);
    g->hostsigmoid();

    hidden_ac->hostassign (neu0_ac);
    hidden_ac->hostcalHiddenacGRU (v, g);
    neu1_ac->hostassign (hidden_ac);
}

void lstmlayer::host_forward (matrix *neu0_ac, matrix *neu1_ac)
{
    float *host_Uz = Uz->gethostdataptr ();
    float *host_Wz = Wz->gethostdataptr ();
    float *host_Ui = Ui->gethostdataptr ();
    float *host_Wi = Wi->gethostdataptr ();
    float *host_Uf = Uf->gethostdataptr ();
    float *host_Wf = Wf->gethostdataptr ();
    float *host_Uo = Uo->gethostdataptr ();
    float *host_Wo = Wo->gethostdataptr ();
    float *host_srcac = neu0_ac->gethostdataptr ();
    float *host_dstac = neu1_ac->gethostdataptr ();
    float *host_hiddenac = hidden_ac->gethostdataptr ();
    float *host_z = z->gethostdataptr ();
    float *host_i = i->gethostdataptr ();
    float *host_f = f->gethostdataptr ();
    float *host_o = o->gethostdataptr ();

    // compute input gate
    memset (host_i, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Ui, host_i, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wi, host_i, ncols, ncols);
    i->hostadddotMultiply(c, Pi);
    i->hostsigmoid ();

    // compute forget gate
    memset (host_f, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uf, host_f, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wf, host_f, ncols, ncols);
    f->hostadddotMultiply (c, Pf);
    f->hostsigmoid ();
    f->hostdotMultiply (c);

    // compute block input
    memset (host_z, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uz, host_z, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wz, host_z, ncols, ncols);
    z->hosttanh ();
    z->hostdotMultiply (i);

    // compute cell state
    newc->hostassign (z);
    newc->hostadd (f);

    c->hostassign (newc);

    // compute output gate
    memset (host_o, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uo, host_o, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wo, host_o, ncols, ncols);
    o->hostadddotMultiply (newc, Po);
    o->hostsigmoid();

    hidden_ac->hostassign (newc);
    hidden_ac->hosttanh ();
    hidden_ac->hostdotMultiply (o);
    neu1_ac->hostassign (hidden_ac);
}

void lstmhighwaylayer::host_forward (matrix *neu0_ac, matrix *neu1_ac)
{
    float *host_Uz = Uz->gethostdataptr ();
    float *host_Wz = Wz->gethostdataptr ();
    float *host_Ui = Ui->gethostdataptr ();
    float *host_Wi = Wi->gethostdataptr ();
    float *host_Uf = Uf->gethostdataptr ();
    float *host_Wf = Wf->gethostdataptr ();
    float *host_Uo = Uo->gethostdataptr ();
    float *host_Wo = Wo->gethostdataptr ();
    float *host_srcac = neu0_ac->gethostdataptr ();
    float *host_dstac = neu1_ac->gethostdataptr ();
    float *host_hiddenac = hidden_ac->gethostdataptr ();
    float *host_z = z->gethostdataptr ();
    float *host_i = i->gethostdataptr ();
    float *host_f = f->gethostdataptr ();
    float *host_o = o->gethostdataptr ();

    // compute input gate
    memset (host_i, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Ui, host_i, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wi, host_i, nrows, ncols);
    i->hostadddotMultiply(c, Pi);
    i->hostsigmoid ();

    // compute forget gate
    memset (host_f, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uf, host_f, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wf, host_f, nrows, ncols);
    f->hostadddotMultiply (c, Pf);
    f->hostsigmoid ();
    f->hostdotMultiply (c);

    // compute block input
    memset (host_z, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uz, host_z, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wz, host_z, nrows, ncols);
    z->hosttanh ();
    z->hostdotMultiply (i);

    // compute cell state
    newc->hostassign (z);
    newc->hostadd (f);

    /******* highway part *********/
    float *host_s = s->gethostdataptr ();
    float *host_Uhw = Uhw->gethostdataptr ();
    memset (host_s, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uhw, host_s, nrows, ncols);
    s->hostadddotMultiply (c, Phw);
    s->hostadddotMultiply (c_hw, Rhw);
    s->hostsigmoid ();
    s->hostdotMultiply (c_hw);
    newc->hostadd (s);
    /******************************/

    c->hostassign (newc);

    // compute output gate
    memset (host_o, 0, ncols*sizeof(float));
    matrixXvector (host_srcac, host_Uo, host_o, nrows, ncols);
    matrixXvector (host_hiddenac, host_Wo, host_o, nrows, ncols);
    o->hostadddotMultiply (newc, Po);
    o->hostsigmoid();

    hidden_ac->hostassign(newc);
    hidden_ac->hosttanh ();
    hidden_ac->hostdotMultiply (o);
    neu1_ac->hostassign (hidden_ac);
}

void feedforwardlayer::host_forward_nosigm (matrix *neu0_ac, matrix *neu1_ac)
{
    float *srcac = neu0_ac->gethostdataptr();
    float *dstac = neu1_ac->gethostdataptr();
    float *wgts  = U->gethostdataptr();
    matrixXvector (srcac, wgts, dstac, nrows, ncols);
}

void feedforwardlayer::host_forward (matrix *neu0_ac, matrix *neu1_ac)
{
    float *srcac = neu0_ac->gethostdataptr();
    float *dstac = neu1_ac->gethostdataptr();
    float *wgts  = U->gethostdataptr();
    // memset (dstac, 0, ncols*sizeof(float));
    matrixXvector (srcac, wgts, dstac, nrows, ncols);
    if (nodetype == 0) // sigmoid
    {
        neu1_ac->hostsigmoid();
    }
    else if (nodetype == 1) // relue
    {
        neu1_ac->hostrelu (reluratio);
    }
}

void linearlayer::host_forward (matrix *neu0_ac, matrix *neu1_ac)
{
    float *srcac = neu0_ac->gethostdataptr();
    float *dstac = neu1_ac->gethostdataptr();
    float *wgts  = U->gethostdataptr();
    memset (dstac, 0, ncols*sizeof(float));
    matrixXvector (srcac, wgts, dstac, nrows, ncols);
}

void outputlayer::setSuccAc (matrix *ac)
{
    neu0_ac_succ = ac;
}

void outputlayer::setSuccEr (matrix *er)
{
    neu0_er_succ = er;
}

void outputlayer::forward (matrix *neu0_ac, matrix *neu1_ac)
{
    if (dropoutrate > 0)
    {
#ifndef VARIATIONALDROPOUT
        if (!evalmode)
        {
            dropoutmaskMat_vec[chunkiter]->gendropoutmask (dropoutrate);
        }
#endif
        neu0_ac->dropout (dropoutmaskMat_vec[chunkiter], dropoutrate, evalmode);
    }
    if (traincrit == 0)                     // CE
    {
        cumatrixXmatrix (U, neu0_ac, neu1_ac, true, false, 1.0, 0.0);
        neu1_ac->softmax(lognormvec[chunkiter]);
    }
    else if (traincrit == 1)                // VR
    {
        cumatrixXmatrix (U, neu0_ac, neu1_ac, true, false, 1.0, 0.0);
        neu1_ac->softmax(lognormvec[chunkiter]);
        // compute the part of  (z - avg(z)) for gradient calculation
        lognormvec[chunkiter]->addScalar (-1*lognorm);
    }
    else if (traincrit == 2)                // NCE
    {
        if (evalmode)
        {
            cumatrixXmatrix (U, neu0_ac, neu1_ac, true, false, 1.0, 0.0);
            neu1_ac->softmax(lognormvec[chunkiter]);
        }
        else
        {
            genNCESample ();
            cumatrixXmatrix (layerN_NCE_vec[chunkiter], neu0_ac, neuN_ac_NCE_vec[chunkiter], true, false, 1.0, 0.0);
        }
    }
    else
    {
        printf ("Error: unknown train criterion!\n");
        exit (0);
    }
}

void recurrentlayer::forward (matrix *neu0_ac, matrix *neu1_ac)
{
    if (dropoutrate > 0)
    {
        if (!evalmode)
        {
            dropoutmaskMat_vec[chunkiter]->gendropoutmask (dropoutrate);
        }
        neu0_ac->dropout (dropoutmaskMat_vec[chunkiter], dropoutrate, evalmode);
    }
    hidden_ac = hidden_ac_vec[chunkiter];

    cumatrixXmatrix (U, neu0_ac, neu1_ac, true, false, 1.0, 0.0);
    cumatrixXmatrix (W, hidden_ac, neu1_ac, true, false, 1.0, 1.0);
    if (nodetype == 0)   // sigmoid
    {
        neu1_ac->sigmoid();
    }
    else if (nodetype == 1)  //relu
    {
        neu1_ac->relu (reluratio);
    }
    // copy for recurrent connection
    hidden_ac_vec[chunkiter+1]->assign(neu1_ac);
}

void grulayer::forward (matrix *neu0_ac, matrix *neu1_ac)
{
    if(dropoutrate > 0)
    {
        if (!evalmode)
        {
            dropoutmaskMat_vec[chunkiter]->gendropoutmask (dropoutrate);
        }
        neu0_ac->dropout (dropoutmaskMat_vec[chunkiter], dropoutrate, evalmode);
    }
    hidden_ac = hidden_ac_vec[chunkiter];
    r = r_vec[chunkiter];
    z = z_vec[chunkiter];
    c = c_vec[chunkiter];
    h_ = h_vec[chunkiter];

    // compute update gate
    cumatrixXmatrix (Uz, neu0_ac, z, true, false, 1.0, 0.0);
    cumatrixXmatrix (Wz, hidden_ac, z, true, false, 1.0, 1.0);
    z->sigmoid();

    // compute reset gate
    cumatrixXmatrix (Ur, neu0_ac, r, true, false, 1.0, 0.0);
    cumatrixXmatrix (Wr, hidden_ac, r, true, false, 1.0, 1.0);
    r->sigmoid();
    c->assign (r);
    c->dotMultiply (hidden_ac);

    // compute candidate ac h_
    cumatrixXmatrix (Uh, neu0_ac, h_, true, false, 1.0, 0.0);
    cumatrixXmatrix (Wh, c, h_, true, false, 1.0, 1.0);
    h_->tanh();

    // compute neu1_ac & hidden_ac
    hidden_ac_vec[chunkiter+1]->calHiddenacGRU (hidden_ac, h_, z);
    neu1_ac->assign(hidden_ac_vec[chunkiter+1]);
}

void gruhighwaylayer::forward (matrix *neu0_ac, matrix *neu1_ac)
{
    // copy forward from grulayer
    // TODO: simplify it later
    // compute update gate
    if(dropoutrate > 0)
    {
        if (!evalmode)
        {
            dropoutmaskMat_vec[chunkiter]->gendropoutmask (dropoutrate);
        }
        neu0_ac->dropout (dropoutmaskMat_vec[chunkiter], dropoutrate, evalmode);
    }
    hidden_ac = hidden_ac_vec[chunkiter];
    r = r_vec[chunkiter];
    z = z_vec[chunkiter];
    c = c_vec[chunkiter];
    h_ = h_vec[chunkiter];
    v = v_vec[chunkiter];
    g = g_vec[chunkiter];

    cumatrixXmatrix (Uz, neu0_ac, z, true, false, 1.0, 0.0);
    cumatrixXmatrix (Wz, hidden_ac, z, true, false, 1.0, 1.0);
    z->sigmoid();

    // compute reset gate
    cumatrixXmatrix (Ur, neu0_ac, r, true, false, 1.0, 0.0);
    cumatrixXmatrix (Wr, hidden_ac, r, true, false, 1.0, 1.0);
    r->sigmoid();
    c->assign (r);
    c->dotMultiply (hidden_ac);

    // compute candidate ac h_
    cumatrixXmatrix (Uh, neu0_ac, h_, true, false, 1.0, 0.0);
    cumatrixXmatrix (Wh, c, h_, true, false, 1.0, 1.0);
    h_->tanh();

    // gru block output
    v->calHiddenacGRU (hidden_ac, h_, z); // modify it later

    // highway part
    cumatrixXmatrix (Uhw, neu0_ac, g, true, false, 1.0, 0.0);
    cumatrixXmatrix (Whw, hidden_ac, g, true, false, 1.0, 1.0);
    g->sigmoid();

    // compute nue1_ac & hidden_ac
    hidden_ac_vec[chunkiter+1]->calHiddenacGRU (neu0_ac, v, g);
    neu1_ac->assign(hidden_ac_vec[chunkiter+1]);
}

void lstmlayer::forward (matrix *neu0_ac, matrix *neu1_ac)
{
    if (dropoutrate > 0)
    {
#ifdef VARIATIONALDROPOUT
        neu0_ac->dropout (dropoutmaskMat_vec[chunkiter], dropoutrate, evalmode);
        hidden_ac_vec[chunkiter]->dropout (dropoutHiddenmaskMat_vec[chunkiter], dropoutrate, evalmode);
#else
        if (!evalmode)
        {
            dropoutmaskMat_vec[chunkiter]->gendropoutmask (dropoutrate);
        }
        neu0_ac->dropout (dropoutmaskMat_vec[chunkiter], dropoutrate, evalmode);
#endif
    }
    hidden_ac = hidden_ac_vec[chunkiter];
    c = c_vec[chunkiter];
    z = z_vec[chunkiter];
    i = i_vec[chunkiter];
    f = f_vec[chunkiter];
    o = o_vec[chunkiter];
    newc = newc_vec[chunkiter];

    // compute block input
    cumatrixXmatrix (Uz, neu0_ac, z, true, false, 1.0, 0.0);
    cumatrixXmatrix (Wz, hidden_ac, z, true, false, 1.0, 1.0);
    z->tanh ();

    // compute input gate
    cumatrixXmatrix (Ui, neu0_ac, i, true, false, 1.0, 0.0);
    cumatrixXmatrix (Wi, hidden_ac, i, true, false, 1.0, 1.0);
    i->adddotMultiply (Pi, c);
    i->sigmoid();

    // compute forget gate
    cumatrixXmatrix (Uf, neu0_ac, f, true, false, 1.0, 0.0);
    cumatrixXmatrix (Wf, hidden_ac, f, true, false, 1.0, 1.0);
    f->adddotMultiply (Pf, c);
    f->sigmoid();

    // compute cell state
    zi->assign (i);
    zi->dotMultiply (z);
    fc->assign (f);
    fc->dotMultiply (c);

    newc->assign (zi);
    newc->add (fc);

    // compute output gate
    cumatrixXmatrix (Uo, neu0_ac, o, true, false, 1.0, 0.0);
    cumatrixXmatrix (Wo, hidden_ac, o, true, false, 1.0, 1.0);
    o->adddotMultiply (Po, newc);
    o->sigmoid ();

    // compute block output
    hidden_ac_vec[chunkiter+1]->assign (newc);
    hidden_ac_vec[chunkiter+1]->tanh ();
    hidden_ac_vec[chunkiter+1]->dotMultiply (o);
    neu1_ac->assign (hidden_ac_vec[chunkiter+1]);
    c_vec[chunkiter+1]->assign(newc);
}

void lstmhighwaylayer::forward (matrix *neu0_ac, matrix *neu1_ac)
{
    if (dropoutrate > 0)
    {
        if (!evalmode)
        {
            dropoutmaskMat_vec[chunkiter]->gendropoutmask (dropoutrate);
        }
        neu0_ac->dropout (dropoutmaskMat_vec[chunkiter], dropoutrate, evalmode);
    }
    hidden_ac = hidden_ac_vec[chunkiter];
    c = c_vec[chunkiter];
    z = z_vec[chunkiter];
    i = i_vec[chunkiter];
    f = f_vec[chunkiter];
    o = o_vec[chunkiter];
    newc = newc_vec[chunkiter];
    // highway part
    s = s_vec[chunkiter];
    c_hw = c_hw_vec[chunkiter];

    // standard LSTM forward
    // compute block input
    cumatrixXmatrix (Uz, neu0_ac, z, true, false, 1.0, 0.0);
    cumatrixXmatrix (Wz, hidden_ac, z, true, false, 1.0, 1.0);
    z->tanh ();

    // compute input gate
    cumatrixXmatrix (Ui, neu0_ac, i, true, false, 1.0, 0.0);
    cumatrixXmatrix (Wi, hidden_ac, i, true, false, 1.0, 1.0);
    i->adddotMultiply (Pi, c);
    i->sigmoid();

    // compute forget gate
    cumatrixXmatrix (Uf, neu0_ac, f, true, false, 1.0, 0.0);
    cumatrixXmatrix (Wf, hidden_ac, f, true, false, 1.0, 1.0);
    f->adddotMultiply (Pf, c);
    f->sigmoid();

    // compute cell state
    zi->assign (i);
    zi->dotMultiply (z);
    fc->assign (f);
    fc->dotMultiply (c);

    newc->assign (zi);
    newc->add (fc);

    /****** highway part ********/
    // highway gate
    cumatrixXmatrix (Uhw, neu0_ac, s, true, false, 1.0, 0.0);
    s->adddotMultiply (Phw, c);
    s->adddotMultiply (Rhw, c_hw);
    s->sigmoid();
    // scale with c from last layer
    sc->assign (c_hw);
    sc->dotMultiply (s);
    newc->add (sc);
    /****************************/

    // compute output gate
    cumatrixXmatrix (Uo, neu0_ac, o, true, false, 1.0, 0.0);
    cumatrixXmatrix (Wo, hidden_ac, o, true, false, 1.0, 1.0);
    o->adddotMultiply (Po, newc);
    o->sigmoid ();

    // compute block output
    hidden_ac_vec[chunkiter+1]->assign (newc);
    hidden_ac_vec[chunkiter+1]->tanh ();
    hidden_ac_vec[chunkiter+1]->dotMultiply (o);
    neu1_ac->assign (hidden_ac_vec[chunkiter+1]);
    c_vec[chunkiter+1]->assign(newc);
}


void feedforwardlayer::forward (matrix *neu0_ac, matrix *neu1_ac)
{
    cumatrixXmatrix (U, neu0_ac, neu1_ac, true, false, 1.0, 1.0);
    if (nodetype == 0)   // sigmoid
    {
        neu1_ac->sigmoid();
    }
    else if (nodetype == 1)  //relu
    {
        neu1_ac->relu (reluratio);
    }
}

void feedforwardlayer::forward_nosigm (matrix *neu0_ac, matrix *neu1_ac)
{
    cumatrixXmatrix (U, neu0_ac, neu1_ac, true, false, 1.0, 1.0);
}



void linearlayer::forward (matrix *neu0_ac, matrix *neu1_ac)
{
    cumatrixXmatrix (U, neu0_ac, neu1_ac, true, false, 1.0, 0.0);
}

void outputlayer::calerr (matrix *neu_ac, matrix *neu_er, int *dev_curwords)
{
    if (traincrit == 0)         // CE training
    {
        neu_er->calerronoutputlayer (neu_ac, dev_curwords);
    }
    else if (traincrit == 1)    // VR training
    {
        neu_er->calerronoutputlayer_vr (neu_ac, dev_curwords, lognormvec[chunkiter], vrpenalty);
    }
    else if (traincrit == 2)    // NCE training
    {
        neuN_er_NCE->setnrows(outputlayersize_NCE[chunkiter]);
        neuN_ac_NCE_vec[chunkiter]->setnrows(outputlayersize_NCE[chunkiter]);
        neuN_er_NCE->calerronOutputLayer (neuN_ac_NCE_vec[chunkiter], logwordnoise->getdevdataptr(), dev_curwords, dev_targetsample[chunkiter], dev_ncesample[chunkiter], dev_ncesamplecnt[chunkiter], dev_mbid2arrid[chunkiter], ntargetsample[chunkiter], nncesample[chunkiter]);
    }
    else
    {
        return;
    }
}

void inputlayer::backward (matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac)
{
    printf ("this function should not be used for input layer!\n");
}

void outputlayer::backward (matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac)
{
    if (traincrit == 0 || traincrit == 1)
    {
        cumatrixXmatrix (U, neu1_er, neu0_er, false, false, 1.0, 0.0);
        cumatrixXmatrix (neu0_ac, neu1_er, dU, false, true, 1.0, 1.0);
    }
    else if (traincrit == 2)                // NCE
    {
        // compute error signal
        layerN_NCE_vec[chunkiter]->setncols(outputlayersize_NCE[chunkiter]);

        gradlayerN_NCE->setncols(outputlayersize_NCE[chunkiter]);

        neuN_er_NCE->setnrows(outputlayersize_NCE[chunkiter]);

        cumatrixXmatrix (layerN_NCE_vec[chunkiter], neuN_er_NCE, neu0_er, false, false, 1.0, 0.0);

        // compute gradient in the output layer
        cumatrixXmatrix (neu0_ac, neuN_er_NCE, gradlayerN_NCE, false, true, 1.0, 0.0);

        if (gradient_cutoff > 0)
        {
            gradlayerN_NCE->L2norm (gradient_cutoff, cunorm2ptr, minibatch);
        }

        dU->addgrad_NCE (gradlayerN_NCE, dev_targetsample[chunkiter], ntargetsample[chunkiter], dev_ncesample[chunkiter], nncesample[chunkiter], 1.0);
    }
    else
    {
        printf ("Error: unknown train criterion!\n");
        exit (0);
    }
    if (dropoutrate > 0)
    {
        neu0_er->dotMultiply (dropoutmaskMat_vec[chunkiter]);
        neu0_er_succ->dotMultiply(dropoutmaskMat_vec[chunkiter]);
    }
}

void recurrentlayer::backward (matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac)
{
    hidden_ac = hidden_ac_vec[chunkiter];

    neu1_er->add (hidden_er);
    if (nodetype == 0)
    {
        neu1_er->multiplysigmoid (neu1_ac);
    }
    else
    {
        neu1_er->multiplyrelue (neu1_ac, reluratio);
    }
    //  to previous layer
    cumatrixXmatrix (U, neu1_er, neu0_er, false, false, 1.0, 0.0);
    cumatrixXmatrix (neu0_ac, neu1_er, dU, false, true, 1.0, 1.0);

    // to previous word
    cumatrixXmatrix (W, neu1_er, hidden_er, false, false, 1.0, 0.0);
    cumatrixXmatrix (hidden_ac, neu1_er, dW, false, true, 1.0, 1.0);
    if (dropoutrate > 0)
    {
        neu0_er->dotMultiply (dropoutmaskMat_vec[chunkiter]);
    }
}

void grulayer::backward (matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac)
{
    hidden_ac = hidden_ac_vec[chunkiter];
    r = r_vec[chunkiter];
    z = z_vec[chunkiter];
    c = c_vec[chunkiter];
    h_= h_vec[chunkiter];

    neu1_er->add (hidden_er);
    hidden_er->initmatrix ();
    // step 1
    hidden_er->addOneMinus(z);
    hidden_er->dotMultiply(neu1_er);
    // step 2
    dz->assign(h_);
    dz->subtract (hidden_ac);
    dz->multiplysigmoid (z);
    dz->dotMultiply (neu1_er);
    // step 3
    cumatrixXmatrix (neu0_ac, dz, dUz, false, true, 1.0, 1.0);
    cumatrixXmatrix (hidden_ac, dz, dWz, false, true, 1.0, 1.0);
    // step 4
    cumatrixXmatrix (Wz, dz, hidden_er, false, false, 1.0, 1.0);
    cumatrixXmatrix (Uz, dz, neu0_er, false, false, 1.0, 0.0);
    // step 5
    dh_->assign (z);
    dh_->multiplytanh (h_);
    dh_->dotMultiply (neu1_er);
    // step 6
    cumatrixXmatrix (neu0_ac, dh_, dUh, false, true, 1.0, 1.0);
    cumatrixXmatrix (c, dh_, dWh, false, true, 1.0, 1.0);
    // step 7
    cumatrixXmatrix (Wh, dh_, dc, false, false, 1.0, 0.0);
    hidden_er->addProduct (r, dc);
    cumatrixXmatrix (Uh, dh_, neu0_er, false, false, 1.0, 1.0);
    // step 8
    dr->assign(dc);
    dr->dotMultiply (hidden_ac);
    dr->multiplysigmoid (r);
    // step 9
    cumatrixXmatrix (neu0_ac, dr, dUr, false, true, 1.0, 1.0);
    cumatrixXmatrix (hidden_ac, dr, dWr, false, true, 1.0, 1.0);
    // step 10
    cumatrixXmatrix (Wr, dr, hidden_er, false, false, 1.0, 1.0);
    cumatrixXmatrix (Ur, dr, neu0_er, false, false, 1.0, 1.0);
    if (dropoutrate > 0)
    {
        neu0_er->dotMultiply (dropoutmaskMat_vec[chunkiter]);
    }
}

void gruhighwaylayer::backward (matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac)
{
    hidden_ac = hidden_ac_vec[chunkiter];
    r = r_vec[chunkiter];
    z = z_vec[chunkiter];
    c = c_vec[chunkiter];
    h_= h_vec[chunkiter];
    v = v_vec[chunkiter];
    g = g_vec[chunkiter];

    // step 1
    neu1_er->add(hidden_er);
    hidden_er->initmatrix();
    neu0_er->initmatrix ();

    // step 2
    neu0_er->addOneMinus (g);
    neu0_er->dotMultiply (neu1_er);

    dg->assign (v);
    dg->subtract (neu0_ac);
    dg->dotMultiply (neu1_er);
    dg->multiplysigmoid (g);

    // step 3
    cumatrixXmatrix (neu0_ac, dg, dUhw, false, true, 1.0, 1.0);
    cumatrixXmatrix (hidden_ac, dg, dWhw, false, true, 1.0, 1.0);
    cumatrixXmatrix (Uhw, dg, neu0_er, false, false, 1.0, 1.0);
    cumatrixXmatrix (Whw, dg, hidden_er_highway, false, false, 1.0, 0.0);

    // backward in GRU block
    dv->assign (g);
    neu1_er->dotMultiply (dv);
    dg->initmatrix ();

    // step 1
    hidden_er->addOneMinus(z);
    hidden_er->dotMultiply(neu1_er);
    // step 2
    dz->assign(h_);
    dz->subtract (hidden_ac);
    dz->multiplysigmoid (z);
    dz->dotMultiply (neu1_er);
    // step 3
    cumatrixXmatrix (neu0_ac, dz, dUz, false, true, 1.0, 1.0);
    cumatrixXmatrix (hidden_ac, dz, dWz, false, true, 1.0, 1.0);
    // step 4
    cumatrixXmatrix (Wz, dz, hidden_er, false, false, 1.0, 1.0);
    cumatrixXmatrix (Uz, dz, neu0_er, false, false, 1.0, 1.0);
    // step 5
    dh_->assign (z);
    dh_->multiplytanh (h_);
    dh_->dotMultiply (neu1_er);
    // step 6
    cumatrixXmatrix (neu0_ac, dh_, dUh, false, true, 1.0, 1.0);
    cumatrixXmatrix (c, dh_, dWh, false, true, 1.0, 1.0);
    // step 7
    cumatrixXmatrix (Wh, dh_, dc, false, false, 1.0, 0.0);
    hidden_er->addProduct (r, dc);
    cumatrixXmatrix (Uh, dh_, neu0_er, false, false, 1.0, 1.0);
    // step 8
    dr->assign(dc);
    dr->dotMultiply (hidden_ac);
    dr->multiplysigmoid (r);
    // step 9
    cumatrixXmatrix (neu0_ac, dr, dUr, false, true, 1.0, 1.0);
    cumatrixXmatrix (hidden_ac, dr, dWr, false, true, 1.0, 1.0);
    // step 10
    cumatrixXmatrix (Wr, dr, hidden_er, false, false, 1.0, 1.0);
    cumatrixXmatrix (Ur, dr, neu0_er, false, false, 1.0, 1.0);

    // add hidden_er from GRU and highway block
    hidden_er->add(hidden_er_highway);
    if (dropoutrate > 0)
    {
        neu0_er->dotMultiply (dropoutmaskMat_vec[chunkiter]);
    }
}

void lstmlayer::backward (matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac)
{
    hidden_ac = hidden_ac_vec[chunkiter];
    c = c_vec[chunkiter];
    z = z_vec[chunkiter];
    i = i_vec[chunkiter];
    f = f_vec[chunkiter];
    o = o_vec[chunkiter];
    newc = newc_vec[chunkiter];

    // step 1
    neu1_er->add (hidden_er);
    hidden_er->initmatrix ();
    // step 2
    do_->assign (newc);
    do_->tanh();
    do_->multiplysigmoid (o);
    do_->dotMultiply (neu1_er);
    // step 3
    cumatrixXmatrix (neu0_ac, do_, dUo, false, true, 1.0, 1.0);
    cumatrixXmatrix (hidden_ac, do_, dWo, false, true, 1.0, 1.0);
    dPo->addProduct (do_, newc);
    cumatrixXmatrix (Uo, do_, neu0_er, false, false, 1.0, 0.0);
    cumatrixXmatrix (Wo, do_, hidden_er, false, false, 1.0, 0.0);
    c_er->adddotMultiply (Po, do_);
    // step 4
    dnewc->assign (o);
    newc->tanh();
    dnewc->multiplytanh (newc);
    dnewc->dotMultiply (neu1_er);
    dnewc->add (c_er);
    c_er->initmatrix();
    // step 5
    c_er->addProduct (dnewc, f);
    // step 6
    df->initmatrix ();
    df->addProduct (dnewc, c);
    df->multiplysigmoid (f);
    cumatrixXmatrix (neu0_ac, df, dUf, false, true, 1.0, 1.0);
    cumatrixXmatrix (hidden_ac, df, dWf, false, true, 1.0, 1.0);
    dPf->addProduct (df, c);
    cumatrixXmatrix (Uf, df, neu0_er, false, false, 1.0, 1.0);
    cumatrixXmatrix (Wf, df, hidden_er, false, false, 1.0, 1.0);
    c_er->adddotMultiply (Pf, df);
    // step 7
    di->initmatrix ();
    di->addProduct (dnewc, z);
    di->multiplysigmoid (i);
    cumatrixXmatrix (neu0_ac, di, dUi, false, true, 1.0, 1.0);
    cumatrixXmatrix (hidden_ac, di, dWi, false, true, 1.0, 1.0);
    dPi->addProduct (di, c);
    cumatrixXmatrix (Ui, di ,neu0_er, false, false, 1.0, 1.0);
    cumatrixXmatrix (Wi, di, hidden_er, false, false, 1.0, 1.0);
    c_er->adddotMultiply (Pi, di);
    // step 8
    dz->initmatrix ();
    dz->addProduct (dnewc, i);
    dz->multiplytanh (z);
    cumatrixXmatrix (neu0_ac, dz, dUz, false, true, 1.0, 1.0);
    cumatrixXmatrix (hidden_ac, dz, dWz, false, true, 1.0, 1.0);
    cumatrixXmatrix (Uz, dz, neu0_er, false, false, 1.0, 1.0);
    cumatrixXmatrix (Wz, dz, hidden_er, false, false, 1.0, 1.0);
    if (dropoutrate > 0)
    {
#ifdef VARIATIONALDROPOUT
        neu0_er->dotMultiply (dropoutmaskMat_vec[chunkiter]);
        hidden_er->dotMultiply (dropoutHiddenmaskMat_vec[chunkiter]);
#else
        neu0_er->dotMultiply (dropoutmaskMat_vec[chunkiter]);
#endif
    }
}

void lstmhighwaylayer::backward (matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac)
{
    hidden_ac = hidden_ac_vec[chunkiter];
    c = c_vec[chunkiter];
    z = z_vec[chunkiter];
    i = i_vec[chunkiter];
    f = f_vec[chunkiter];
    o = o_vec[chunkiter];
    newc = newc_vec[chunkiter];
    s = s_vec[chunkiter];
    c_hw = c_hw_vec[chunkiter];

    // standard LSTM backward
    // step 1
    neu1_er->add (hidden_er);
    hidden_er->initmatrix ();
    // step 2
    do_->assign (newc);
    do_->tanh();
    do_->multiplysigmoid (o);
    do_->dotMultiply (neu1_er);
    // step 3
    cumatrixXmatrix (neu0_ac, do_, dUo, false, true, 1.0, 1.0);
    cumatrixXmatrix (hidden_ac, do_, dWo, false, true, 1.0, 1.0);
    dPo->addProduct (do_, newc);
    cumatrixXmatrix (Uo, do_, neu0_er, false, false, 1.0, 0.0);
    cumatrixXmatrix (Wo, do_, hidden_er, false, false, 1.0, 0.0);
    c_er->adddotMultiply (Po, do_);
    // step 4
    dnewc->assign (o);
    newc->tanh();
    dnewc->multiplytanh (newc);
    dnewc->dotMultiply (neu1_er);
    dnewc->add (c_er);
    c_er->initmatrix();
    // step 5
    c_er->addProduct (dnewc, f);
    // step 6
    df->initmatrix ();
    df->addProduct (dnewc, c);
    df->multiplysigmoid (f);
    cumatrixXmatrix (neu0_ac, df, dUf, false, true, 1.0, 1.0);
    cumatrixXmatrix (hidden_ac, df, dWf, false, true, 1.0, 1.0);
    dPf->addProduct (df, c);
    cumatrixXmatrix (Uf, df, neu0_er, false, false, 1.0, 1.0);
    cumatrixXmatrix (Wf, df, hidden_er, false, false, 1.0, 1.0);
    c_er->adddotMultiply (Pf, df);
    // step 7
    di->initmatrix ();
    di->addProduct (dnewc, z);
    di->multiplysigmoid (i);
    cumatrixXmatrix (neu0_ac, di, dUi, false, true, 1.0, 1.0);
    cumatrixXmatrix (hidden_ac, di, dWi, false, true, 1.0, 1.0);
    dPi->addProduct (di, c);
    cumatrixXmatrix (Ui, di ,neu0_er, false, false, 1.0, 1.0);
    cumatrixXmatrix (Wi, di, hidden_er, false, false, 1.0, 1.0);
    c_er->adddotMultiply (Pi, di);
    // step 8
    dz->initmatrix ();
    dz->addProduct (dnewc, i);
    dz->multiplytanh (z);
    cumatrixXmatrix (neu0_ac, dz, dUz, false, true, 1.0, 1.0);
    cumatrixXmatrix (hidden_ac, dz, dWz, false, true, 1.0, 1.0);
    cumatrixXmatrix (Uz, dz, neu0_er, false, false, 1.0, 1.0);
    cumatrixXmatrix (Wz, dz, hidden_er, false, false, 1.0, 1.0);

    // highway part
    // step 1
    c_er_hw->initmatrix ();
    c_er_hw->addProduct (dnewc, s);
    ds->initmatrix ();
    ds->addProduct (dnewc, c_hw);
    // step 2
    ds->multiplysigmoid (s);
    cumatrixXmatrix (neu0_ac, ds, dUhw, false, true, 1.0, 1.0);
    dPhw->addProduct (ds, c);
    dRhw->addProduct (ds, c_hw);
    cumatrixXmatrix (Uhw, ds, neu0_er, false, false, 1.0, 1.0);
    c_er->adddotMultiply (Phw, ds);
    c_er_hw->adddotMultiply (Rhw, ds);
    if (dropoutrate > 0)
    {
        neu0_er->dotMultiply (dropoutmaskMat_vec[chunkiter]);
    }
}

void lstmhighwaylayer::copyLSTMhighwayc (layer *layer0, int chunkiter)
{
    lstmlayer *lstmlayer0 = dynamic_cast <lstmlayer *> (layer0);
    matrix *srcc = lstmlayer0->c_vec[chunkiter+1];
    c_hw_vec[chunkiter]->assign (srcc);
}
void lstmlayer::copyLSTMhighwayc_er (layer *layer1, int chunkiter)
{
    lstmhighwaylayer *lstmlayer1 = dynamic_cast<lstmhighwaylayer *> (layer1);
    matrix *srcc_er = lstmlayer1->c_er_hw;
    c_er->add (srcc_er);
}

void lstmhighwaylayer::host_copyLSTMhighwayc (layer *layer0)
{
    lstmlayer *lstmlayer0 = dynamic_cast <lstmlayer *> (layer0);
    c_hw->hostassign (lstmlayer0->newc);
}

void feedforwardlayer::backward (matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac)
{
    if (nodetype == 0)
    {
        neu1_er->multiplysigmoid (neu1_ac);
    }
    else
    {
        neu1_er->multiplyrelue (neu1_ac, reluratio);
    }
    cumatrixXmatrix (U, neu1_er, neu0_er, false, false, 1.0, 0.0);

    cumatrixXmatrix (neu0_ac, neu1_er, dU, false, true, 1.0, 1.0);
}

void feedforwardlayer::backward_succ (matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac)
{
    cumatrixXmatrix (U, neu1_er, neu0_er, false, false, 1.0, 0.0);
    cumatrixXmatrix (neu0_ac, neu1_er, dU, false, true, 1.0, 1.0);
}

void linearlayer::backward (matrix *neu1_er, matrix *neu1_ac, matrix *neu0_er, matrix *neu0_ac)
{
    cumatrixXmatrix (U, neu1_er, neu0_er, false, false, 1.0, 0.0);

    cumatrixXmatrix (neu0_ac, neu1_er, dU, false, true, 1.0, 1.0);
}


void outputlayer::update(float alpha)
{
    if (gradient_cutoff > 0 && traincrit != 2)
    {
        dU->L2norm (gradient_cutoff, cunorm2ptr, minibatch);
    }
    if (lrtunemode == 0)
    {
        U->addgrad (dU, alpha, l2reg);
    }
    else if (lrtunemode == 1)
    {
        accsdU->addsquaregrad (dU, 1.0, 1.0);
        U->addadagrad (dU, accsdU, alpha, l2reg);
    }
    else if (lrtunemode == 2)
    {
        // accsdU = accsdU * gamma + dU * beta
        accsdU->addsquaregrad (dU, gamma, 1-gamma);
        U->addadagrad (dU, accsdU, alpha, l2reg);
    }

#ifdef MOMENTUM
    dU->multiplyScalar (momentum);
#else
    dU->initmatrix();
#endif
}

void recurrentlayer::update (float alpha)
{
    if (lrtunemode == 0)
    {
        U->addgrad (dU, alpha, l2reg);
        W->addgrad (dW, alpha, l2reg);
    }
    else if (lrtunemode == 1)
    {
        accsdU->addsquaregrad (dU, 1.0, 1.0);
        U->addadagrad (dU, accsdU, alpha, l2reg);
        accsdW->addsquaregrad (dW, 1.0, 1.0);
        W->addadagrad (dW, accsdW, alpha, l2reg);
    }
    else if (lrtunemode == 2)
    {
        accsdU->addsquaregrad (dU, gamma, 1-gamma);
        U->addadagrad (dU, accsdU, alpha, l2reg);
        accsdW->addsquaregrad (dW, gamma, 1-gamma);
        W->addadagrad (dW, accsdW, alpha, l2reg);
    }
#ifdef MOMENTUM
    dU->multiplyScalar (momentum);
    dW->multiplyScalar (momentum);
#else
    dU->initmatrix();
    dW->initmatrix();
#endif
}


void grulayer::update (float alpha)
{
    if (lrtunemode == 0)
    {
        Uz->addgrad (dUz, alpha, l2reg);
        Wz->addgrad (dWz, alpha, l2reg);
        Ur->addgrad (dUr, alpha, l2reg);
        Wr->addgrad (dWr, alpha, l2reg);
        Uh->addgrad (dUh, alpha, l2reg);
        Wh->addgrad (dWh, alpha, l2reg);
    }
    else if (lrtunemode == 1)
    {
        accsdUz->addsquaregrad (dUz, 1.0, 1.0);
        Uz->addadagrad (dUz, accsdUz, alpha, l2reg);
        accsdWz->addsquaregrad (dWz, 1.0, 1.0);
        Wz->addadagrad (dWz, accsdWz, alpha, l2reg);
        accsdUr->addsquaregrad (dUr, 1.0, 1.0);
        Ur->addadagrad (dUr, accsdUr, alpha, l2reg);
        accsdWr->addsquaregrad (dWr, 1.0, 1.0);
        Wr->addadagrad (dWr, accsdWr, alpha, l2reg);
        accsdUh->addsquaregrad (dUh, 1.0, 1.0);
        Uh->addadagrad (dUh, accsdUh, alpha, l2reg);
        accsdWh->addsquaregrad (dWh, 1.0, 1.0);
        Wh->addadagrad (dWh, accsdWh, alpha, l2reg);
    }
    else if (lrtunemode == 2)
    {
        accsdUz->addsquaregrad (dUz, gamma, 1-gamma);
        Uz->addadagrad (dUz, accsdUz, alpha, l2reg);
        accsdWz->addsquaregrad (dWz, gamma, 1-gamma);
        Wz->addadagrad (dWz, accsdWz, alpha, l2reg);
        accsdUr->addsquaregrad (dUr, gamma, 1-gamma);
        Ur->addadagrad (dUr, accsdUr, alpha, l2reg);
        accsdWr->addsquaregrad (dWr, gamma, 1-gamma);
        Wr->addadagrad (dWr, accsdWr, alpha, l2reg);
        accsdUh->addsquaregrad (dUh, gamma, 1-gamma);
        Uh->addadagrad (dUh, accsdUh, alpha, l2reg);
        accsdWh->addsquaregrad (dWh, gamma, 1-gamma);
        Wh->addadagrad (dWh, accsdWh, alpha, l2reg);
    }

#ifdef MOMENTUM
    dUz->multiplyScalar (momentum);
    dWz->multiplyScalar (momentum);
    dUr->multiplyScalar (momentum);
    dWr->multiplyScalar (momentum);
    dUh->multiplyScalar (momentum);
    dWh->multiplyScalar (momentum);
#else
    dUz->initmatrix ();
    dWz->initmatrix ();
    dUr->initmatrix ();
    dWr->initmatrix ();
    dUh->initmatrix ();
    dWh->initmatrix ();
#endif
}

void gruhighwaylayer::update (float alpha)
{
    if (lrtunemode == 0)
    {
        Uz->addgrad (dUz, alpha, l2reg);
        Wz->addgrad (dWz, alpha, l2reg);
        Ur->addgrad (dUr, alpha, l2reg);
        Wr->addgrad (dWr, alpha, l2reg);
        Uh->addgrad (dUh, alpha, l2reg);
        Wh->addgrad (dWh, alpha, l2reg);
    }
    else if (lrtunemode == 1)
    {
        accsdUz->addsquaregrad (dUz, 1.0, 1.0);
        Uz->addadagrad (dUz, accsdUz, alpha, l2reg);
        accsdWz->addsquaregrad (dWz, 1.0, 1.0);
        Wz->addadagrad (dWz, accsdWz, alpha, l2reg);
        accsdUr->addsquaregrad (dUr, 1.0, 1.0);
        Ur->addadagrad (dUr, accsdUr, alpha, l2reg);
        accsdWr->addsquaregrad (dWr, 1.0, 1.0);
        Wr->addadagrad (dWr, accsdWr, alpha, l2reg);
        accsdUh->addsquaregrad (dUh, 1.0, 1.0);
        Uh->addadagrad (dUh, accsdUh, alpha, l2reg);
        accsdWh->addsquaregrad (dWh, 1.0, 1.0);
        Wh->addadagrad (dWh, accsdWh, alpha, l2reg);
    }
    else if (lrtunemode == 2)
    {
        accsdUz->addsquaregrad (dUz, gamma, 1-gamma);
        Uz->addadagrad (dUz, accsdUz, alpha, l2reg);
        accsdWz->addsquaregrad (dWz, gamma, 1-gamma);
        Wz->addadagrad (dWz, accsdWz, alpha, l2reg);
        accsdUr->addsquaregrad (dUr, gamma, 1-gamma);
        Ur->addadagrad (dUr, accsdUr, alpha, l2reg);
        accsdWr->addsquaregrad (dWr, gamma, 1-gamma);
        Wr->addadagrad (dWr, accsdWr, alpha, l2reg);
        accsdUh->addsquaregrad (dUh, gamma, 1-gamma);
        Uh->addadagrad (dUh, accsdUh, alpha, l2reg);
        accsdWh->addsquaregrad (dWh, gamma, 1-gamma);
        Wh->addadagrad (dWh, accsdWh, alpha, l2reg);
    }

#ifdef MOMENTUM
    dUz->multiplyScalar (momentum);
    dWz->multiplyScalar (momentum);
    dUr->multiplyScalar (momentum);
    dWr->multiplyScalar (momentum);
    dUh->multiplyScalar (momentum);
    dWh->multiplyScalar (momentum);
#else
    dUz->initmatrix ();
    dWz->initmatrix ();
    dUr->initmatrix ();
    dWr->initmatrix ();
    dUh->initmatrix ();
    dWh->initmatrix ();
#endif
    // highway part
    if (lrtunemode == 0)
    {
        Uhw->addgrad (dUhw, alpha, l2reg);
        Whw->addgrad (dWhw, alpha, l2reg);
    }
    else if (lrtunemode == 1)
    {
        accsdUhw->addsquaregrad (dUhw, 1.0, 1.0);
        Uhw->addadagrad (dUhw, accsdUhw, alpha, l2reg);
        accsdWhw->addsquaregrad (dWhw, 1.0, 1.0);
        Whw->addadagrad (dWhw, accsdWhw, alpha, l2reg);
    }
    else if (lrtunemode == 2)
    {
        accsdUhw->addsquaregrad (dUhw, gamma, 1-gamma);
        Uhw->addadagrad (dUhw, accsdUhw, alpha, l2reg);
        accsdWhw->addsquaregrad (dWhw, gamma, 1-gamma);
        Whw->addadagrad (dWhw, accsdWhw, alpha, l2reg);
    }

#ifdef MOMENTUM
    dUhw->multiplyScalar (momentum);
    dWhw->multiplyScalar (momentum);
#else
    dUhw->initmatrix ();
    dWhw->initmatrix ();
#endif
}

void lstmlayer::update (float alpha)
{
    if (gradient_cutoff > 0)
    {
        dUz->L2norm (gradient_cutoff, cunorm2ptr, minibatch);
        dWz->L2norm (gradient_cutoff, cunorm2ptr, minibatch);
        dUi->L2norm (gradient_cutoff, cunorm2ptr, minibatch);
        dWi->L2norm (gradient_cutoff, cunorm2ptr, minibatch);
        dUf->L2norm (gradient_cutoff, cunorm2ptr, minibatch);
        dWf->L2norm (gradient_cutoff, cunorm2ptr, minibatch);
        dUo->L2norm (gradient_cutoff, cunorm2ptr, minibatch);
        dWo->L2norm (gradient_cutoff, cunorm2ptr, minibatch);
        dPi->L2norm (gradient_cutoff, cunorm2ptr, minibatch);
        dPf->L2norm (gradient_cutoff, cunorm2ptr, minibatch);
        dPo->L2norm (gradient_cutoff, cunorm2ptr, minibatch);
    }
    if (lrtunemode == 0)
    {
        Uz->addgrad (dUz, alpha, l2reg);
        Wz->addgrad (dWz, alpha, l2reg);
        Ui->addgrad (dUi, alpha, l2reg);
        Wi->addgrad (dWi, alpha, l2reg);
        Uf->addgrad (dUf, alpha, l2reg);
        Wf->addgrad (dWf, alpha, l2reg);
        Uo->addgrad (dUo, alpha, l2reg);
        Wo->addgrad (dWo, alpha, l2reg);
        Pi->addpeepholegrad (dPi, alpha, l2reg);
        Pf->addpeepholegrad (dPf, alpha, l2reg);
        Po->addpeepholegrad (dPo, alpha, l2reg);
    }
    else if (lrtunemode == 1)
    {
        accsdUz->addsquaregrad (dUz, 1.0, 1.0);
        Uz->addadagrad (dUz, accsdUz, alpha, l2reg);
        accsdWz->addsquaregrad (dWz, 1.0, 1.0);
        Wz->addadagrad (dWz, accsdWz, alpha, l2reg);
        accsdUi->addsquaregrad (dUi, 1.0, 1.0);
        Ui->addadagrad (dUi, accsdUi, alpha, l2reg);
        accsdWi->addsquaregrad (dWi, 1.0, 1.0);
        Wi->addadagrad (dWi, accsdWi, alpha, l2reg);
        accsdUf->addsquaregrad (dUf, 1.0, 1.0);
        Uf->addadagrad (dUf, accsdUf, alpha, l2reg);
        accsdWf->addsquaregrad (dWf, 1.0, 1.0);
        Wf->addadagrad (dWf, accsdWf, alpha, l2reg);
        accsdUo->addsquaregrad (dUo, 1.0, 1.0);
        Uo->addadagrad (dUo, accsdUo, alpha, l2reg);
        accsdWo->addsquaregrad (dWo, 1.0, 1.0);
        Wo->addadagrad (dWo, accsdWo, alpha, l2reg);
        // add all gradient up
        dPi_1col->addpeepholegrad (dPi, 1.0, 0.0);
        dPf_1col->addpeepholegrad (dPf, 1.0, 0.0);
        dPo_1col->addpeepholegrad (dPo, 1.0, 0.0);
        accsdPi->addsquaregrad (dPi_1col, 1.0, 1.0);
        Pi->addadagrad (dPi_1col, accsdPi, alpha, l2reg);
        accsdPf->addsquaregrad (dPf_1col, 1.0, 1.0);
        Pf->addadagrad (dPf_1col, accsdPf, alpha, l2reg);
        accsdPo->addsquaregrad (dPo_1col, 1.0, 1.0);
        Po->addadagrad (dPo_1col, accsdPo, alpha, l2reg);
    }
    else if (lrtunemode == 2)
    {
        accsdUz->addsquaregrad (dUz, gamma, 1-gamma);
        Uz->addadagrad (dUz, accsdUz, alpha, l2reg);
        accsdWz->addsquaregrad (dWz, gamma, 1-gamma);
        Wz->addadagrad (dWz, accsdWz, alpha, l2reg);
        accsdUi->addsquaregrad (dUi, gamma, 1-gamma);
        Ui->addadagrad (dUi, accsdUi, alpha, l2reg);
        accsdWi->addsquaregrad (dWi, gamma, 1-gamma);
        Wi->addadagrad (dWi, accsdWi, alpha, l2reg);
        accsdUf->addsquaregrad (dUf, gamma, 1-gamma);
        Uf->addadagrad (dUf, accsdUf, alpha, l2reg);
        accsdWf->addsquaregrad (dWf, gamma, 1-gamma);
        Wf->addadagrad (dWf, accsdWf, alpha, l2reg);
        accsdUo->addsquaregrad (dUo, gamma, 1-gamma);
        Uo->addadagrad (dUo, accsdUo, alpha, l2reg);
        accsdWo->addsquaregrad (dWo, gamma, 1-gamma);
        Wo->addadagrad (dWo, accsdWo, alpha, l2reg);
        // add all gradient up
        dPi_1col->addpeepholegrad (dPi, 1.0, 0.0);
        dPf_1col->addpeepholegrad (dPf, 1.0, 0.0);
        dPo_1col->addpeepholegrad (dPo, 1.0, 0.0);
        accsdPi->addsquaregrad (dPi_1col, gamma, 1-gamma);
        Pi->addadagrad (dPi_1col, accsdPi, alpha, l2reg);
        accsdPf->addsquaregrad (dPf_1col, gamma, 1-gamma);
        Pf->addadagrad (dPf_1col, accsdPf, alpha, l2reg);
        accsdPo->addsquaregrad (dPo_1col, gamma, 1-gamma);
        Po->addadagrad (dPo_1col, accsdPo, alpha, l2reg);
    }

#ifdef MOMENTUM
    dUz->multiplyScalar (momentum);
    dWz->multiplyScalar (momentum);
    dUi->multiplyScalar (momentum);
    dWi->multiplyScalar (momentum);
    dUf->multiplyScalar (momentum);
    dWf->multiplyScalar (momentum);
    dUo->multiplyScalar (momentum);
    dWo->multiplyScalar (momentum);
    dPi->multiplyScalar (momentum);
    dPf->multiplyScalar (momentum);
    dPo->multiplyScalar (momentum);
#else
    dUz->initmatrix ();
    dWz->initmatrix ();
    dUi->initmatrix ();
    dWi->initmatrix ();
    dUf->initmatrix ();
    dWf->initmatrix ();
    dUo->initmatrix ();
    dWo->initmatrix ();
    dPi->initmatrix ();
    dPf->initmatrix ();
    dPo->initmatrix ();
#endif
}

void lstmhighwaylayer::update (float alpha)
{
    // standard LSTM update
    if (lrtunemode == 0)
    {
        Uz->addgrad (dUz, alpha, l2reg);
        Wz->addgrad (dWz, alpha, l2reg);
        Ui->addgrad (dUi, alpha, l2reg);
        Wi->addgrad (dWi, alpha, l2reg);
        Uf->addgrad (dUf, alpha, l2reg);
        Wf->addgrad (dWf, alpha, l2reg);
        Uo->addgrad (dUo, alpha, l2reg);
        Wo->addgrad (dWo, alpha, l2reg);
        Pi->addpeepholegrad (dPi, alpha, l2reg);
        Pf->addpeepholegrad (dPf, alpha, l2reg);
        Po->addpeepholegrad (dPo, alpha, l2reg);
    }
    else if (lrtunemode == 1)
    {
        accsdUz->addsquaregrad (dUz, 1.0, 1.0);
        Uz->addadagrad (dUz, accsdUz, alpha, l2reg);
        accsdWz->addsquaregrad (dWz, 1.0, 1.0);
        Wz->addadagrad (dWz, accsdWz, alpha, l2reg);
        accsdUi->addsquaregrad (dUi, 1.0, 1.0);
        Ui->addadagrad (dUi, accsdUi, alpha, l2reg);
        accsdWi->addsquaregrad (dWi, 1.0, 1.0);
        Wi->addadagrad (dWi, accsdWi, alpha, l2reg);
        accsdUf->addsquaregrad (dUf, 1.0, 1.0);
        Uf->addadagrad (dUf, accsdUf, alpha, l2reg);
        accsdWf->addsquaregrad (dWf, 1.0, 1.0);
        Wf->addadagrad (dWf, accsdWf, alpha, l2reg);
        accsdUo->addsquaregrad (dUo, 1.0, 1.0);
        Uo->addadagrad (dUo, accsdUo, alpha, l2reg);
        accsdWo->addsquaregrad (dWo, 1.0, 1.0);
        Wo->addadagrad (dWo, accsdWo, alpha, l2reg);
        // add all gradient up
        dPi_1col->addpeepholegrad (dPi, 1.0, 0.0);
        dPf_1col->addpeepholegrad (dPf, 1.0, 0.0);
        dPo_1col->addpeepholegrad (dPo, 1.0, 0.0);
        accsdPi->addsquaregrad (dPi_1col, 1.0, 1.0);
        Pi->addadagrad (dPi_1col, accsdPi, alpha, l2reg);
        accsdPf->addsquaregrad (dPf_1col, 1.0, 1.0);
        Pf->addadagrad (dPf_1col, accsdPf, alpha, l2reg);
        accsdPo->addsquaregrad (dPo_1col, 1.0, 1.0);
        Po->addadagrad (dPo_1col, accsdPo, alpha, l2reg);
    }
    else if (lrtunemode == 2)
    {
        accsdUz->addsquaregrad (dUz, gamma, 1-gamma);
        Uz->addadagrad (dUz, accsdUz, alpha, l2reg);
        accsdWz->addsquaregrad (dWz, gamma, 1-gamma);
        Wz->addadagrad (dWz, accsdWz, alpha, l2reg);
        accsdUi->addsquaregrad (dUi, gamma, 1-gamma);
        Ui->addadagrad (dUi, accsdUi, alpha, l2reg);
        accsdWi->addsquaregrad (dWi, gamma, 1-gamma);
        Wi->addadagrad (dWi, accsdWi, alpha, l2reg);
        accsdUf->addsquaregrad (dUf, gamma, 1-gamma);
        Uf->addadagrad (dUf, accsdUf, alpha, l2reg);
        accsdWf->addsquaregrad (dWf, gamma, 1-gamma);
        Wf->addadagrad (dWf, accsdWf, alpha, l2reg);
        accsdUo->addsquaregrad (dUo, gamma, 1-gamma);
        Uo->addadagrad (dUo, accsdUo, alpha, l2reg);
        accsdWo->addsquaregrad (dWo, gamma, 1-gamma);
        Wo->addadagrad (dWo, accsdWo, alpha, l2reg);
        // add all gradient up
        dPi_1col->initmatrix ();
        dPf_1col->initmatrix ();
        dPo_1col->initmatrix ();
        dPi_1col->addpeepholegrad (dPi, 1.0, 0.0);
        dPf_1col->addpeepholegrad (dPf, 1.0, 0.0);
        dPo_1col->addpeepholegrad (dPo, 1.0, 0.0);
        accsdPi->addsquaregrad (dPi_1col, gamma, 1-gamma);
        Pi->addadagrad (dPi_1col, accsdPi, alpha, l2reg);
        accsdPf->addsquaregrad (dPf_1col, gamma, 1-gamma);
        Pf->addadagrad (dPf_1col, accsdPf, alpha, l2reg);
        accsdPo->addsquaregrad (dPo_1col, gamma, 1-gamma);
        Po->addadagrad (dPo_1col, accsdPo, alpha, l2reg);
    }

#ifdef MOMENTUM
    dUz->multiplyScalar (momentum);
    dWz->multiplyScalar (momentum);
    dUi->multiplyScalar (momentum);
    dWi->multiplyScalar (momentum);
    dUf->multiplyScalar (momentum);
    dWf->multiplyScalar (momentum);
    dUo->multiplyScalar (momentum);
    dWo->multiplyScalar (momentum);
    dPi->multiplyScalar (momentum);
    dPf->multiplyScalar (momentum);
    dPo->multiplyScalar (momentum);
#else
    dUz->initmatrix ();
    dWz->initmatrix ();
    dUi->initmatrix ();
    dWi->initmatrix ();
    dUf->initmatrix ();
    dWf->initmatrix ();
    dUo->initmatrix ();
    dWo->initmatrix ();
    dPi->initmatrix ();
    dPf->initmatrix ();
    dPo->initmatrix ();
#endif

    // highway part
    if (lrtunemode == 0)
    {
        Uhw->addgrad (dUhw, alpha, l2reg);
        Phw->addpeepholegrad (dPhw, alpha, l2reg);
        Rhw->addpeepholegrad (dRhw, alpha, l2reg);
    }
    else if (lrtunemode == 1)
    {
        accsdUhw->addsquaregrad (dUhw, 1.0, 1.0);
        Uhw->addadagrad (dUhw, accsdUhw, alpha, l2reg);
        dPhw_1col->initmatrix ();
        dRhw_1col->initmatrix ();
        dPhw_1col->addpeepholegrad (dPhw, 1.0, 0.0);
        dRhw_1col->addpeepholegrad (dRhw, 1.0, 0.0);
        accsdPhw->addsquaregrad (dPhw_1col, 1.0, 1.0);
        Phw->addadagrad (dPhw_1col, accsdPhw, alpha, l2reg);
        accsdRhw->addsquaregrad (dRhw_1col, 1.0, 1.0);
        Rhw->addadagrad (dRhw_1col, accsdRhw, alpha, l2reg);
    }
    else if (lrtunemode == 2)
    {
        accsdUhw->addsquaregrad (dUhw, gamma, 1-gamma);
        Uhw->addadagrad (dUhw, accsdUhw, alpha, l2reg);
        dPhw_1col->initmatrix ();
        dRhw_1col->initmatrix ();
        dPhw_1col->addpeepholegrad (dPhw, 1.0, 0.0);
        dRhw_1col->addpeepholegrad (dRhw, 1.0, 0.0);
        accsdPhw->addsquaregrad (dPhw_1col, gamma, 1-gamma);
        Phw->addadagrad (dPhw_1col, accsdPhw, alpha, l2reg);
        accsdRhw->addsquaregrad (dRhw_1col, gamma, 1-gamma);
        Rhw->addadagrad (dRhw_1col, accsdRhw, alpha, l2reg);
    }
#ifdef MOMENTUM
    dUhw->multiplyScalar (momentum);
    dPhw->multiplyScalar (momentum);
    dRhw->multiplyScalar (momentum);
#else
    dUhw->initmatrix ();
    dPhw->initmatrix ();
    dRhw->initmatrix ();
#endif
}


void feedforwardlayer::update (float alpha)
{
    if (lrtunemode == 0)
    {
        U->addgrad (dU, alpha, l2reg);
    }
    else if (lrtunemode == 1)
    {
        accsdU->addsquaregrad (dU, 1.0, 1.0);
        U->addadagrad (dU, accsdU, alpha, l2reg);
    }
    else if (lrtunemode == 2)
    {
        accsdU->addsquaregrad (dU, gamma, 1-gamma);
        U->addadagrad (dU, accsdU, alpha, l2reg);
    }

#ifdef MOMENTUM
    dU->multiplyScalar (momentum);
#else
    dU->initmatrix();
#endif
}

void linearlayer::update (float alpha)
{
    if (gradient_cutoff > 0)
    {
        dU->L2norm (gradient_cutoff, cunorm2ptr, minibatch);
    }
    if (lrtunemode == 0)
    {
        U->addgrad (dU, alpha, l2reg);
    }
    else if (lrtunemode == 1)
    {
        accsdU->addsquaregrad (dU, 1.0, 1.0);
        U->addadagrad (dU, accsdU, alpha, l2reg);
    }
    else if (lrtunemode == 2)
    {
        accsdU->addsquaregrad (dU, gamma, 1-gamma);
        U->addadagrad (dU, accsdU, alpha, l2reg);
    }
#ifdef MOMENTUM
    dU->multiplyScalar (momentum);
#else
    dU->initmatrix();
#endif
}

inputlayer::~inputlayer()
{
    DeleteMat (U_fea);
    DeleteMat (dU_fea);
    DeleteMat (feamatrix);
    DeleteMatVec (ac_fea_vec);
}

void inputlayer::allocFeaMem ()
{
    if (mbfeaindices != NULL)  // already allocate memory when reading model
    {
        return;
    }
    mbfeaindices = (int *)malloc (sizeof(int)*minibatch);
    U_fea = new matrix (dim_fea, ncols);
    U_fea->random(MINRANDINITVALUE, MAXRANDINITVALUE);
    dU_fea = new matrix (dim_fea, ncols);
    dU_fea->initmatrix ();
    ac_fea_vec.resize (chunksize);
    for (int i=0; i<chunksize; i++)
    {
        ac_fea_vec[i] = new matrix (dim_fea, minibatch);
        ac_fea_vec[i]->initmatrix ();
    }
    ac_fea = ac_fea_vec[0];
}

inputlayer::inputlayer (int nr, int nc, int mbsize, int cksize, int dim): layer(nr, nc, mbsize, cksize)
{
    type = "input";
    dim_fea = dim;
    feamatrix = NULL;
    mbfeaindices = NULL;
    feaindices = NULL;
    num_fea = 0;
    if (dim_fea == 0)
    {
        U_fea = NULL;
        dU_fea = NULL;
        ac_fea = NULL;
        ac_fea_vec.clear();
    }
    else
    {
        allocFeaMem ();
    }
    dropoutHiddenmaskMat_vec.resize (chunksize);
    for (int i=0; i<chunksize; i++)
    {
        dropoutHiddenmaskMat_vec[i] = new matrix (ncols, minibatch);
        dropoutHiddenmaskMat_vec[i]->assignmatvalue (1.0);
    }
}

outputlayer::~outputlayer ()
{
    int i;
    DeleteMatVec (lognormvec);
    // NCE training
    if (traincrit == 2)
    {
        DeleteMat (gradlayerN_NCE);
        DeleteMat (gradlayerN_NCE_succ);
        DeleteMat (neuN_er_NCE);
        DeleteMat (W_NCE);
        DeleteMat (dW_NCE);
        DeleteMat (logwordnoise);
        DeleteMatVec (lognormvec);
        DeleteMatVec (neuN_ac_NCE_vec);
        DeleteMatVec (layerN_NCE_vec);
        DeleteMatVec (layerN_succ_NCE_vec);
        DeleteMatVec (targetsample_vec);
        DeleteMatVec (ncesample_vec);
        for (int i=0; i<chunksize; i++)
        {
            free(mbid2arrid[i]);
            cufree(dev_mbid2arrid[i]);
            free(ncesamplecnt[i]);
            cufree(dev_ncesamplecnt[i]);
            free(targetsample[i]);
            free(ncesample[i]);
            cufree(dev_targetsample[i]);
            cufree(dev_ncesample[i]);
        }
        free(ncesamplecnt);
        free(dev_ncesamplecnt);
        free(mbid2arrid);
        free(dev_mbid2arrid);
        free(ncesample);
        free(dev_ncesample);
        free(targetsample);
        free(dev_targetsample);
        free(ntargetsample);
        free(nncesample);
        free(outputlayersize_NCE);
    }
}

outputlayer::outputlayer (int nr, int nc, int mbsize, int cksize): layer (nr, nc, mbsize, cksize)
{
    type = "output";
    vrpenalty = 0.0;
    traincrit = 0;          // CE training by default
    lognormvec.resize(chunksize);
    for (int i=0; i<chunksize; i++)
    {
        lognormvec[i] = new matrix (1, minibatch);
        lognormvec[i]->initmatrix ();
    }
}

void outputlayer::allocNCEMem (int num_noisesample)
{
    type = "output";
    traincrit = 2;          // NCE training
    chunkiter = 0;
    ntargetsample = 0;
    nncesample = 0;
    k = num_noisesample;
    N = k+minibatch;
    outOOSindex = ncols - 1;   // last word maps to the OOS node
    evalmode = false;
    neuN_ac_NCE_vec.resize(chunksize);
    layerN_NCE_vec.resize (chunksize);
    layerN_succ_NCE_vec.resize (chunksize);
    for (int i=0; i<chunksize; i++)
    {
        neuN_ac_NCE_vec[i] = new matrix (N, minibatch);
        layerN_NCE_vec[i] = new matrix (nrows, N);
        layerN_succ_NCE_vec[i] = new matrix (nrows, N);
    }
    gradlayerN_NCE = new matrix (nrows, N);
    gradlayerN_NCE_succ = new matrix (nrows, N);
    neuN_er_NCE = new matrix (N, minibatch);
    W_NCE  = new matrix (nrows, N);
    dW_NCE = new matrix (nrows, N);

    gradlayerN_NCE->initmatrix ();
    gradlayerN_NCE_succ->initmatrix ();
    neuN_er_NCE->initmatrix ();
    W_NCE->initmatrix ();
    dW_NCE->initmatrix ();

    // create thread-specific RNGs
    rngs = new rng_type[k];
    for (int t = 0; t < k; t++)
    {
        // rngs[t] = rng_type(rand_seed + 15791 * t);
        rngs[t] = rng_type(1 + 15791 * t);
    }

    accprobvec.resize (ncols);
    unigram.resize (ncols);
    logunigram.resize (ncols);
    logwordnoise = new matrix (1, ncols);
    ncesamplecnt = (int **) calloc (chunksize, sizeof(int *));
    dev_ncesamplecnt = (int **) calloc (chunksize, sizeof(int *));
    mbid2arrid      = (int **) calloc (chunksize, sizeof (int *));
    dev_mbid2arrid  = (int **) calloc (chunksize, sizeof (int *));
    ncesample = (int **) calloc (chunksize, sizeof(int *));
    dev_ncesample = (int **) calloc (chunksize, sizeof(int *));
    targetsample = (int **) calloc (chunksize, sizeof(int *));
    dev_targetsample = (int **) calloc (chunksize, sizeof(int *));
    for (int i=0; i<chunksize; i++)
    {
        ncesamplecnt[i] = (int *) calloc (k, sizeof (int));
        dev_ncesamplecnt[i] = (int *)cucalloc (k*sizeof(int));
        mbid2arrid[i] = (int *)calloc (minibatch, sizeof(int) );
        dev_mbid2arrid[i] = (int *)cucalloc (minibatch*sizeof(int));
        ncesample[i] = (int *)calloc (k, sizeof(int));
        dev_ncesample[i] = (int *)cucalloc (k * sizeof(int));
        targetsample[i] = (int *)calloc (minibatch, sizeof(int));
        dev_targetsample[i] = (int *)cucalloc (minibatch * sizeof(int));
    }

    ntargetsample = (int *) calloc (chunksize, sizeof(int));
    nncesample = (int *) calloc (chunksize, sizeof(int));
    outputlayersize_NCE = (int *) calloc (chunksize, sizeof(int));
    memset (ntargetsample, 0, sizeof(int)*chunksize);
    memset (nncesample, 0, sizeof(int)*chunksize);
    memset (outputlayersize_NCE, 0, sizeof(int)*chunksize);
}

recurrentlayer::~recurrentlayer ()
{
    int i;
    DeleteMat (accsdW);
    DeleteMat (W);
    DeleteMat (dW);
    DeleteMat (hidden_er);
    DeleteMat (hidden_ac);
    DeleteMat (hidden_ac_last);
    DeleteMatVec (hidden_ac_vec);
}
recurrentlayer::recurrentlayer (int nr, int nc, int mbsize, int cksize) : layer (nr, nc, mbsize, cksize)
{
    type = "recurrent";
    nodetype = 0;
    W = new matrix (ncols, ncols);
    dW = new matrix (ncols, ncols);
    hidden_er = new matrix (ncols, minibatch);
    hidden_ac = NULL;
    hidden_ac_last = new matrix (ncols, minibatch);
    hidden_ac_vec.resize(chunksize+1);
    for (int i=0; i<chunksize+1; i++)
    {
        hidden_ac_vec[i] = new matrix (ncols, minibatch);

        hidden_ac_vec[i]->initmatrix();
    }
    hidden_ac_last->assignmatvalue (RESETVALUE);
    W->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    dW->initmatrix();
    hidden_er->initmatrix();
    hidden_ac = hidden_ac_vec[0];
}

lstmlayer::~lstmlayer ()
{
    DeleteMat (newc);
    DeleteMatVec (c_vec);
    DeleteMatVec (dropoutHiddenmaskMat_vec);
    DeleteMat (hidden_er);
    DeleteMat (hidden_ac);
    DeleteMat (c_er);
    DeleteMat (hidden_ac_last);
    DeleteMat (c_last);
    DeleteMat (Uz);
    DeleteMat (Ui);
    DeleteMat (Uf);
    DeleteMat (Uo);
    DeleteMat (Wz);
    DeleteMat (Wi);
    DeleteMat (Wf);
    DeleteMat (Wo);
    DeleteMat (dUz);
    DeleteMat (dUi);
    DeleteMat (dUf);
    DeleteMat (dUo);
    DeleteMat (dWz);
    DeleteMat (dWi);
    DeleteMat (dWf);
    DeleteMat (dWo);
    DeleteMat (accsdUz);
    DeleteMat (accsdUi);
    DeleteMat (accsdUf);
    DeleteMat (accsdUo);
    DeleteMat (accsdWz);
    DeleteMat (accsdWi);
    DeleteMat (accsdWf);
    DeleteMat (accsdWo);
    DeleteMat (dPi_1col);
    DeleteMat (dPf_1col);
    DeleteMat (dPo_1col);
    DeleteMat (Pi);
    DeleteMat (Pf);
    DeleteMat (Po);
    DeleteMat (dPi);
    DeleteMat (dPf);
    DeleteMat (dPo);
    DeleteMat (z);
    DeleteMat (i);
    DeleteMat (f);
    DeleteMat (c);
    DeleteMat (zi);
    DeleteMat (fc);
    DeleteMat (o);
    DeleteMatVec (z_vec);
    DeleteMatVec (i_vec);
    DeleteMatVec (f_vec);
    DeleteMatVec (newc_vec);
    DeleteMatVec (o_vec);
    DeleteMatVec (hidden_ac_vec);
    DeleteMat (dz);
    DeleteMat (di);
    DeleteMat (df);
    DeleteMat (dnewc);
    DeleteMat (do_);
}
lstmlayer::lstmlayer (int nr, int nc, int mbsize, int cksize) : layer (nr, nc, mbsize, cksize)
{
    type = "lstm";

    delete U;
    delete dU;
    U = NULL;
    dU = NULL;
    Uz = new matrix (nrows, ncols);
    dUz = new matrix (nrows, ncols);
    Ui = new matrix (nrows, ncols);
    dUi = new matrix (nrows, ncols);
    Uf = new matrix (nrows, ncols);
    dUf = new matrix (nrows, ncols);
    Uo = new matrix (nrows, ncols);
    dUo = new matrix (nrows, ncols);

    Uz->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Ui->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Uf->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Uo->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    dUo->initmatrix ();
    dUz->initmatrix ();
    dUi->initmatrix ();
    dUf->initmatrix ();

    Wz = new matrix (ncols, ncols);
    dWz = new matrix (ncols, ncols);
    Wi = new matrix (ncols, ncols);
    dWi = new matrix (ncols, ncols);
    Wf = new matrix (ncols, ncols);
    dWf = new matrix (ncols, ncols);
    Wo = new matrix (ncols, ncols);
    dWo = new matrix (ncols, ncols);

    Wz->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Wi->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Wf->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Wo->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    dWz->initmatrix ();
    dWi->initmatrix ();
    dWf->initmatrix ();
    dWo->initmatrix ();

    Pi = new matrix (ncols, 1);
    Pf = new matrix (ncols, 1);
    Po = new matrix (ncols, 1);
    dPi = new matrix (ncols, minibatch);
    dPf = new matrix (ncols, minibatch);
    dPo = new matrix (ncols, minibatch);

    Pi->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Pf->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Po->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    dPi->initmatrix ();
    dPf->initmatrix ();
    dPo->initmatrix ();

    dz = new matrix (ncols, minibatch);
    di = new matrix (ncols, minibatch);
    df = new matrix (ncols, minibatch);
    newc = new matrix (ncols, minibatch);
    dnewc = new matrix (ncols, minibatch);
    do_ = new matrix (ncols, minibatch);
    zi = new matrix (ncols, minibatch);
    fc = new matrix (ncols, minibatch);

    dz->initmatrix ();
    di->initmatrix ();
    df->initmatrix ();
    do_->initmatrix ();
    zi->initmatrix ();
    fc->initmatrix ();

    z_vec.resize (chunksize);
    i_vec.resize (chunksize);
    f_vec.resize (chunksize);
    c_vec.resize (chunksize+1);
    newc_vec.resize (chunksize);
    o_vec.resize (chunksize);
    hidden_ac_vec.resize(chunksize+1);
    // not use in the code, consider to investigate where to add dropout
    for (int i=0; i<chunksize; i++)
    {
        z_vec[i] = new matrix (ncols, minibatch);
        i_vec[i] = new matrix (ncols, minibatch);
        f_vec[i] = new matrix (ncols, minibatch);
        c_vec[i] = new matrix (ncols, minibatch);
        newc_vec[i] = new matrix (ncols, minibatch);
        o_vec[i] = new matrix (ncols, minibatch);
        hidden_ac_vec[i] = new matrix (ncols, minibatch);

        z_vec[i]->initmatrix ();
        i_vec[i]->initmatrix ();
        f_vec[i]->initmatrix ();
        c_vec[i]->initmatrix ();
        o_vec[i]->initmatrix ();
        hidden_ac_vec[i]->initmatrix ();
    }
    hidden_ac_vec[chunksize] = new matrix (ncols, minibatch);
    hidden_ac_vec[chunksize]->initmatrix ();
    c_vec[chunksize] = new matrix (ncols, minibatch);
    c_vec[chunksize]->initmatrix();

    hidden_ac = NULL;
    hidden_ac_last = new matrix (ncols, minibatch);
    c_last = new matrix (ncols, minibatch);
    hidden_er = new matrix (ncols, minibatch);
    c_er = new matrix (ncols, minibatch);
    hidden_ac_last->assignmatvalue (RESETVALUE);
    c_last->assignmatvalue (RESETVALUE);
    hidden_er->initmatrix();
    c_er->initmatrix ();

    // when ppl and nbest, they need to use hidden_ac, c, etc.
    hidden_ac = hidden_ac_vec[0];
    c = c_vec[0];
    z = z_vec[0];
    i = i_vec[0];
    f = f_vec[0];
    o = f_vec[0];

#ifdef VARIATIONALDROPOUT
    dropoutHiddenmaskMat_vec.resize (chunksize);
    for (int i=0; i<chunksize; i++)
    {
        dropoutHiddenmaskMat_vec[i] = new matrix (ncols, minibatch);
        dropoutHiddenmaskMat_vec[i]->assignmatvalue (1.0);
    }
#endif
}

lstmhighwaylayer::~lstmhighwaylayer()
{
    DeleteMat (c_hw);
    DeleteMat (c_er_hw);
    DeleteMat (s);
    DeleteMat (sc);
    DeleteMat (ds);
    DeleteMat (dsc);
    DeleteMat (Uhw);
    DeleteMat (Phw);
    DeleteMat (Rhw);
    DeleteMat (dUhw);
    DeleteMat (dPhw);
    DeleteMat (dRhw);
    DeleteMat (accsdUhw);
    DeleteMat (accsdPhw);
    DeleteMat (accsdRhw);
    DeleteMat (dPhw_1col);
    DeleteMat (dRhw_1col);
    DeleteMatVec (c_hw_vec);
    DeleteMatVec (s_vec);
}
lstmhighwaylayer::lstmhighwaylayer (int nr, int nc, int mbsize, int cksize) : lstmlayer (nr, nc, mbsize, cksize)
{
    type = "lstm-highway";
    assert (nrows == ncols);        // for highway connection, the nrows=ncols
    Uhw = new matrix (nrows, ncols);
    dUhw = new matrix (nrows, ncols);
    Phw = new matrix (ncols, 1);
    dPhw = new matrix (ncols, minibatch);
    Rhw = new matrix (nrows, 1);
    dRhw = new matrix (nrows, minibatch);
    c_er_hw = new matrix (ncols, minibatch);
    sc   = new matrix (nrows, minibatch);
    ds   = new matrix (nrows, minibatch);
    dsc  = new matrix (nrows, minibatch);
    c_hw_vec.resize (chunksize);
    s_vec.resize (chunksize);
    for (int i=0; i<chunksize; i++)
    {
        c_hw_vec[i] = new matrix (ncols, minibatch);
        s_vec[i] = new matrix (ncols, minibatch);
        c_hw_vec[i]->initmatrix ();
        s_vec[i]->initmatrix ();
    }

    Uhw->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Phw->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Rhw->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    dUhw->initmatrix ();
    dPhw->initmatrix ();
    dRhw->initmatrix ();
    sc->initmatrix ();
    ds->initmatrix ();
    dsc->initmatrix ();
    c_er_hw->initmatrix ();

    // when ppl and nbest, they need to use hidden_ac, c, etc.
    s = s_vec[0];
    c_hw = new matrix (ncols, minibatch);
    c_hw->initmatrix();
}

grulayer::~grulayer ()
{
    DeleteMat (hidden_er);
    DeleteMat (hidden_ac);
    DeleteMat (hidden_ac_last);
    DeleteMat (Wr);
    DeleteMat (Wz);
    DeleteMat (Wh);
    DeleteMat (Ur);
    DeleteMat (Uz);
    DeleteMat (Uh);
    DeleteMat (dWr);
    DeleteMat (dWz);
    DeleteMat (dWh);
    DeleteMat (dUr);
    DeleteMat (dUz);
    DeleteMat (dUh);
    DeleteMat (accsdWr);
    DeleteMat (accsdWz);
    DeleteMat (accsdWh);
    DeleteMat (accsdUr);
    DeleteMat (accsdUz);
    DeleteMat (accsdUh);
    DeleteMat (r);
    DeleteMat (z);
    DeleteMat (c);
    DeleteMat (h_);
    DeleteMat (dr);
    DeleteMat (dz);
    DeleteMat (dc);
    DeleteMat (dh_);
    DeleteMatVec (r_vec);
    DeleteMatVec (z_vec);
    DeleteMatVec (c_vec);
    DeleteMatVec (h_vec);
    DeleteMatVec (hidden_ac_vec);
}

grulayer::grulayer (int nr, int nc, int mbsize, int cksize) : layer (nr, nc, mbsize, cksize)
{
    type = "gru";

    delete U;
    delete dU;
    U = NULL;
    dU = NULL;
    Uh = new matrix (nrows, ncols);
    dUh = new matrix (nrows, ncols);
    Ur  = new matrix (nrows, ncols);
    dUr = new matrix (nrows, ncols);
    Uz = new matrix (nrows, ncols);
    dUz = new matrix (nrows, ncols);

    Wh = new matrix (ncols, ncols);
    dWh = new matrix (ncols, ncols);
    Wr  = new matrix (ncols, ncols);
    dWr = new matrix (ncols, ncols);
    Wz = new matrix (ncols, ncols);
    dWz = new matrix (ncols, ncols);

    r_vec.resize(chunksize);
    z_vec.resize(chunksize);
    c_vec.resize(chunksize);
    h_vec.resize(chunksize);
    hidden_ac_vec.resize(chunksize+1);
    for (int i=0; i<chunksize; i++)
    {
        r_vec[i] = new matrix (ncols, minibatch);
        z_vec[i] = new matrix (ncols, minibatch);
        c_vec[i] = new matrix (ncols, minibatch);
        h_vec[i]= new matrix (ncols, minibatch);
        hidden_ac_vec[i] = new matrix (ncols, minibatch);

        z_vec[i]->initmatrix();
        r_vec[i]->initmatrix();
        c_vec[i]->initmatrix();
        h_vec[i]->initmatrix();
        hidden_ac_vec[i]->initmatrix();
    }
    hidden_ac_vec[chunksize] = new matrix (ncols, minibatch);
    hidden_ac_vec[chunksize]->initmatrix();

    dz = new matrix (ncols, minibatch);
    dr = new matrix (ncols, minibatch);
    dc = new matrix (ncols, minibatch);
    dh_ = new matrix (ncols, minibatch);

    hidden_ac = NULL;
    hidden_ac_last = new matrix (ncols, minibatch);
    hidden_er = new matrix (ncols, minibatch);

    Wh->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Wr->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Wz->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Uh->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Ur->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Uz->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    dUz->initmatrix ();
    dUh->initmatrix ();
    dUr->initmatrix ();
    dWz->initmatrix ();
    dWh->initmatrix ();
    dWr->initmatrix ();

    hidden_ac_last->assignmatvalue (RESETVALUE);
    hidden_er->initmatrix();
    dz->initmatrix ();
    dr->initmatrix ();
    dh_->initmatrix();

    // when ppl and nbest, they need to use hidden_ac, c, etc.
    hidden_ac = hidden_ac_vec[0];
    r = r_vec[0];
    z = z_vec[0];
    h_= h_vec[0];
}

gruhighwaylayer::~gruhighwaylayer()
{
    DeleteMat (g);
    DeleteMat (v);
    DeleteMat (dg);
    DeleteMat (dv);
    DeleteMat (hidden_er_highway);
    DeleteMat (Uhw);
    DeleteMat (Whw);
    DeleteMat (dUhw);
    DeleteMat (dWhw);
    DeleteMat (accsdUhw);
    DeleteMat (accsdWhw);
    DeleteMatVec (g_vec);
    DeleteMatVec (v_vec);
}
gruhighwaylayer::gruhighwaylayer (int nr, int nc, int mbsize, int cksize) : grulayer (nr, nc, mbsize, cksize)
{
    type = "gru-highway";
    assert (nrows == ncols);        // for highway connection, the nrows=ncols
    Uhw  = new matrix (nrows, ncols);
    dUhw = new matrix (nrows, ncols);

    Whw  = new matrix (ncols, ncols);
    dWhw = new matrix (ncols, ncols);

    dg   = new matrix (ncols, minibatch);
    dv   = new matrix (ncols, minibatch);
    hidden_er_highway = new matrix (ncols, minibatch);

    v_vec.resize (chunksize);
    g_vec.resize (chunksize);
    for (int i=0; i<chunksize; i++)
    {
        v_vec[i] = new matrix (ncols, minibatch);
        g_vec[i] = new matrix (ncols, minibatch);
        v_vec[i]->initmatrix ();
        g_vec[i]->initmatrix ();
    }

    Uhw->random (MINRANDINITVALUE, MAXRANDINITVALUE);
    Whw->random (MINRANDINITVALUE, MAXRANDINITVALUE);

    dUhw->initmatrix ();
    dWhw->initmatrix ();
    dg->initmatrix ();
    dv->initmatrix ();
    hidden_er_highway->initmatrix ();

    // when ppl and nbest, they need to use hidden_ac, c, etc.
    g = g_vec[0];
    v = v_vec[0];
}

feedforwardlayer::feedforwardlayer (int nr, int nc, int mbsize, int cksize) : layer (nr, nc, mbsize, cksize)
{
    type = "feedforward";
    nodetype = 0;
}

linearlayer::linearlayer (int nr, int nc, int mbsize, int cksize) : layer (nr, nc, mbsize, cksize) {type = "linear";}


void inputlayer::ReadFeaFile (string filestr)
{
    feafile = filestr;
    int i, j, t;
    int err;
    float value;
    FILE *fptr = fopen (feafile.c_str(), "r");
    if (fptr == NULL)
    {
        printf ("Error: Failed to open feature file: %s\n", feafile.c_str());
        exit(0);
    }
    err = fscanf (fptr, "%d %d", &num_fea, &dim_fea);
    // if the fea file is two large, just allocate cpu memory
    feamatrix = new matrix (dim_fea, num_fea);
    feamatrix->initmatrix();
    printf ("%d lines feature (with %d dimensions) will be read from %s\n", num_fea, dim_fea, feafile.c_str());
    i = 0;
    while (i < num_fea)
    {
        if (feof(fptr))         break;
        err = fscanf (fptr, "%d", &j);
        assert (j == i);
        for (t=0; t<dim_fea; t++)
        {
            err = fscanf (fptr, "%f", &value);
            feamatrix->assignhostvalue(t, i, value);
        }
        i ++;
    }
    if (i != num_fea)
    {
        printf ("Warning: only read %d lines from the feature file: %s, should be %d lines\n", i, feafile.c_str(), num_fea);
    }
    feamatrix->assign();
    printf ("%d feature lines (with %d dimensions) is read from %s successfully\n", num_fea, dim_fea,  feafile.c_str());
    fclose(fptr);

    // allocate memeory for additional feature in input layer
    // if memory already allocated during laoding model
    allocFeaMem ();
}

void inputlayer::setFeaIndices (int *ptr)
{
    feaindices = ptr;
    for (int i=0; i<minibatch; i++)
    {
        mbfeaindices[i] = i;
    }
}

bool inputlayer::usefeainput ()
{
    if (dim_fea > 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void inputlayer::fillInputFeature ()
{
    int i, t, index;
    for (i=0; i<minibatch; i++)
    {
        t = mbfeaindices[i];
        index = feaindices[t];
        cucpyInGpu (ac_fea->getdevdataptr(0, i), feamatrix->getdevdataptr(0, index), sizeof(float)*dim_fea);
    }
}

void inputlayer::host_assignFeaVec (int feaid)
{
    float *feaptr = feamatrix->gethostdataptr (0, feaid);
    float *acptr  = ac_fea->gethostdataptr ();
    memcpy (acptr, feaptr, sizeof(float)*dim_fea);
}

void inputlayer::assignFeaMat ()
{
    ac_fea = ac_fea_vec[0];
    fillInputFeature ();
    for (int i=1; i<chunksize; i++)
    {
        ac_fea_vec[i]->assign (ac_fea_vec[0]);
    }
}

void inputlayer::updateFeaMat (int mbidx)
{
    mbfeaindices[mbidx] += minibatch;
    int index = feaindices[mbfeaindices[mbidx]];
    ac_fea = ac_fea_vec[chunkiter];
    cucpyInGpu (ac_fea->getdevdataptr(0, mbidx), feamatrix->getdevdataptr(0, index), sizeof(float)*dim_fea);

}

void inputlayer::getWordEmbedding (int *dev_prevwords, matrix *neu_ac)
{
    int minibatch = neu_ac->cols();
    neu_ac->initmatrix();
    neu1addneu0words (dev_prevwords, neu_ac->getdevdataptr(), U->getdevdataptr(), nrows, ncols, minibatch);
#ifdef DROPOUTINWORD
    if (dropoutrate > 0)
    {
        // try a smaller dropout in the input layer
        dropoutHiddenmaskMat_vec[chunkiter]->genEmbeddropoutmask (dropoutrate);
        neu_ac->dropout (dropoutHiddenmaskMat_vec[chunkiter], dropoutrate, evalmode);
    }
#endif
    if (dim_fea > 0)
    {
        ac_fea = ac_fea_vec[chunkiter];
        cumatrixXmatrix (U_fea, ac_fea, neu_ac, true, false, 1.0, 1.0);
    }
}

void inputlayer::host_getWordEmbedding (int prevword, matrix *neu_ac)
{
    float *wgts = U->gethostdataptr ();
    for (int i=0; i<ncols; i++)
    {
        neu_ac->assignhostvalue(i, 0, wgts[prevword+nrows*i]);
    }
    if (dim_fea > 0)
    {
        float *srcac = ac_fea->gethostdataptr();
        float *wgts  = U_fea->gethostdataptr();
        float *dstac = neu_ac->gethostdataptr();
        matrixXvector (srcac, wgts, dstac, dim_fea, ncols);
    }
}

void inputlayer::updateWordEmbedding (int *dev_prevwords, matrix *neu_er)
{
#ifdef DROPOUTINWORD
    if (dropoutrate > 0)
    {
        neu_er->dotMultiply (dropoutHiddenmaskMat_vec[chunkiter]);
    }
#endif
#ifdef SIMPLEUPDATEINPUTLYAER
    U->updatelayer0_word (neu_er, dev_prevwords, alpha, 0.0);
#else
    dU->updatelayer0_word (neu_er, dev_prevwords, 1.0, 0.0);
#endif
    if (dim_fea > 0)
    {
        ac_fea = ac_fea_vec[chunkiter];
        cumatrixXmatrix (ac_fea, neu_er, dU_fea, false, true, 1.0, 1.0);
    }
}

void inputlayer::update (float alpha)
{
    if (dim_fea > 0)
    {
        if (gradient_cutoff > 0)
        {
            dU_fea->L2norm (gradient_cutoff, cunorm2ptr, minibatch);
        }
        U_fea->addgrad (dU_fea, alpha, l2reg);
#ifdef MOMENTUM
        dU_fea->multiplyScalar (momentum);
#else
        dU_fea->initmatrix();
#endif
    }
#ifdef SIMPLEUPDATEINPUTLYAER
    return;
#else
    if (gradient_cutoff > 0)
    {
        dU->L2norm (gradient_cutoff, cunorm2ptr, minibatch);
    }
    if (lrtunemode == 0)
    {
        U->addgrad (dU, alpha, l2reg);
    }
    else if (lrtunemode == 1)
    {
        accsdU->addsquaregrad (dU, 1.0, 1.0);
        U->addadagrad (dU, accsdU, alpha, l2reg);
    }
    else if (lrtunemode == 2)
    {
        accsdU->addsquaregrad (dU, gamma, 1-gamma);
        U->addadagrad (dU, accsdU, alpha, l2reg);
    }
#ifdef MOMENTUM
    dU->multiplyScalar (momentum);
#else
    dU->initmatrix();
#endif
#endif
}


void recurrentlayer::resetHiddenAc (int mbidx)
{
    hidden_ac_vec[chunkiter]->assigndevcolumnvalue (mbidx, RESETVALUE);
}
void grulayer::resetHiddenAc (int mbidx)
{
    hidden_ac_vec[chunkiter]->assigndevcolumnvalue (mbidx, RESETVALUE);
}
void lstmlayer::resetHiddenAc (int mbidx)
{
    hidden_ac_vec[chunkiter]->assigndevcolumnvalue (mbidx, RESETVALUE);
    c_vec[chunkiter]->assigndevcolumnvalue (mbidx, RESETVALUE);
}

void recurrentlayer::resetErrVec (int mbidx)
{
    hidden_er->assigndevcolumnvalue (mbidx, 0);
}
void grulayer::resetErrVec (int mbidx)
{
    hidden_er->assigndevcolumnvalue (mbidx, 0);
}
void lstmlayer::resetErrVec (int mbidx)
{
    hidden_er->assigndevcolumnvalue (mbidx, 0);
    c_er->assigndevcolumnvalue (mbidx, 0);
}

void recurrentlayer::initHiddenEr ()
{
    hidden_er->initmatrix ();
}
void grulayer::initHiddenEr ()
{
    hidden_er->initmatrix ();
}
void lstmlayer::initHiddenEr ()
{
    c_er->initmatrix ();
    hidden_er->initmatrix ();
}

void recurrentlayer::saveHiddenAc ()
{
    hidden_ac_last->assign (hidden_ac_vec[chunksize]);
}
void grulayer::saveHiddenAc ()
{
    hidden_ac_last->assign (hidden_ac_vec[chunksize]);
}
void lstmlayer::saveHiddenAc ()
{
    hidden_ac_last->assign (hidden_ac_vec[chunksize]);
    c_last->assign (c_vec[chunksize]);
}

void recurrentlayer::loadHiddenAc ()
{
    hidden_ac_vec[0]->assign(hidden_ac_last);
}
void grulayer::loadHiddenAc ()
{
    hidden_ac_vec[0]->assign(hidden_ac_last);
}
void lstmlayer::loadHiddenAc ()
{
    hidden_ac_vec[0]->assign(hidden_ac_last);
    c_vec[0]->assign (c_last);
}

void recurrentlayer::initHiddenAc ()
{
    hidden_ac_last->assignmatvalue (RESETVALUE);
}
void grulayer::initHiddenAc ()
{
    hidden_ac_last->assignmatvalue (RESETVALUE);
}
void lstmlayer::initHiddenAc ()
{
    hidden_ac_last->assignmatvalue (RESETVALUE);
    c_last->assignmatvalue (RESETVALUE);
}

void outputlayer::ComputeAccMeanVar (int *host_curwords, double &lognorm_mean, double &lognorm_var)
{
    for (int chunkiter=0; chunkiter<chunksize; chunkiter++)
    {
        lognormvec[chunkiter]->fetch ();
        for (int mbiter=0; mbiter<minibatch; mbiter++)
        {
            if (host_curwords[mbiter+chunkiter*minibatch] != INVALID_INT)
            {
                float v = lognormvec[chunkiter]->fetchhostvalue (0, mbiter);
                if (traincrit == 1)     // for VR training
                {
                    v += lognorm;
                }
                lognorm_mean += v;
                lognorm_var += v*v;
            }
        }
    }
}

void outputlayer::prepareNCEtrain (double *wordprob, vector<string> &outputvec)
{
    float log_num_noise = log (k);
    for (int i=0; i<ncols; i++)
    {
        unigram[i] = wordprob[i];
        if (i == 0)
        {
            accprobvec[i] = wordprob[i];
        }
        else
        {
            accprobvec[i] = accprobvec[i-1] + wordprob[i];
        }
    }
    for (int i=0; i<ncols; i++)
    {
        logunigram[i] = log(unigram[i]);
        // unigram[i] = wordprob[i]*exp(lognorm);
        float v = log_num_noise + logunigram[i] + lognorm;
        logwordnoise->assignhostvalue (0, i, v);
    }
    logwordnoise->assign();
    double prob = accprobvec[ncols - 1];
    uniform_real_dist = uniform_real_distribution<double>(0.f, prob - prob / ncols);

    // check whether there is zero unigram probability
    for (int i=0; i<ncols; i++)
    {
        if (unigram[i] == 0)
        {
            printf ("the unigram probability for word %s is zero, check!\n", outputvec[i].c_str());
            exit (0);
        }
    }
}

void outputlayer::assignTargetSample (int *words)
{
    // memcpy (targetsample[chunkiter], host_curwords, sizeof(int)*minibatch);
    host_curwords = words;
}

void outputlayer::genNCESample ()
{
    unsigned int token;
    for (int i=0; i<k; i++)
    {
        if (i==0)
        {
            token = outOOSindex;
        }
        else
        {
            double v = uniform_real_dist(rngs[i]);
            token = upper_bound (accprobvec.begin(), accprobvec.end(), v) - accprobvec.begin();
        }
        ncesample[chunkiter][i] = token;
    }

    // compute ncesamplecnt
    int i;
    map<int, int> noisesample2cntmap;
    noisesample2cntmap.clear();
    for (i=0; i<k; i++)
    {
        int sampleid = ncesample[chunkiter][i];
        if (noisesample2cntmap.find(sampleid) == noisesample2cntmap.end())
        {
            noisesample2cntmap.insert(make_pair(sampleid, 1));
        }
        else
        {
            noisesample2cntmap[sampleid] ++;
        }
    }
    i = 0;
    for (map<int, int>::iterator it = noisesample2cntmap.begin(); it != noisesample2cntmap.end(); it++)
    {
        ncesample[chunkiter][i] = it->first;
        ncesamplecnt[chunkiter][i] = it->second;
        i ++;
    }
    // set valid nunber of nce noise sample
    nncesample[chunkiter] = noisesample2cntmap.size();

    // compute mbid2arrid
    int idx = 0;
    map<int, int> targetsample2idxmap;
    targetsample2idxmap.clear();
    for (int mbidx=0; mbidx<minibatch; mbidx++)
    {
        // int wordid = targetsample[chunkiter][mbidx];
        int wordid = host_curwords[mbidx];
        if (targetsample2idxmap.find(wordid) == targetsample2idxmap.end())
        {
            targetsample[chunkiter][idx] = wordid;
            targetsample2idxmap.insert(make_pair(wordid, idx));
            idx ++;
        }
        mbid2arrid[chunkiter][mbidx] = targetsample2idxmap[wordid];
    }
    // set valid nunber of target sample
    ntargetsample[chunkiter] = targetsample2idxmap.size();
    // set valid output layer size for NCE weight
    outputlayersize_NCE[chunkiter] = ntargetsample[chunkiter]+nncesample[chunkiter];

    cucpytoGpu (dev_ncesample[chunkiter], ncesample[chunkiter], sizeof(int)*k);
    cucpytoGpu (dev_targetsample[chunkiter], targetsample[chunkiter], sizeof(int)*minibatch);
    cucpytoGpu (dev_ncesamplecnt[chunkiter], ncesamplecnt[chunkiter], sizeof(int)*k);
    cucpytoGpu (dev_mbid2arrid[chunkiter], mbid2arrid[chunkiter], sizeof(int)*minibatch);

    // copy output weight of sample (noise and target) to layerN_NCE
    layerN_NCE_vec[chunkiter]->copyOutputWgtsforNCE (U, dev_targetsample[chunkiter], ntargetsample[chunkiter], dev_ncesample[chunkiter], nncesample[chunkiter]);
    layerN_NCE_vec[chunkiter]->setncols(outputlayersize_NCE[chunkiter]);

    neuN_ac_NCE_vec[chunkiter]->setnrows(outputlayersize_NCE[chunkiter]);
}


void inputlayer::setLRtunemode (int lr)
{
    lrtunemode = lr;
    if (lrtunemode == 0)
    {
        accsdU = NULL;
    }
    else
    {
        accsdU = new matrix (nrows, ncols);
        accsdU->initmatrix ();
    }
}

void outputlayer::setLRtunemode (int lr)
{
    lrtunemode = lr;
    if (lrtunemode == 0)
    {
        accsdU = NULL;
    }
    else
    {
        accsdU = new matrix (nrows, ncols);
        accsdU->initmatrix ();
    }
}

void recurrentlayer::setLRtunemode (int lr)
{
    lrtunemode = lr;
    if (lrtunemode == 0)
    {
        accsdU  = NULL;
        accsdW  = NULL;
    }
    else
    {
        accsdU = new matrix (nrows, ncols);
        accsdW = new matrix (ncols, ncols);
        accsdU->initmatrix ();
        accsdW->initmatrix ();
    }
}

void feedforwardlayer::setLRtunemode (int lr)
{
    lrtunemode = lr;
    if (lrtunemode == 0)
    {
        accsdU  = NULL;
    }
    else if (lrtunemode == 1 || lrtunemode == 2)
    {
        accsdU = new matrix (nrows, ncols);
        accsdU->initmatrix ();
    }
}

void linearlayer::setLRtunemode (int lr)
{
    lrtunemode = lr;
    if (lrtunemode == 0)
    {
        accsdU  = NULL;
    }
    else if (lrtunemode == 1 || lrtunemode == 2)
    {
        accsdU = new matrix (nrows, ncols);
        accsdU->initmatrix ();
    }
}


void grulayer::setLRtunemode (int lr)
{
    lrtunemode = lr;
    if (lrtunemode == 0)
    {
        accsdWr = NULL;
        accsdWz = NULL;
        accsdWh = NULL;
        accsdUr = NULL;
        accsdWz = NULL;
        accsdWh = NULL;
    }
    else
    {
        accsdWr = new matrix (ncols, ncols);
        accsdWz = new matrix (ncols, ncols);
        accsdWh = new matrix (ncols, ncols);
        accsdUr = new matrix (nrows, ncols);
        accsdUz = new matrix (nrows, ncols);
        accsdUh = new matrix (nrows, ncols);
        accsdWr->initmatrix ();
        accsdWz->initmatrix ();
        accsdWh->initmatrix ();
        accsdUr->initmatrix ();
        accsdUz->initmatrix ();
        accsdUh->initmatrix ();
    }
}

void gruhighwaylayer::setLRtunemode (int lr)
{
    // GRU part
    lrtunemode = lr;
    if (lrtunemode == 0)
    {
        accsdWr = NULL;
        accsdWz = NULL;
        accsdWh = NULL;
        accsdUr = NULL;
        accsdWz = NULL;
        accsdWh = NULL;
    }
    else
    {
        accsdWr = new matrix (ncols, ncols);
        accsdWz = new matrix (ncols, ncols);
        accsdWh = new matrix (ncols, ncols);
        accsdUr = new matrix (nrows, ncols);
        accsdUz = new matrix (nrows, ncols);
        accsdUh = new matrix (nrows, ncols);
        accsdWr->initmatrix ();
        accsdWz->initmatrix ();
        accsdWh->initmatrix ();
        accsdUr->initmatrix ();
        accsdUz->initmatrix ();
        accsdUh->initmatrix ();
    }
    // highway part
    if (lrtunemode == 0)
    {
        accsdUhw = NULL;
        accsdWhw = NULL;
    }
    else
    {
        accsdUhw = new matrix (nrows, ncols);
        accsdWhw = new matrix (ncols, ncols);
        accsdUhw->initmatrix ();
        accsdWhw->initmatrix ();
    }
}

void lstmlayer::setLRtunemode (int lr)
{
    lrtunemode = lr;
    if (lrtunemode == 0)
    {
        accsdUz = NULL;
        accsdUi = NULL;
        accsdUf = NULL;
        accsdUo = NULL;
        accsdWz = NULL;
        accsdWi = NULL;
        accsdWf = NULL;
        accsdWo = NULL;
        accsdPi = NULL;
        accsdPf = NULL;
        accsdPo = NULL;
    }
    else
    {
        accsdUz = new matrix (nrows, ncols);
        accsdUi = new matrix (nrows, ncols);
        accsdUf = new matrix (nrows, ncols);
        accsdUo = new matrix (nrows, ncols);
        accsdWz = new matrix (ncols, ncols);
        accsdWi = new matrix (ncols, ncols);
        accsdWf = new matrix (ncols, ncols);
        accsdWo = new matrix (ncols, ncols);
        accsdPi = new matrix (ncols, 1);
        accsdPf = new matrix (ncols, 1);
        accsdPo = new matrix (ncols, 1);
        dPi_1col = new matrix (ncols, 1);
        dPf_1col = new matrix (ncols, 1);
        dPo_1col = new matrix (ncols, 1);
        accsdUz->initmatrix ();
        accsdUi->initmatrix ();
        accsdUf->initmatrix ();
        accsdUo->initmatrix ();
        accsdWz->initmatrix ();
        accsdWi->initmatrix ();
        accsdWf->initmatrix ();
        accsdWo->initmatrix ();
        accsdPi->initmatrix ();
        accsdPf->initmatrix ();
        accsdPo->initmatrix ();
        dPi_1col->initmatrix ();
        dPf_1col->initmatrix ();
        dPo_1col->initmatrix ();
    }
}

void lstmhighwaylayer::setLRtunemode (int lr)
{
    // LSTM part
    lrtunemode = lr;
    if (lrtunemode == 0)
    {
        accsdUz = NULL;
        accsdUi = NULL;
        accsdUf = NULL;
        accsdUo = NULL;
        accsdWz = NULL;
        accsdWi = NULL;
        accsdWf = NULL;
        accsdWo = NULL;
        accsdPi = NULL;
        accsdPf = NULL;
        accsdPo = NULL;
        dPi_1col = NULL;
        dPo_1col = NULL;
        dPf_1col = NULL;
    }
    else if (lrtunemode == 1 || lrtunemode == 2)
    {
        accsdUz = new matrix (nrows, ncols);
        accsdUi = new matrix (nrows, ncols);
        accsdUf = new matrix (nrows, ncols);
        accsdUo = new matrix (nrows, ncols);
        accsdWz = new matrix (ncols, ncols);
        accsdWi = new matrix (ncols, ncols);
        accsdWf = new matrix (ncols, ncols);
        accsdWo = new matrix (ncols, ncols);
        accsdPi = new matrix (ncols, 1);
        accsdPf = new matrix (ncols, 1);
        accsdPo = new matrix (ncols, 1);
        dPi_1col = new matrix (ncols, 1);
        dPf_1col = new matrix (ncols, 1);
        dPo_1col = new matrix (ncols, 1);
        accsdUz->initmatrix ();
        accsdUi->initmatrix ();
        accsdUf->initmatrix ();
        accsdUo->initmatrix ();
        accsdWz->initmatrix ();
        accsdWi->initmatrix ();
        accsdWf->initmatrix ();
        accsdWo->initmatrix ();
        accsdPi->initmatrix ();
        accsdPf->initmatrix ();
        accsdPo->initmatrix ();
        dPi_1col->initmatrix ();
        dPf_1col->initmatrix ();
        dPo_1col->initmatrix ();
    }
    // highway part
    if (lrtunemode == 0)
    {
        accsdUhw = NULL;
        accsdPhw = NULL;
        accsdRhw = NULL;
        dPhw_1col = NULL;
        dRhw_1col = NULL;
    }
    else if (lrtunemode == 1 || lrtunemode == 2)
    {
        accsdUhw = new matrix (nrows, ncols);
        accsdPhw = new matrix (nrows, 1);
        accsdRhw = new matrix (nrows, 1);
        dPhw_1col = new matrix (nrows, 1);
        dRhw_1col = new matrix (nrows, 1);
        accsdUhw->initmatrix ();
        accsdPhw->initmatrix ();
        accsdRhw->initmatrix ();
        dPhw_1col->initmatrix ();
        dRhw_1col->initmatrix ();
    }
}

bool layer::isDropout ()
{
    if (dropoutrate > 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void lstmlayer::copyvardropoutmask ()
{
    if (chunkiter != 0)
    {
        dropoutmaskMat_vec[chunkiter]->assign (dropoutmaskMat_vec[chunkiter-1]);
        dropoutHiddenmaskMat_vec[chunkiter]->assign (dropoutHiddenmaskMat_vec[chunkiter-1]);
    }
    else
    {
        dropoutmaskMat_vec[chunkiter]->assign (dropoutmaskMat_vec[chunksize-1]);
        dropoutHiddenmaskMat_vec[chunkiter]->assign (dropoutHiddenmaskMat_vec[chunksize-1]);
    }
}
void outputlayer::copyvardropoutmask ()
{
    if (chunkiter != 0)
    {
        dropoutmaskMat_vec[chunkiter]->assign (dropoutmaskMat_vec[chunkiter-1]);
    }
    else
    {
        dropoutmaskMat_vec[chunkiter]->assign (dropoutmaskMat_vec[chunksize-1]);
    }
}


void lstmlayer::genvardropoutmask (int mbidx)
{
    dropoutmaskMat_vec[chunkiter]->genvardropoutmask(mbidx, dropoutrate);
    dropoutHiddenmaskMat_vec[chunkiter]->genvardropoutmask(mbidx, dropoutrate);
}
void outputlayer::genvardropoutmask (int mbidx)
{
    dropoutmaskMat_vec[chunkiter]->genvardropoutmask(mbidx, dropoutrate);
}

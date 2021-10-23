
/***************************************************************************/
/* Name:       active_contour_minimization_mex.c                               
Date:          April 3, 2009                                                 
Author:        Xavier Bresson (xbresson@math.ucla.edu)                  
Description:   Fast minimization algorithm for General Active Contour Model
For more details, see the report:
X. Bresson, "A Short Guide on a Fast Global Minimization Algorithm for Active Contour Models"
See also these reports: 
1- T. Goldstein, X. Bresson, and S. Osher, Geometric Applications of the 
Split Bregman Method: Segmentation and Surface Reconstruction,
CAM Report 09-06, 2009. 
2- T.F.Chan, L.A.Vese, Active contours without edges, IEEE
Transactions on Image Processing. 10:2, pp. 266-277, 2001.
(Segmentation model for smooth/non-texture images)
3- N. Houhou, J-P. Thiran and X. Bresson, 
Fast Texture Segmentation based on Semi-Local Region Descriptor and Active Contour, 2009
(Segmentation model for texture images)

With corrections from
L. Antonelli and V. De Simone, Comparison of minimization methods for
nonsmooth image segmentation, Communications in Applied and Industrial                                                                                                                        
Mathematics, 9:1, pp. 68-86, 2018
Name:          ac_mex.c
Date:          2018
Author:        Laura Antonelli
Change Log:    Removed code parts that handle segmentation of texture images
               Replace macro SQRT with sqrt function from math.h
               Replace function vComputChanVese with function GetRegionsAverages
	       that returns also area values inside and outside segmented region
	       Correct the sign of term h_r
               Remove normalization of lambda with respect the range of h_r
	       Remove the term square of Edge Detector function in the
	       computation of d^{k+1} (Soft-Thresholding)
	       Parameterize the maximum number of outer/ inner iterations
	       Decrease the error tolerance */
/***************************************************************************/


/*  compilation command (under matlab): mex ac_mex.c  */

/*  Organization of the code:
 First part: Functions that computes the region-term for Chan-Vese Model
 Second part: Functions that minimizes the energy */

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mex.h>

const double PI = 3.14159265358979323846264338327950288;

#define X(ix,iy) (ix)*iNy+ (iy)

#define ABS(x) ( (x) > 0.0 ? x : -(x) )
#define SQR(x) (x)*(x)





/****************************************/
/* SEGMENTATION MODEL FOR CHAN-VESE MODEL */
/* See: T.F.Chan, L.A.Vese, Active contours without edges, IEEE
Transactions on Image Processing. 10:2, pp. 266-277, 2001. */
/****************************************/
void GetRegionsAverages (
  double  *pfIm0, 
  double  *pfu,
  int    iNx,
  int    iNy,
  double  *pfHr,
  int    NitUpdateHr,
  double  *pfMeanIn,
  double  *pfMeanOut,
  double  *pfAreaIn,
  double  *pfAreaOut,
  int    iNI
  )
 
{
    double  fMeanIn, fMeanOut, fNormalizationIn, fNormalizationOut;
    int    ix, iy;
    
    
    /* Update Chan-Vese region function Hr every "NitUpdateHr" iterations */
    if ( (iNI==0) || (iNI%NitUpdateHr==0) )
    {
        
        /* Compute inside and outside means */
        fMeanIn = 0.0; fMeanOut = 0.0;
        fNormalizationIn = 0.0; fNormalizationOut = 0.0;

        for (ix=0; ix< iNx; ix++) {
            for (iy=0; iy< iNy; iy++) {
            fMeanIn += pfIm0[X(ix,iy)]*pfu[X(ix,iy)];
            fNormalizationIn += pfu[X(ix,iy)];
            fMeanOut += pfIm0[X(ix,iy)]*(1.0-pfu[X(ix,iy)]);
            fNormalizationOut += 1.0-pfu[X(ix,iy)];
            }
       }

        if (fNormalizationIn>0.0) 
           fMeanIn = fMeanIn/ fNormalizationIn;
        else
           fMeanIn = 0.;

        *pfAreaIn = fNormalizationIn;
        if (fNormalizationOut>0.0) 
           fMeanOut = fMeanOut/ fNormalizationOut;
        else
           fMeanOut = 0.;
        *pfAreaOut = fNormalizationOut;
        
        
        /* Compute region term h_r = (c1-I)^2 - (c2-I)^2*/
        for (ix=0; ix< iNx; ix++)
            for (iy=0; iy< iNy; iy++)
                pfHr[X(ix,iy)] = +( SQR(fMeanIn-pfIm0[X(ix,iy)]) - SQR(fMeanOut-pfIm0[X(ix,iy)]) ); /* MODIFICATO da LAURA */
        
        *pfMeanIn = fMeanIn;
        *pfMeanOut = fMeanOut;
        /* printf("??? su DAPI fMeanIn %lf fMeanOut %lf fNormalizationIn %lf fNormalizationOut %lf \n",fMeanIn, fNormalizationIn, fMeanOut, fNormalizationOut); */
        
    }
}

/* ---------------------------------------------------------------------------
 Main segmentation algorithm.  Segment a grayscale image into foreground and
 background regions based on the model described in the paper
 "Active Contours Without Edges" by Chan & Vese. */

void ChanVeseSegmentationNew (double *pfIm0, int width, int height, int iNf, double *pfu, int KMax, int LMax, double fLambda, double fMu, double *pfHr, double *fMeanIn, double *fMeanOut, double *pfAreaIn, double *pfAreaOut)
{
    
    double   *pfdx, *pfdy,  *pfuOld, *pfuOld2;
    double   *pfbx, *pfby;
    double   fct1, fct2, fctST, fG, fDxu, fDyu, fs, fTemp, fSumDiff;
    double   fct1b, fct2b, fct1c, fct2c, fInvMu, fInvMu2, f1, f2;
    double   fRangeHr, fMinHr, fMaxHr, fNyx;
    double   fDiffNew, fDiffOld, fDiffNew2, fSumU, fError;
    double   fSumUold, fDiffFirst2, fStopThres;
    int     iNy, iNx, ix, iy;
    int     iNI, iX, iGS, iRD ;
    int     NitUpdateHr, iMeanGS, iCptGS;

    int MODEL_CV;

 
    NitUpdateHr = 1; /* laura: valutare */

    /* Image Size */
    iNx = height; /* Number of rows */
    iNy = width;  /* Number of columns */

    MODEL_CV = 0;
    iRD = MODEL_CV; /* Chan Vese Model */ /* laura fissare iRD come argomento di input */
    
    pfdx = (double *) calloc( (unsigned)(iNy*iNx), sizeof(double) );
    if (!pfdx)
        printf("Memory allocation failure\n");
    
    pfdy = (double *) calloc( (unsigned)(iNy*iNx), sizeof(double) );
    if (!pfdy)
        printf("Memory allocation failure\n");
    
    pfbx = (double *) calloc( (unsigned)(iNy*iNx), sizeof(double) );
    if (!pfbx)
        printf("Memory allocation failure\n");
    
    pfby = (double *) calloc( (unsigned)(iNy*iNx), sizeof(double) );
    if (!pfby)
        printf("Memory allocation failure\n");
    
    pfuOld = (double *) calloc( (unsigned)(iNy*iNx), sizeof(double) );
    if (!pfuOld)
        printf("Memory allocation failure\n");
    
    pfuOld2 = (double *) calloc( (unsigned)(iNy*iNx), sizeof(double) );
    if (!pfuOld2)
        printf("Memory allocation failure\n");

    
    /* Normalize lambda value for the computation of Gauss-Seidel iterations */
    /* Estimate of h_r */
    if (iRD==MODEL_CV) /* CHAN-VESE MODEL */
        GetRegionsAverages(pfIm0,pfu,iNx,iNy,pfHr,NitUpdateHr,fMeanIn,fMeanOut,pfAreaIn,pfAreaOut,0);

    /* printf("matlab prima del ciclo ==> fMeanIn %lf, fMeanOut %lf, pfAreaIn %lf, pfAreaOut  %lf\n",*fMeanIn,*fMeanOut,*pfAreaIn,*pfAreaOut); */


    /* Estimate of the range of h_r */
    fMinHr = 1e10;
    fMaxHr = -1e10;
    for (ix=1; ix< iNx-1; ix++)
        for (iy=1; iy< iNy-1; iy++)
        {
           if ( pfHr[X(ix,iy)]>fMaxHr ) fMaxHr = pfHr[X(ix,iy)];
           if ( pfHr[X(ix,iy)]<fMinHr ) fMinHr = pfHr[X(ix,iy)];
        }
    fRangeHr = fMaxHr-fMinHr;

    /* Normalize lambda with respect to the range of h_r */
    /* fLambda /= fRangeHr; */ /* laura: commento la normalizzazione del parametro */
    
    /* Constants */
    NitUpdateHr = 1;  /* number of iterations to update the region function h_r */
    
    fInvMu = 1./ fMu;
    fInvMu2 = SQR(fInvMu);
    
    fct1 = 1./4.;
    fct2 = fLambda/(4.0*fMu);
    fct1b = 1./3.;
    fct2b = fLambda/(3.0*fMu);
    fct1c = 1./2.;
    fct2c = fLambda/(2.0*fMu);

    fNyx = (double)(iNy*iNx);
    

    if (iRD==MODEL_CV) /* CHAN-VESE MODEL */
        fStopThres = 1e-6;
        
    
    
    /* Iterative minimization scheme
    Split Bregman Method
    Iterative scheme:
    (u^k+1,d^k+1) = arg min int g_b |d| + lambda h_r u + mu/2 |d - grad u - b^k|^2
    b^k+1 = b^k + grad u^k+1 - d^k+1 */
    
    fDiffOld = 1e10; fDiffNew = 1e11;
    iMeanGS = 0; iCptGS = 0;
    iNI=0; /* number of iterations (outer iterations) */
    while ( ABS(fDiffNew-fDiffOld)>fStopThres && iNI<KMax )  /* default value KMax = 30 */
    {
        
        /* Store u^old for outer iterations */
        for (ix=0; ix< iNx; ix++)
            for (iy=0; iy< iNy; iy++)
                pfuOld[X(ix,iy)] = pfu[X(ix,iy)];
        
        /* Update region function hr */
        if (iRD==MODEL_CV) /* CHAN-VESE MODEL */
            GetRegionsAverages(pfIm0,pfu,iNx,iNy,pfHr,NitUpdateHr,fMeanIn,fMeanOut,pfAreaIn,pfAreaOut,iNI);
        
        /* Compute u^{k+1} with Gauss-Seidel */
        /* Solve u^k+1 = arg min int lambda h_r u + mu/2 |d - grad u - b^k|^2 */
        /* Euler-Lagrange is  mu Laplacian u = lambda hr + mu div (b^k-d^k), u in [0,1] */
        iGS=0; /* number of iterations for Gauss-Seidel (inner iterations) */
        fError = 1e10;
        while ( fError>1e-3 && iGS<LMax )  /* default value LMax = 50 */
        {
            
            /* Store u^old for inner iterations (Gauss-Seidel) */
            for (ix=0; ix< iNx; ix++)
                for (iy=0; iy< iNy; iy++)
                    pfuOld2[X(ix,iy)] = pfu[X(ix,iy)];
            
            /* Center */
            for (ix=1; ix< iNx-1; ix++)
                for (iy=1; iy< iNy-1; iy++)
            {
                iX = X(ix,iy);
                fG = pfu[X(ix+1,iy)] + pfu[X(ix-1,iy)] + pfu[X(ix,iy+1)] + pfu[X(ix,iy-1)];
                fG += pfdx[X(ix-1,iy)] - pfdx[iX] - pfbx[X(ix-1,iy)] + pfbx[iX];
                fG += pfdy[X(ix,iy-1)] - pfdy[iX] - pfby[X(ix,iy-1)] + pfby[iX];
                fG *= fct1;
                fG -= fct2* pfHr[iX];
                if(fG>1.0) fG=1.0; else if(fG<0.0) fG=0.0;
                pfu[iX] = fG;
                }
            
            /* Borders */
            ix=0;
            for (iy=1; iy< iNy-1; iy++)
            {
                iX = X(ix,iy);
                fG = pfu[X(ix+1,iy)] + pfu[X(ix,iy+1)] + pfu[X(ix,iy-1)];
                fG += - pfdx[iX] + pfbx[iX];
                fG += pfdy[X(ix,iy-1)] - pfdy[iX] - pfby[X(ix,iy-1)] + pfby[iX];
                fG *= fct1b;
                fG -= fct2b* pfHr[iX];
                if(fG>1.0) fG=1.0; else if(fG<0.0) fG=0.0;
                pfu[iX] = fG;
            }
            
            ix=iNx-1;
            for (iy=1; iy< iNy-1; iy++)
            {
                iX = X(ix,iy);
                fG = pfu[X(ix-1,iy)] + pfu[X(ix,iy+1)] + pfu[X(ix,iy-1)];
                fG += pfdx[X(ix-1,iy)] - pfdx[iX] - pfbx[X(ix-1,iy)] + pfbx[iX];
                fG += pfdy[X(ix,iy-1)] - pfdy[iX] - pfby[X(ix,iy-1)] + pfby[iX];
                fG *= fct1b;
                fG -= fct2b* pfHr[iX];
                if(fG>1.0) fG=1.0; else if(fG<0.0) fG=0.0;
                pfu[iX] = fG;
            }
            
            iy=0;
            for (ix=1; ix< iNx-1; ix++)
            {
                iX = X(ix,iy);
                fG = pfu[X(ix+1,iy)] + pfu[X(ix-1,iy)] + pfu[X(ix,iy+1)];
                fG += pfdx[X(ix-1,iy)] - pfdx[iX] - pfbx[X(ix-1,iy)] + pfbx[iX];
                fG += - pfdy[iX] + pfby[iX];
                fG *= fct1b;
                fG -= fct2b* pfHr[iX];
                if(fG>1.0) fG=1.0; else if(fG<0.0) fG=0.0;
                pfu[iX] = fG;
            }
            
            iy=iNy-1;
            for (ix=1; ix< iNx-1; ix++)
            {
                iX = X(ix,iy);
                fG = pfu[X(ix+1,iy)] + pfu[X(ix-1,iy)] + pfu[X(ix,iy-1)];
                fG += pfdx[X(ix-1,iy)] - pfdx[iX] - pfbx[X(ix-1,iy)] + pfbx[iX];
                fG += pfdy[X(ix,iy-1)] - pfdy[iX] - pfby[X(ix,iy-1)] + pfby[iX];
                fG *= fct1b;
                fG -= fct2b* pfHr[iX];
                if(fG>1.0) fG=1.0; else if(fG<0.0) fG=0.0;
                pfu[iX] = fG;
            }
            
            ix=0; iy=0;
            iX = X(ix,iy);
            fG = pfu[X(ix+1,iy)] + pfu[X(ix,iy+1)];
            fG += - pfdx[iX] + pfbx[iX];
            fG += - pfdy[iX] + pfby[iX];
            fG *= fct1c;
            fG -= fct2c* pfHr[iX];
            if(fG>1.0) fG=1.0; else if(fG<0.0) fG=0.0;
            pfu[iX] = fG;
            
            ix=iNx-1; iy=0;
            iX = X(ix,iy);
            fG = pfu[X(ix-1,iy)] + pfu[X(ix,iy+1)];
            fG += pfdx[X(ix-1,iy)] - pfdx[iX] - pfbx[X(ix-1,iy)] + pfbx[iX];
            fG += - pfdy[iX] + pfby[iX];
            fG *= fct1c;
            fG -= fct2c* pfHr[iX];
            if(fG>1.0) fG=1.0; else if(fG<0.0) fG=0.0;
            pfu[iX] = fG;
            
            ix=0; iy=iNy-1;
            iX = X(ix, iy);
            fG = pfu[X(ix+1, iy)] + pfu[X(ix, iy-1)];
            fG += - pfdx[iX] + pfbx[iX];
            fG += pfdy[X(ix, iy-1)] - pfdy[iX] - pfby[X(ix, iy-1)] + pfby[iX];
            fG *= fct1c;
            fG -= fct2c* pfHr[iX];
            if(fG>1.0) fG=1.0; else if(fG<0.0) fG=0.0;
            pfu[iX] = fG;
            
            ix=iNx-1; iy=iNy-1;
            iX = X(ix,iy);
            fG = pfu[X(ix-1,iy)] + pfu[X(ix,iy-1)];
            fG += pfdx[X(ix-1,iy)] - pfdx[iX] - pfbx[X(ix-1,iy)] + pfbx[iX];
            fG += pfdy[X(ix,iy-1)] - pfdy[iX] - pfby[X(ix,iy-1)] + pfby[iX];
            fG *= fct1c;
            fG -= fct2c* pfHr[iX];
            if(fG>1.0) fG=1.0; else if(fG<0.0) fG=0.0;
            pfu[iX] = fG;
            /* end Borders */
            
            
            /* Compute diff ( u - uold ) */
            fSumDiff = 0.0;
            fSumU = 0.0;
            fSumUold = 0.0;
            for (ix=0; ix< iNx; ix++)
                for (iy=0; iy< iNy; iy++)
                    fSumDiff += SQR(pfu[X(ix,iy)]-pfuOld2[X(ix,iy)]);
            fDiffNew2 = fSumDiff/ fNyx;
            if ( iGS==0 ) 
            {
                fDiffFirst2 = fDiffNew2;
                fDiffNew2 = 1e10;
                fError = 1e10;
            }
            else
                fError = 1.0 - ABS(fDiffNew2-fDiffFirst2)/fDiffFirst2;
            iGS++;
            
        }
        

        iMeanGS += iGS;
        iCptGS++;
        

        /* Compute d^{k+1} (Soft-Thresholding) and b^{k+1} (Bregman function) */
        /* d^k+1 = arg min int g_b |d| + mu/2 |d - grad u - b^k|^2 */
        /* d^k+1 = (grad u^k+1 + b^k)/ |grad u^k+1 + b^k| max(|grad u^k+1 + b^k|-1/mu,0) */
        /* b^k+1 = b^k + grad u^k+1 - d^k+1 */
        /* Center */
        for (ix=0; ix< iNx-1; ix++)
            for (iy=0; iy< iNy-1; iy++)
        {
            iX = X(ix,iy);
            /* d */
            fDxu = pfu[X(ix+1,iy)] - pfu[iX];
            fDyu = pfu[X(ix,iy+1)] - pfu[iX];
            f1 = fDxu+pfbx[iX];
            f2 = fDyu+pfby[iX];
            fs = SQR(f1)+SQR(f2);
            fctST = fInvMu2;
            if ( fs<fctST ) { pfdx[iX]=0.0; pfdy[iX]=0.0; }
            else {
                fs = sqrt(fs);
                fctST = sqrt(fctST);
                fTemp = fs-fctST; fTemp /= fs;
                pfdx[iX] = fTemp* f1;
                pfdy[iX] = fTemp* f2; }
            /* b */
            pfbx[iX] += fDxu - pfdx[iX];
            pfby[iX] += fDyu - pfdy[iX];
            }
        
        /* Borders */
        ix=iNx-1;
        for (iy=1; iy< iNy-1; iy++)
        {
            iX = X(ix,iy);
            fDxu = 0.0;
            fDyu = pfu[X(ix,iy+1)] - pfu[iX];
            f1 = fDxu+pfbx[iX]; f2 = fDyu+pfby[iX];
            fs = SQR(f1)+SQR(f2); fctST = fInvMu2;
            if ( fs<fctST ) { pfdx[iX]=0.0; pfdy[iX]=0.0; }
            else {
                fs = sqrt(fs); fctST = sqrt(fctST);
                fTemp = fs-fctST; fTemp /= fs;
                pfdx[iX] = fTemp* f1;
                pfdy[iX] = fTemp* f2; }
            pfbx[iX] += fDxu - pfdx[iX];
            pfby[iX] += fDyu - pfdy[iX];
        }
        
        iy=iNy-1;
        for (ix=1; ix< iNx-1; ix++)
        {
            iX = X(ix,iy);
            fDxu = pfu[X(ix+1,iy)] - pfu[iX];
            fDyu = 0.0;
            f1 = fDxu+pfbx[iX]; f2 = fDyu+pfby[iX];
            fs = SQR(f1)+SQR(f2); fctST = fInvMu2;
            if ( fs<fctST ) { pfdx[iX]=0.0; pfdy[iX]=0.0; }
            else {
                fs = sqrt(fs); fctST = sqrt(fctST);
                fTemp = fs-fctST; fTemp /= fs;
                pfdx[iX] = fTemp* f1;
                pfdy[iX] = fTemp* f2; }
            pfbx[iX] += fDxu - pfdx[iX];
            pfby[iX] += fDyu - pfdy[iX];
        }
        
        ix=iNx-1; iy=0;
        iX = X(ix,iy);
        fDxu = 0.0;
        fDyu = pfu[X(ix,iy+1)] - pfu[iX];
        f1 = fDxu+pfbx[iX]; f2 = fDyu+pfby[iX];
        fs = SQR(f1)+SQR(f2); fctST = fInvMu2;
        if ( fs<fctST ) { pfdx[iX]=0.0; pfdy[iX]=0.0; }
        else {
            fs = sqrt(fs); fctST = sqrt(fctST);
            fTemp = fs-fctST; fTemp /= fs;
            pfdx[iX] = fTemp* f1;
            pfdy[iX] = fTemp* f2; }
        pfbx[iX] += fDxu - pfdx[iX];
        pfby[iX] += fDyu - pfdy[iX];
        
        ix=0; iy=iNy-1;
        iX = X(ix,iy);
        fDxu = pfu[X(ix+1,iy)] - pfu[iX];
        fDyu = 0.0;
        f1 = fDxu+pfbx[iX]; f2 = fDyu+pfby[iX];
        fs = SQR(f1)+SQR(f2); fctST = fInvMu2;
        if ( fs<fctST ) { pfdx[iX]=0.0; pfdy[iX]=0.0; }
        else {
            fs = sqrt(fs); fctST = sqrt(fctST);
            fTemp = fs-fctST; fTemp /= fs;
            pfdx[iX] = fTemp* f1;
            pfdy[iX] = fTemp* f2; }
        pfbx[iX] += fDxu - pfdx[iX];
        pfby[iX] += fDyu - pfdy[iX];
        
        ix=iNx-1; iy=iNy-1;
        iX = X(ix,iy);
        fDxu = 0.0;
        fDyu = 0.0;
        f1 = fDxu+pfbx[iX]; f2 = fDyu+pfby[iX];
        fs = SQR(f1)+SQR(f2); fctST = fInvMu2;
        if ( fs<fctST ) { pfdx[iX]=0.0; pfdy[iX]=0.0; }
        else {
            fs = sqrt(fs); fctST = sqrt(fctST);
            fTemp = fs-fctST; fTemp /= fs;
            pfdx[iX] = fTemp* f1;
            pfdy[iX] = fTemp* f2; }
        pfbx[iX] += fDxu - pfdx[iX];
        pfby[iX] += fDyu - pfdy[iX];
        /* Borders */


        
        fSumDiff = 0.0;
        fSumU = 0.0;
        fSumUold = 0.0;
        for (ix=0; ix< iNx; ix++)
            for (iy=0; iy< iNy; iy++)
        {
            fSumDiff += SQR(pfu[X(ix,iy)]-pfuOld[X(ix,iy)]);
            fSumU += SQR(pfu[X(ix,iy)]);
            fSumUold += SQR(pfuOld[X(ix,iy)]);
            }
        fDiffOld = fDiffNew;
        fDiffNew = fSumDiff/ (fSumU*fSumUold);

        iNI++;

    }
    iMeanGS /= iNI;

    printf("     outer iterations %d - (mean) inner iterations %d \n",iNI, iMeanGS);
    
    
    /* Free memory */
    free( (double *) pfdx );
    free( (double *) pfdy );
    free( (double *) pfbx );
    free( (double *) pfby );
    free( (double *) pfuOld );
    free( (double *) pfuOld2 );
    

    
}
/* --------------------------------------------------------------------------- */


void WritePGMimage( unsigned char *img, int width, int height, char *filename ){

  int i;
  FILE *file;

  /* open the image file for writing binary */
  if ( (file = fopen( filename, "wb") ) == NULL )
  {
    printf("file %s could not be created\n", filename );
    exit( EXIT_FAILURE );
  }

  /* write out the PGM Header information */
  fprintf( file, "P5\n");
  fprintf( file, "%d %d\n", width, height );
  fprintf( file, "%d\n", 255 );

  /* write the pixel data */
  for (i = 0; i < height; i++)
    fwrite( &img[i*width], sizeof(unsigned char), width, file );

  /* close the file */
  fclose( file );

}

/* --------------------------------------------------------------------------- */


/* ***** MAIN MEX FUNCTION ***** */


extern void mexFunction(int iNOut, mxArray *pmxOut[], int iNIn, const mxArray *pmxIn[]) 
{
/* mex function per il modulo ChanVeseSegmentationNew */

  /* iNOut: number of outputs
     pmxOut: array of pointers to output arguments */

  /* iNIn: number of inputs
     pmxIn: array of pointers to input arguments */

  int KMax, LMax;
  int str_to_replace_lenght;
  int kk, ii, tt;
  int deltag, deltai;
/*  char *fstr = NULL; */
  char fstr[10];
  char *current_fname = NULL;
  int width, height, iNDim, iDim[2];
  double alfa, alfag;
  double thresh;
  double c1, c2;
  unsigned char *imageOut, *imageTemp;
  double fmin, fmax;
  char *current_fout;
  double dPhiX, dPhiY;
  int flag_bckg_color;

  int i, j;
  int iNf, iNx, iNy;
  double fLambda, fMu, den1, den2;

  int iNbItersUpdateHr;
  double *pfHr; 
  double *pfImDap, *pfImLam, *pfImPcG, *pfImPcG_sc, *pfVecParameters;
  double *pfImReg, *pfImCnt, *pfu, *pfunew;
  double *pfMeanIn, *pfMeanOut, *pfDen1, *pfDen2;


  /* Input Data */
  
  iNIn = 5;
  pfImDap = mxGetData(pmxIn[0]);  /* Dapi Image  */
  pfImLam = mxGetData(pmxIn[1]);  /* Lamin Image */
  pfImPcG = mxGetData(pmxIn[2]);  /* PcG Image   */
  pfu = mxGetData(pmxIn[3]);      /* u function minimizer */
  pfVecParameters = mxGetData(pmxIn[4]); /* Vector of AC parameters */

  /* AC parameters */
  iNf = (int) pfVecParameters[0];   /* frame index       */
  iNy = (int) pfVecParameters[1];   /* number of rows    */
  iNx = (int) pfVecParameters[2];   /* number of columns */
  fLambda =  pfVecParameters[3];    /* regions weight    */
  fMu =  pfVecParameters[4];        /* curvature weight  */
  KMax = (int) pfVecParameters[5];  /* maximum outer iterations */
  LMax = (int) pfVecParameters[6];  /* maximum inner iterations */

  /* Output Data */
  iNOut = 7;
  iNDim = 2;
  iDim[0] = iNy; /* width image */
  iDim[1] = iNx; /* height image */

  pmxOut[0] = mxCreateNumericArray(iNDim, (const int*)iDim, mxDOUBLE_CLASS, mxREAL);
  pfImReg = mxGetData(pmxOut[0]);
  pmxOut[1] = mxCreateNumericArray(iNDim, (const int*)iDim, mxDOUBLE_CLASS, mxREAL);
  pfImCnt = mxGetData(pmxOut[1]);
  pmxOut[2] = mxCreateNumericArray(iNDim, (const int*)iDim, mxDOUBLE_CLASS, mxREAL);
  pfunew = mxGetData(pmxOut[2]);

  iDim[0] = 1;
  iDim[1] = 1;
  pmxOut[3] = mxCreateNumericArray(iNDim, (const int*)iDim, mxDOUBLE_CLASS, mxREAL);
  pfMeanIn = mxGetData(pmxOut[3]);
  pmxOut[4] = mxCreateNumericArray(iNDim, (const int*)iDim, mxDOUBLE_CLASS, mxREAL);
  pfMeanOut = mxGetData(pmxOut[4]);
  pmxOut[5] = mxCreateNumericArray(iNDim, (const int*)iDim, mxDOUBLE_CLASS, mxREAL);
  pfDen1 = mxGetData(pmxOut[5]);
  pmxOut[6] = mxCreateNumericArray(iNDim, (const int*)iDim, mxDOUBLE_CLASS, mxREAL);
  pfDen2 = mxGetData(pmxOut[6]);

  sprintf(fstr,"%d",iNf);

  /* printf("Nel MEX ===> Frame Number %s..\n",fstr); */
  
  iNbItersUpdateHr = 1; 

  width = iNy;
  height = iNx;

  pfHr = (double *) calloc( (unsigned)(iNy*iNx), sizeof(double) );
  if (!pfHr)
     printf("Memory allocation failure\n");


  ChanVeseSegmentationNew(pfImDap, width, height, iNf, pfu, KMax, LMax, fLambda, fMu, pfHr, &c1, &c2, &den1, &den2);  

  printf("     Nucleus Frame ==> fMeanIn %f fMeanOut %f \n",c1,c2);

  GetRegionsAverages(pfImPcG, pfu, width, height, pfHr, iNbItersUpdateHr, &c1, &c2, &den1, &den2, 0); 

  printf("     PcG Frame     ==> fMeanIn %f fMeanOut %f \n",c1,c2);
  /* printf("*** c1 = %f / c2 = %f / den1 = %f / den2 = %f / fmin = %f / fmax = %f ***\n", c1, c2, den1, den2, fmin, fmax); */
  
  pfunew = pfu;
  pfMeanIn[0] = c1;
  pfMeanOut[0] = c2;

  pfDen1[0] = den1;
  pfDen2[0] = den2;


  /* den1 == pixels of nuclear regions */ /* da correggere rispetto al minimo u */
  /* den2 == pixels of background */

  if (den1 > den2)   
    flag_bckg_color = 1; 
  else 
    flag_bckg_color = 0;

/* imageOut conterra' la segmentazione bianco-nero */
  imageOut = (unsigned char *) malloc( width*height * sizeof(unsigned char) );
  if ( imageOut == NULL ) {
      fprintf( stderr, "Not enough memory (%d)\n", width*height );
      return;
  }

if ( flag_bckg_color == 0 ){
for (i = 0; i < height; i++)
  for (j = 0; j < width; j++)
     if (pfu[i*width+j] >= 0.5) {
          imageOut[i*width+j] = (unsigned char) 0;
          pfImReg[i*width+j] = 0;
      }
      else {
          imageOut[i*width+j] = (unsigned char) 255;
          pfImReg[i*width+j] = 255;
      }
}
else {
for (i = 0; i < height; i++)
  for (j = 0; j < width; j++)
     if (pfu[i*width+j] < 0.5) {
          imageOut[i*width+j] = (unsigned char) 0;
          pfImReg[i*width+j] = 0;
      }
      else {
          imageOut[i*width+j] = (unsigned char) 255;
          pfImReg[i*width+j] = 255;
      }
}


/* imageTemp conterra' l'immagine pfImPcG con i contorni */
  imageTemp = (unsigned char *) malloc( width*height * sizeof(unsigned char) );
  if ( imageTemp == NULL ) {
    fprintf( stderr, "Not enough memory (%d)\n", width*height );
    return;
  }

/* imageTemp conterra' l'immagine pfImPcG scalata */
  pfImPcG_sc = (double *) malloc( width*height * sizeof(unsigned char) );
  if ( pfImPcG_sc == NULL ) {
    fprintf( stderr, "Not enough memory (%d)\n", width*height );
    return;
  }
  fmin = 0;
  fmax = 0;
  for (i = 0; i < height*width; i++){
         if (pfImPcG[i] < fmin)
             fmin = pfImPcG[i];
         if (pfImPcG[i] > fmax)
             fmax = pfImPcG[i];
    }
  /*for (i = 0; i < height*width; i++)
           pfImPcG_sc[i] = (pfImPcG[i] - fmin)/(fmax - fmin)* 255.;
   
  for (i = 1; i < height-1; i++)
    for (j = 1; j < width-1; j++) {
       dPhiX = (double) imageOut[(i+1)*width+j] - (double) imageOut[(i-1)*width+j];
       dPhiY = (double) imageOut[i*width+j+1] - (double) imageOut[i*width+j-1];
       imageTemp[i*width+j] = (unsigned char)(dPhiX*dPhiX + dPhiY*dPhiY > 0.0) ? 255 : pfImPcG_sc[i*width+j];
       pfImCnt[i*width+j] = (dPhiX*dPhiX + dPhiY*dPhiY > 0.0) ? 255 : pfImPcG_sc[i*width+j];
    } */

/* commento la scrittura di CVoutRegions
  current_fout = (char*) malloc( (str_to_replace_lenght+strlen("CVoutRegions")+strlen(".pgm")+1) * sizeof(char) );
  if ( current_fout == NULL ) {
    fprintf( stderr, "Not enough memory (%d)\n", (int)(str_to_replace_lenght+strlen("CVoutRegions")+strlen(".pgm")+1) );
    return;
  }
  sprintf(current_fout, "%s%s%s", "CVoutRegions", fstr, ".pgm");
  WritePGMimage(imageOut,width,height,current_fout);
  free(current_fout);
*/
/*  current_fout = (char*) malloc( (str_to_replace_lenght+strlen("CVoutContour")+strlen(".pgm")+1) * sizeof(char) );
  if ( current_fout == NULL ) {
    fprintf( stderr, "Not enough memory (%d)\n", (int)(str_to_replace_lenght+strlen("CVoutContour")+strlen(".pgm")+1) );
    return;
  }
  sprintf(current_fout, "%s%s%s", "CVoutContour", fstr, ".pgm");
  WritePGMimage(imageTemp,width,height,current_fout);
  free(current_fout);*/

/*  current_fout = (char*) malloc( (str_to_replace_lenght+strlen("CVoutRegPcg")+strlen(".pgm")+1) * sizeof(char) );
  if ( current_fout == NULL ) {
    fprintf( stderr, "Not enough memory (%ld)\n", str_to_replace_lenght+strlen("CVoutRegPcg")+strlen(".pgm")+1 );
    return;
  }
  sprintf(current_fout, "%s%s%s", "CVoutRegPcg", fstr, ".pgm");
  WritePGMimage(imageOut,width,height,current_fout);
  free(current_fout); */

  /* printf("------------------------------------------------------------------------------ \n");
  printf("\n"); */


  free(imageOut);
  free(imageTemp);
  free(pfImPcG_sc);

  free(pfHr);
  /*free(base_nucleus_fname);
  free(base_pc_fname);*/

}

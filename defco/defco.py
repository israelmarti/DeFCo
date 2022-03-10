#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The main functions are find2c, hselect, normalize, rvbina, rvextract, setrvs, spbina, splot and vgrid. 
For a more detailed description of how to use this package, please read the ReadMe.
COMPLETAR--https://github.com/NickMilsonPhysics/BinaryStarSolver/blob/master/README.md--
"""
import os
import glob
import os.path 
import numpy as np
import math
from astropy.io import fits
from astropy import units as u
from astropy.io.fits.verify import VerifyWarning
from PyAstronomy import pyasl
from specutils import Spectrum1D
from specutils.manipulation import SplineInterpolatedResampler
from specutils.fitting import fit_continuum
from astropy.modeling.fitting import LinearLSQFitter
from astropy.modeling.polynomial import Chebyshev1D
import scipy.interpolate as sci
from scipy.optimize import curve_fit
import matplotlib.pylab as plt
from matplotlib import gridspec
from matplotlib.widgets import Button
from numba import jit
from progress.bar import ChargingBar
from multiprocessing import Pool
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from specutils.manipulation import gaussian_smooth
spline3 = SplineInterpolatedResampler()

print("==================================\nDetecting Faint Companion (v0.1)\n==================================\n")
print("Available functions list:\n")
print("\tfind2c\n\thselect\n\tnormalize\n\trvbina\n\trvextract\n\tsetrvs\n\tspbina\n\tsplot\n\tvgrid\n\tvexplore\n\n")
def help(function):
    if function.lower() == 'find2c':
        print('\nFIND2C: detect mass ratio and effective temperature of secondary companion for a spectra dataset.\n')
        print('Mandatory arguments:')
        print('       lis - list of observed spectra to process')
        print('       lit - list of templates to use')
        print('    vgamma - best known value for systemic radial velocity (km/s)\n')
        print('Optional arguments:')
        print('       spa - string for primary component name')
        print('       spb - string for secondary component name')
        print('      qmin - minimum mass ratio for cross correlation analysis')
        print('      qmax - maximum mass ratio for cross correlation analysis')    
        print('    deltaq - mass increments for cross correlation analysis')
        print('      wreg - spectral regions for cross correlation analysis sperated by "-"')
        print('     nproc - number of CPU cores to use in computing process\n')
    elif function.lower() == 'hselect':
        print('\nHSELECT: extract keyword values from file or file list.\n')
        print('Mandatory arguments:')
        print('       img - file or file list to process')
        print('   keyword - keyword to be extracted from each selected file\n')
    elif function.lower() == 'normalize':
        print('\nNORMALIZE: scale each spectrum from dataset to a mean continuum factor.\n')
        print('Mandatory arguments:')
        print('       lis - list of observed spectra to process\n')
        print('Optional arguments:')
        print('   interac - show corrected spectra (interactive mode)\n')
    elif function.lower() == 'rvbina':
        print('\nRVBINA: compute radial velocities for spectra dataset\n')
        print('Mandatory arguments:')
        print('       lis - list of observed spectra to process\n')
        print('Optional arguments:')
        print('       spa - string for primary component name')
        print('       spb - string for secondary component name')
        print('        ta - spectrum template to use as primary component')
        print('        tb - spectrum template to use as secondary component')
        print('      wreg - spectral regions for cross correlation analysis sperated by "-"')
        print('       drv - radial velocity step for cross correlation (km/s)')
        print('   rvrange - radial velocities limits for cross correlation (km/s)')    
        print('     aconv - absorver convergence factor')
        print('     keyjd - header keyword for Julian Date')
        print('   interac - show cross correlation function (interactive mode)\n')
    elif function.lower() == 'rvextract':
        print('\nRVEXTRACT: extract radial velocities from spectra dataset and graphicate RV convergence as function of interations.\n')
        print('Mandatory arguments:')
        print('       lis - list of observed spectra to process\n')
        print('Optional arguments:')
        print('    output - output file name')
        print('     graph - show convergence graphics\n')
    elif function.lower() == 'setrvs':
        print('\nSETRVS: set radial velocities to each spectrum from dataset.\n')
        print('Mandatory arguments:')
        print('       lis - list of observed spectra to process\n')
        print('Optional arguments:')
        print('        ta - spectrum template to use as primary component')
        print('        tb - spectrum template to use as secondary component')
        print('      wreg - spectral regions for cross correlation analysis sperated by "-"')
        print('       drv - radial velocity step for cross correlation (km/s)')
        print('   rvrange - radial velocities limits for cross correlation (km/s)')    
        print('     keyjd - header keyword for Julian Date')
        print('   interac - show cross correlation function (interactive mode)\n')
    elif function.lower() == 'spbina':
        print('\nSPBINA: computing spectra for primary and secondary components.\n')
        print('Mandatory arguments:')
        print('       lis - list of observed spectra to process\n')
        print('Optional arguments:')
        print('       spa - string for primary component name')
        print('       spb - string for secondary component name')
        print('       nit - number of iterations')
        print('      frat - flux ratio between components')
        print('    reject - reject pixels using a sigma clipping algorithm')     
        print('         q - mass ratio between components')
        print('    vgamma - best known value for systemic radial velocity (km/s)')      
        print('    obspha - calculate spectra for all phases')
        print('   showtit - enable user interface')
    elif function.lower() == 'splot':
        print('\nSPLOT: plot spectrum from a file in FITS format.\n')
        print('Mandatory arguments:')
        print('      file - file name\n')
        print('Optional arguments:')
        print('      xmin - lower wavelength limit')
        print('      xmax - upper wavelength limit')
        print('      ymin - lower flux limit')
        print('      ymax - upper flux limit')
        print('     scale - flux scale factor')
        print('   markpix - mark pixel values')
        print('    newfig - show spectrum in a new window') 
        print('     color - graph color') 
    elif function.lower() == 'vgrid':
        print('\nVGRID: detect mass ratio and effective temperature of secondary companion using a systemic radial velocities grid.\n')
        print('Mandatory arguments:')
        print('       lis - list of observed spectra to process')
        print('       lit - list of templates to use')
        print('Optional arguments:')
        print('     svmin - lower systemic radial velocity (km/s)')
        print('     svmax - upper systemic radial velocity (km/s)')
        print('      step - radial velocity step for messhing grid (km/s)')
        print('      qmin - minimum mass ratio for cross correlation analysis')
        print('      qmax - maximum mass ratio for cross correlation analysis')    
        print('    deltaq - mass incrememnts forcross correlation analysis')
        print('      wreg - spectral regions for cross correlation analysis sperated by "-"')
        print('     nproc - number of CPU cores to use in computing process\n')
    elif function.lower() == 'vexplore':
        print('\nVEXPLORE: show results for systemic radial velocities grid analysis\n')
        print('Mandatory arguments:')
        print('       obj - object name to show (extracted from header keyword in VGRID')
    else:
        print('\nUnknown function. Please check the name from functions available list:\n')
        print("\tfind2c\n\tnormalize\n\trvbina\n\trvextract\n\tsetrvs\n\tspbina\n\tsplot\n\tvgrid\n\tvexplore\n")

def find2c(lis, lit, spa='A', spb='B', vgamma=0, qmin=0.02, qmax=0.5, deltaq=0.01, wreg='4000-4090,4110-4320,4360-4850,4875-5290,5350-5900',nproc=6):
    if os.path.isdir('CC')==False:
        os.mkdir('CC')
    instanteInicial = datetime.now()
    plt.ion()
    print('\n\tRunning FIND2C (v0.1)\n')
    VerifyWarning('ignore')
    larch=makelist(lis)
    filet=open(lit,'r')
    ltemp=[]
    for tx in filet:
        ltemp.append(tx.rstrip('\n'))
    filet.close()
    q_array=np.arange(round(qmin,7),round(qmax+deltaq,7),round(deltaq,7))
    path1=os.getcwd()
    #compute B spectrum for each element from q_array
    print('Calculating spectra...')
    pool=Pool(processes=nproc)
    qa2=np.array_split(q_array,nproc)
    pres=[pool.apply_async(qparallel, args= [chk,lis,larch,spa,spb,deltaq,vgamma]) for chk in qa2]
    pool.close()
    #pool.join()
    pres = [chk.get() for chk in pres]
    pool.terminate()
    print('\t\t\t\t\tDone!')
    print('')
    #create fading mask
    dlist=[]
    hobs = fits.open(larch[0], 'update')
    d1 = hobs[0].header['CDELT1']
    dlist.append(d1)
    try:
        obj1 = hobs[0].header['OBJECT']
        obj1=obj1.replace(' ','')
    except KeyError:
        obj1='NoObject'
    hobs.close(output_verify='ignore')
    htmp = fits.open(ltemp[0], 'update')
    d2 = htmp[0].header['CDELT1']
    dlist.append(d2)
    htmp.close(output_verify='ignore')
    waux1,faux1 = pyasl.read1dFitsSpec(larch[0])
    waux2,faux2 = pyasl.read1dFitsSpec(ltemp[0])
    #gap expresed wavelenght magins in Angstroms
    gap=50
    winf=max(waux1[0],waux2[0])+gap
    wsup=min(waux1[-1],waux2[-1])-gap
    new_disp_grid,fmask = setregion(wreg,np.max(dlist),winf,wsup)
    #Apply mask over B//q spectra (filtering and apodizing)
    matrix_sq=np.zeros(shape=(len(q_array),len(new_disp_grid)))
    qout = open(obj1+'_q.txt', 'w')
    bar1 = ChargingBar('Loading B_q spectra:', max=len(q_array))
    for j,xq in enumerate(q_array):
        qout.write(str(round(xq,len(str(deltaq+1))-2))+'\n')
        aux1=str(round(xq,len(str(deltaq+1))-2)).replace('.','')
        wimg,fimg = pyasl.read1dFitsSpec(spb+aux1+'.fits')
        spec_cont=continuum(wimg, fimg, type='diff',lo=2.5,hi=3.5, graph=False)
        aux_img = Spectrum1D(flux=spec_cont*u.Jy, spectral_axis=wimg*0.1*u.nm)
        aux2_img = spline3(aux_img, new_disp_grid*0.1*u.nm)
        matrix_sq[j] = splineclean(aux2_img.flux.value)*fmask
        bar1.next()
    bar1.finish()
    print('')
    qout.close()
    #Cross correlation between each spectrum and B//q
    vector_t=np.zeros(len(ltemp))
    matrix_cc=np.zeros(shape=(len(ltemp),len(q_array)))
    tout = open(obj1+'_Teff.txt', 'w')
    vgindex=str(vgamma)
    ccout = open('CC/'+obj1+'_CC_'+'vg_'+vgindex+'.txt', 'w')
    bar2 = ChargingBar('Comparing templates:', max=len(ltemp))
    for k,tmp in enumerate(ltemp):
        htmp = fits.open(tmp, 'update')
        teff= htmp[0].header['TEFF']
        vector_t[k]=teff
        htmp.close(output_verify='ignore')
        wt1,ft1 = pyasl.read1dFitsSpec(tmp)
        temp_cont = continuum(wt1, ft1, type='diff', lo=2.5,hi=6, graph=False)
        aux_tmp1 = Spectrum1D(flux=temp_cont*u.Jy, spectral_axis=wt1*0.1*u.nm)
        aux2_tmp1 = spline3(aux_tmp1, new_disp_grid*0.1*u.nm)
        template1=splineclean(aux2_tmp1.flux.value*fmask)
        tt=np.mean(template1**2)
        tout.write(str(int(teff))+'\n')
        for l,xq in enumerate(q_array):
            bb=np.mean(matrix_sq[l]**2)
            tb=np.mean((template1*matrix_sq[l]))
            cc=tb/(np.sqrt(bb)*np.sqrt(tt))
            matrix_cc[k,l]=cc
            ccout.write(str(round(cc,4))+'\t')
        ccout.write('\n')
        bar2.next()
    bar2.finish()
    print('')
    tout.close()
    ccout.close()
    #Estimate mass ratio q ante effective temperature Teff for according to a parabole
    qmed=np.max(matrix_cc,axis=0)
    iq2=int(np.where(qmed == np.max(qmed))[0])
    if qmed[iq2] !=qmed[0] and qmed[iq2] !=qmed[-1]:
        best_q = q_array[iq2]-(qmed[iq2+1]-qmed[iq2-1])*deltaq/2/(qmed[iq2-1]+qmed[iq2+1]-2*qmed[iq2])
    else:
        best_q=q_array[iq2]
    tmed=np.max(matrix_cc,axis=1)
    jt2=np.argmax(tmed)
    best_B=matrix_sq[iq2]
    rvblist=[]
    for img in larch:
        hdul = fits.open(img, 'update')
        vra = hdul[0].header['VRA']
        vrb = vgamma - (vra-vgamma)/q_array[iq2]
        rvblist.append(round(vrb,3))
        hdul.flush()
        hdul.close()
    rvrange=max(rvblist)-min(rvblist)
    print('\t· · · · · · · · · · · · · ·')
    print('\t· Teff='+str(int(vector_t[jt2]))+' K\tq = '+str(round(best_q,2))+'  ·')
    print('\t· · · · · · · · · · · · · ·')
    #Graph results for q vs Teff
    fig=plt.figure(figsize=[8,5])
    ax = Axes3D(fig)
    ax.set_xlabel("Mass ratio ($q$)", fontsize=10)
    ax.set_ylabel("Temp. [1000 K]", fontsize=10)
    X, Y = np.meshgrid(q_array, vector_t/1000)
    Z = matrix_cc
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet')
    aux3 = os.path.isfile(obj1+'_CC.jpeg')
    if aux3:
        os.remove(obj1+'_CC.jpeg')
    plt.savefig(path1+'/'+obj1+'_CC.jpeg')
    #Graph results for best q vs CC
    fig=plt.figure(figsize=[8,5])
    plt.plot(q_array,qmed,marker='',ls='-', color='blue')
    plt.axvline(x=best_q, color='black', linestyle='--',linewidth=1)
    plt.ylabel("Correlation", fontsize=12)
    plt.xlabel("Mass ratio ($q$)", fontsize=12)
    plt.legend(('q = '+str(round(q_array[iq2],len(str(deltaq+1))-2))))
    plt.tight_layout()
    aux4 = os.path.isfile(obj1+'_q.jpeg')
    if aux4:
        os.remove(obj1+'_q.jpeg')
    plt.savefig(path1+'/'+obj1+'_q.jpeg')
    #Graph results for best Teff vs CC
    fig=plt.figure(figsize=[8,5])
    plt.plot(vector_t,tmed,marker='',ls='-', color='red')
    plt.axvline(x=vector_t[jt2], color='black', linestyle='--',linewidth=1)
    plt.ylabel("Correlation", fontsize=12)
    plt.xlabel("Temperature [K]", fontsize=12)
    plt.legend(('Teff = '+str(int(vector_t[jt2]))+' K'))
    plt.tight_layout()
    aux5 = os.path.isfile(obj1+'_Teff.jpeg')
    if aux5:
        os.remove(obj1+'_Teff.jpeg')
    plt.savefig(path1+'/'+obj1+'_Teff.jpeg')
    #Graph results for RV cross correlation for best q and best Teff
    fig=plt.figure(figsize=[8,5])
    if jt2>=1 and jt2<=(len(ltemp)-1):
        wtj1,ftj1 = pyasl.read1dFitsSpec(ltemp[jt2-1]) 
        wtj2,ftj2 = pyasl.read1dFitsSpec(ltemp[jt2]) 
        wtj3,ftj3 = pyasl.read1dFitsSpec(ltemp[jt2+1]) 
        tleg1=str(int(vector_t[jt2-1]))
        tleg2=str(int(vector_t[jt2]))
        tleg3=str(int(vector_t[jt2+1]))
        drv1, fcc1 = fxcor(new_disp_grid,best_B,wtj1,ftj1,rvmin=-rvrange,rvmax=rvrange,drv=int(rvrange)/50)
        drv2, fcc2 = fxcor(new_disp_grid,best_B,wtj2,ftj2,rvmin=-rvrange,rvmax=rvrange,drv=int(rvrange)/50)
        drv3, fcc3 = fxcor(new_disp_grid,best_B,wtj3,ftj3,rvmin=-rvrange,rvmax=rvrange,drv=int(rvrange)/50)
        plt.plot(drv1,fcc1,marker='',ls='--', color='red')
        plt.plot(drv2,fcc2,marker='',ls='-', color='black')
        plt.plot(drv3,fcc3,marker='',ls='--', color='blue')
        plt.legend((tleg1+' K',tleg2+ 'K',tleg3+' K'))
    else:
        wtj,ftj = pyasl.read1dFitsSpec(larch[jt2])     
        drv1, fcc1 = fxcor(new_disp_grid,best_B,wtj,ftj,rvmin=-rvrange*2,rvmax=rvrange*2,drv=int(4*rvrange)/100)
        plt.plot(drv1,fcc1,marker='',ls='-', color='green')
    plt.ylabel("Correlation", fontsize=12)
    plt.xlabel('Radial Velocity [km/s]')
    plt.tight_layout()
    aux5 = os.path.isfile(obj1+'_RV.jpeg')
    if aux5:
        os.remove(obj1+'_RV.jpeg')
    plt.savefig(path1+'/'+obj1+'_RV.jpeg')
    instanteFinal = datetime.now()
    tiempo = instanteFinal - instanteInicial
    print('')
    print('Total Time Processing:',tiempo.seconds,'s')
################################################################
################################################################
################################################################
def hselect(img,keyword):
    VerifyWarning('ignore')
    larch=makelist(img)
    for i,img in enumerate(larch):
        hdul = fits.open(img)
        try:
            print(img+': '+str(hdul[0].header[keyword]))
        except KeyError:
            print(img)
################################################################
################################################################
################################################################   
def normalize(lis,interac=True):
    #Apply doppler correction for RV=0
    global yours
    plt.close()
    plt.ioff()
    VerifyWarning('ignore')
    larch=makelist(lis)
    #Calculate mean spectrum
    hobs = fits.open(larch[0], 'update')
    dx= hobs[0].header['CDELT1']
    wx1 = hobs[0].header['CRVAL1']
    wx2 = hobs[0].header['CRVAL1'] + dx *(hobs[0].header['NAXIS1']-1)    
    new_disp_grid=np.arange(wx1-50,wx2+50,dx)
    nwvl = len(new_disp_grid)
    obs_matrix = np.zeros(shape=(len(larch),nwvl))
    ldel=[]
    print('\n\tProcessing spectra list...\n')
    larch2=larch.copy()
    for k,img in enumerate(larch):
        wimg,fimg = pyasl.read1dFitsSpec(img)
        rp=np.where(fimg ==0)
        fimg[rp]=np.mean(fimg)
        hdul = fits.open(img, 'update')
        vra = hdul[0].header['VRA']
        hdul.close(output_verify='ignore')
        w2 = wimg * np.sqrt((1.-vra/299792.458)/(1.+vra/299792.458))
        aux_img = Spectrum1D(flux=fimg *u.Jy, spectral_axis=w2 *0.1*u.nm)
        aux_img2 = spline3(aux_img,new_disp_grid*0.1*u.nm)
        if interac:
            fig = plt.figure(figsize=[15,8])
            plt.title('Spectrum '+img)
            plt.xlabel('Wavelenght [A]')
            plt.ylabel('Flux')
            plt.plot(wimg,fimg, color='red',ls='-')  
            plt.plot(new_disp_grid,aux_img2.flux.value,ls='--',marker='',color='blue')
            plt.legend(('Original','RV corrected'))
            plt.tight_layout()
            fig.canvas.mpl_connect('key_press_event', on_key)
            print('\t............................................')
            print('\t:      Press Y to save and continue        :')
            print('\t:  Press any button to discard spectrum    :')
            print('\t············································')
            plt.show()
            if yours == 'y' or yours == 'Y':
                obs_matrix[k]=splineclean(aux_img2.flux.value)
            else:
                ldel.append(k)
                larch2.remove(img)
        else:
            obs_matrix[k]=splineclean(aux_img2.flux.value)
    #delete spectra unselected
    obs_matrix = np.delete(obs_matrix,ldel,axis=0)
    wei=np.mean(obs_matrix,axis=1)/np.max(np.mean(obs_matrix,axis=1))
    smean1 = sfcomb(obs_matrix,wei)
    if interac:      
        fig = plt.figure(figsize=[15,8])
        plt.plot(new_disp_grid,smean1,c='black',ls='--')
        plt.show()
    wtr1 = np.abs(new_disp_grid - (new_disp_grid[0]+50)).argmin(0)
    wtr2 = np.abs(new_disp_grid - (new_disp_grid[-1]-50)).argmin(0)
    grid2=new_disp_grid[wtr1:wtr2]
    for i,img in enumerate(larch2):
        wv,fl = pyasl.read1dFitsSpec(img)
        rp=np.where(fl ==0)
        fl[rp]=np.mean(fl)
        aux_spec = Spectrum1D(flux=fl *u.Jy, spectral_axis=wv *0.1*u.nm)
        spec = spline3(aux_spec, grid2*0.1*u.nm)
        fs=splineclean(spec.flux.value)
        hdul = fits.open(img, 'update')
        vra = hdul[0].header['VRA']
        hdul.close(output_verify='ignore')
        grid_aux = new_disp_grid * np.sqrt((1.+vra/299792.458)/(1.-vra/299792.458))
        aux_smean = Spectrum1D(flux=smean1 *u.Jy, spectral_axis=grid_aux *0.1*u.nm)
        aux_smean2 = spline3(aux_smean,grid2*0.1*u.nm)
        smean2=splineclean(aux_smean2.flux.value)
        s1=fs/smean2
        f_cont = continuum(grid2, s1 ,type='fit',graph=False)
        f_norm = fs / f_cont
        naux = img.replace('.fits','')
        print('')
        print('\tSaving '+img+'...')
        pyasl.write1dFitsSpec(naux+'_NORM.fits', f_norm, grid2, clobber=True)
        copyheader(img,naux+'_NORM.fits')        
################################################################
################################################################
################################################################
def rvbina(lis, spa='A', spb='B', ta='templateA', tb='templateB', wreg='4000-4090,4110-4320,4360-4850,4875-5290,5350-5900', drv=5, rvrange=200, aconv=0.5, keyjd='MJD-OBS', interac=True):
    global rva1,rva2,rvb1,rvb2,gcca,minvalA,minvalB,dimy,cgaussA,cgaussB,xga1,xga2,xgb1,xgb2,initfitA,initfitB
    global vra_aux,vrb_aux,no_rv,yga1,yga2,ygb1,ygb2,yours
    cgaussA=0
    cgaussB=0
    plt.ioff()
    spline3 = SplineInterpolatedResampler()
    VerifyWarning('ignore')
    larch=makelist(lis)
    ta = ta.replace('.fits','')
    tb = tb.replace('.fits','')
    aaux1 = os.path.isfile(ta+'.fits')
    baux1 = os.path.isfile(tb+'.fits')
    if aaux1 == False:
        print('Can not access to primary template spectrum')
        print('END')
    if baux1 == False:
        print('Can not access to secondary template spectrum')
        print('END')
    aaux2 = os.path.isfile(spa+'.fits')
    baux2 = os.path.isfile(spb+'.fits')
    if aaux2 == False:
        print('Can not access to '+spa+'.fits')
        print('END')
    if baux2 == False:
        print('Can not access to '+spb+'.fits')
        print('END')
    if aaux1 and baux1 and aaux2 and baux2:
        for k,img in enumerate(larch):
            print('\t·············································')
            print('\tProcessing '+img+'...\n')
            wimg,fimg = pyasl.read1dFitsSpec(img)
            wa,fa = pyasl.read1dFitsSpec(spa+'.fits')
            wb,fb = pyasl.read1dFitsSpec(spb+'.fits')
            wta, fta = pyasl.read1dFitsSpec(ta+'.fits')
            wtb, ftb = pyasl.read1dFitsSpec(tb+'.fits')
            hdul = fits.open(img, 'update')
            if k==0:
                gap=50
                inta1=10
                inta2=10
                intb1=10
                intb2=10
                yga1=0
                yga2=0
                ygb1=0
                ygb2=0
                delta = hdul[0].header['CDELT1']
                winf=max(wimg[0],wta[0],wtb[0])+gap
                wsup=min(wimg[-1],wta[-1],wtb[-1])-gap
                new_disp_grid,dumm1 = setregion(wreg,delta,winf,wsup)
                try:
                    obj1 = hdul[0].header['OBJECT']
                except KeyError:
                    obj1='NoObject'
            #read RV for each component
            try:
                vra = hdul[0].header['VRA']
            except KeyError:
                vra = None
            try:
                vrb = hdul[0].header['VRB']
            except KeyError:
                vrb = None
            try:
                xjd = hdul[0].header[keyjd]
            except KeyError:
                xjd = np.nan
            #Set RVA y RVB if them do not exist
            if vra==None or vrb==None:
                no_rv=0
                print('\tRadial velocities not found.')
                if k==0 or interac==True:
                    print('\tPlease mark the RV for each component')
                    drvaux1, ccaux1 = fxcor(wimg,fimg,wta,fta,rvmin=-rvrange,rvmax=rvrange,drv=drv)
                    fig=plt.figure(figsize=[8,8])
                    plt.title('Cross-correlation for '+img+' (obs. spec.)')
                    plt.xlabel('Radial Velocity [km/s]')
                    plt.ylabel('Correlation')
                    plt.plot(drvaux1, ccaux1, color='black')
                    plt.tight_layout()
                    fig.canvas.mpl_connect('button_press_event', onclick1)
                    plt.show()
                    aaa1=drvaux1.flat[np.abs(drvaux1 - vra_aux).argmin()]
                    bbb1=drvaux1.flat[np.abs(drvaux1 - vrb_aux).argmin()]
                    aaa2=int(np.where(drvaux1 == aaa1)[0])
                    bbb2=int(np.where(drvaux1 == bbb1)[0])
                    vra=max(drvaux1[aaa2-5:aaa2+5])
                    vrb=max(drvaux1[bbb2-5:bbb2+5])
                else:
                    print('\tPlease input manually the RVs')
                    ira = input('RV for primary comp. (A): ')
                    vra=float(ira)
                    irb = input('RV for secondary comp. (B): ')
                    vrb=float(irb)
            wlprime_A = wa * np.sqrt((1.+vra/299792.458)/(1.-vra/299792.458))
            wlprime_B = wb * np.sqrt((1.+vrb/299792.458)/(1.-vrb/299792.458))
            aux_img = Spectrum1D(flux=fimg*u.Jy, spectral_axis=wimg*0.1*u.nm)
            aux_sa = Spectrum1D(flux=fa*u.Jy, spectral_axis=wlprime_A*0.1*u.nm)
            aux_sb = Spectrum1D(flux=fb*u.Jy, spectral_axis=wlprime_B*0.1*u.nm)
            #Resampling of each spectrum (lineal interpolation) with the template grid dispersion
            aux2_img = spline3(aux_img, new_disp_grid*0.1*u.nm)
            aux2_sa = spline3(aux_sa, new_disp_grid*0.1*u.nm)
            aux2_sb = spline3(aux_sb, new_disp_grid*0.1*u.nm)
            dsA = aux2_img.flux.value - aux2_sa.flux.value
            dsA = splineclean(dsA)
            dsB = aux2_img.flux.value - aux2_sb.flux.value
            dsB = splineclean(dsB)
            #Modify RV ranges
            rva1 = vra-rvrange
            rva2 = vra+rvrange
            rvb1 = vrb-rvrange
            rvb2 = vrb+rvrange      
            stat = 'm'
            initfitA=True
            initfitB=True
            while stat == 'm':
                minvalA=True
                minvalB=True
                drva, cca = fxcor(new_disp_grid,dsB,wta,fta,rvmin=rva1,rvmax=rva2,drv=drv)
                drvb, ccb = fxcor(new_disp_grid,dsA,wtb,ftb,rvmin=rvb1,rvmax=rvb2,drv=drv)
                #Adjust gaussian function to RVA maximum
                if initfitA:
                    iamax=np.argmax(cca)
                    if iamax<=inta1:
                        inta1=iamax
                    elif iamax>=(len(cca)-inta2):
                        inta2=len(cca)-iamax
                else:
                    anear1=drva.flat[np.abs(drva - min(xga1,xga2)).argmin()]
                    anear2=drva.flat[np.abs(drva - max(xga1,xga2)).argmin()]
                    anew1=int(np.where(drva == anear1)[0])
                    anew2=int(np.where(drva == anear2)[0])
                    xa=drva[anew1:anew2]
                    ya=cca[anew1:anew2]
                    iamax=np.argmax(ya)+anew1
                    inta1=iamax-anew1
                    inta2=anew2-iamax
                try:
                    xa=drva[iamax-inta1:iamax+inta2]
                    ya=cca[iamax-inta1:iamax+inta2]
                    ma = np.sum(xa * (ya-(yga1+yga2)/2)) / np.sum(ya-(yga1+yga2)/2)
                    siga = np.sqrt(np.abs(np.sum((ya-(yga1+yga2)/2) * (xa - ma)**2) / np.sum(ya-(yga1+yga2)/2)))
                    pa1,pa2 = curve_fit(Gauss, xa, ya-(yga1+yga2)/2, p0=[np.max((ya-(yga1+yga2)/2)), ma, siga])
                    yagauss=Gauss(xa, *pa1)+(yga1+yga2)/2
                    best_vra=pa1[1]
                    aerr = np.sqrt(np.diag(pa2))[1]
                    shga=True
                except Exception:
                    best_vra=vra
                    aerr=np.nan
                    shga=False
                    inta1=10
                    inta2=10
                    yga1=0
                    yga2=0
                #Adjust gaussian function to RVB maximum
                if initfitB:
                    ibmax=np.argmax(ccb)
                    if ibmax<=intb1:
                        intb1=ibmax
                    elif ibmax>=(len(ccb)-intb2):
                        intb2=len(ccb)-ibmax
                else:
                    bnear1=drvb.flat[np.abs(drvb - min(xgb1,xgb2)).argmin()]
                    bnear2=drvb.flat[np.abs(drvb - max(xgb1,xgb2)).argmin()]
                    bnew1=int(np.where(drvb == bnear1)[0])
                    bnew2=int(np.where(drvb == bnear2)[0])
                    xb=drvb[bnew1:bnew2]
                    yb=ccb[bnew1:bnew2]
                    ibmax=np.argmax(yb)+bnew1
                    intb1=ibmax-bnew1
                    intb2=bnew2-ibmax
                try:
                    xb=drvb[ibmax-intb1:ibmax+intb2]
                    yb=ccb[ibmax-intb1:ibmax+intb2]
                    mb = np.sum(xb * (yb-(ygb1+ygb2)/2)) / np.sum(yb-(ygb1+ygb2)/2)
                    sigb = np.sqrt(np.abs(np.sum((yb-(ygb1+ygb2)/2) * (xb - mb)**2) / np.sum(yb-(ygb1+ygb2)/2)))
                    pb1,pb2 = curve_fit(Gauss, xb, yb-(ygb1+ygb2)/2, p0=[np.max((yb-(ygb1+ygb2)/2)), mb, sigb])
                    ybgauss=Gauss(xb, *pb1)+(ygb1+ygb2)/2
                    best_vrb=pb1[1]
                    berr = np.sqrt(np.diag(pb2))[1]
                    shgb=True
                except Exception:
                    best_vrb=vrb
                    berr=np.nan
                    shgb=False
                    intb1=10
                    intb2=10
                    ygb1=0
                    ygb2=0 
                new_vra = best_vra * aconv + vra * (1 - aconv)
                new_vrb = best_vrb * aconv + vrb * (1 - aconv)
                err_a = np.abs(new_vra - vra)
                err_b = np.abs(new_vrb - vrb)
                if k==0 or interac==True:
                    plt.close()
                    fig = plt.figure(figsize=[8,8])
                    fy1=str(fig).find('x')
                    fy2=len(str(fig))
                    dimy=int(str(fig)[fy1+1:fy2-1])/2
                    ax1 = fig.add_subplot(2, 1, 1)     
                    ax1.set_title('Cross-correlation primary comp. ('+img+')')
                    ax1.set_xlabel('Radial Velocity [km/s]')
                    ax1.set_ylabel('Correlation')
                    ax1.plot(drva, cca, 'b-')
                    if shga:
                        ax1.plot(xa, ya, color='black',marker='.',linestyle='')
                        ax1.plot(xa, yagauss, color='green', label='fit',linestyle='--')
                    else:
                        plt.legend(('No fit'))
                    ax1.text(best_vra, np.max(ya), str(int(best_vra))+' km/s', fontdict={'color':'blue','size': 10,})
                    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1,sharey=ax1)  
                    ax2.set_title('Cross-correlation secondary comp. ('+img+')')
                    ax2.set_xlabel('Radial Velocity [km/s]')
                    ax2.set_ylabel('Correlation')
                    ax2.plot(drvb, ccb, 'r-')
                    if shgb:
                        ax2.plot(xb, yb, color='black',marker='.',linestyle='')
                        ax2.plot(xb, ybgauss, color='green', label='fit',linestyle='--') 
                    else:
                        plt.legend(('No fit'))
                    ax2.text(best_vrb, np.max(yb), str(int(best_vrb))+' km/s', fontdict={'color':'red','size': 10,})
                    plt.tight_layout()
                    fig.canvas.mpl_connect('button_press_event', onclick2)
                    fig.canvas.mpl_connect('key_press_event', on_key)
                    print('\t..........................................')
                    print('\t:        Press any key to refit          :')
                    print('\t:                                        :')
                    print('\t:   Press M to manually input RV range   :')
                    print('\t:      Press R to restart RV range       :')
                    print('\t:     Press Y to save and continue       :')
                    print('\t:   Press N to process next spectrum     :') 
                    print('\t··········································')
                    plt.show()
                    if yours == 'm' or yours == 'M':
                        rva1 = float(input('Estimated minimum RV for primary (A):\n'))
                        rva2 = float(input('Estimated maximum RV for primary (A):\n'))
                        rvb1 = float(input('Estimated minimum RV for secondary (B):\n'))
                        rvb2 = float(input('Estimated maximum RV for secondary (B):\n'))
                    elif  yours == 'r' or yours == 'R':
                        rva1 = vra-rvrange
                        rva2 = vra+rvrange
                        rvb1 = vrb-rvrange
                        rvb2 = vrb+rvrange
                        initfitA=True
                        initfitB=True
                        inta1=10
                        inta2=10
                        intb1=10
                        intb2=10
                        yga1=0
                        yga2=0
                        ygb1=0
                        ygb2=0
                    elif yours == 'y' or yours == 'Y':
                        fsave = True
                        print('\t\t\t\t\tDone!')
                        plt.close()
                        break
                    elif yours == 'n' or yours == 'N' or yours == 'q' or yours == 'Q':
                        fsave = False
                        plt.close()
                        break
                else:
                    fsave = True
                    print('\t\t\t\t\tDone!')
                    break
            if fsave:
                #Save new RV values to the FITS files
                hdul[0].header['VRA'] = round(new_vra,6)
                hdul[0].header['VRB'] = round(new_vrb,6)
                print(img+', VRA: '+str(round(vra,6))+' km/s --> '+str(round(new_vra,6))+' km/s')
                print(img+', VRB: '+str(round(vrb,6))+' km/s --> '+str(round(new_vrb,6))+' km/s')
                hdul.flush(output_verify='ignore')
                hdul.close(output_verify='ignore')
                name=img.replace('.fits','')
                aux2 = os.path.isfile(name+'.log')
                if aux2 == False:
                    flog = open(name+'.log', 'w')
                    flog.write('#RV_A\te_it_A\tRV_B\te_it_B\n')
                else:
                    flog = open(name+'.log', 'a')
                flog.write(str(round(new_vra,6))+'\t'+str(round(err_a,6))+'\t'+str(round(new_vrb,6))+'\t'+str(round(err_b,6))+'\n')
                flog.close()
                #create vr.txt output file
                aux3 = os.path.isfile(obj1+'_RV.txt')
                if aux3 == False:
                    frv = open(obj1+'_RV.txt', 'w')
                    frv.write('#JD\tRV_A\te_A\tRV_B\te_B\n')
                else:
                    frv = open(obj1+'_RV.txt', 'a')
                frv.write(str(xjd)+'\t'+str(round(new_vra,6))+'\t'+str(round(aerr,6))+'\t'+str(round(new_vrb,6))+'\t'+str(round(berr,6))+'\n')
                frv.close()
################################################################
################################################################
################################################################
def rvextract(lis, output='rv.txt', graph=True):
    plt.ion()
    plt.close()
    larch=makelist(lis)
    print('\t······························')
    print('\t    Press ENTER to continue') 
    print('\t      or press Q to exit')
    print('\t······························')
    f2 = open(output, 'w')
    f2.write('#RV_A\tRV_B\n')
    for img in larch:
        print('RV convergence for: '+img)
        name=img.replace('.fits','')
        a=np.loadtxt(name+'.log')
        nit=np.arange(1,len(a)+1,1)
        if graph:
            fig = plt.figure(figsize=[8,6])
            ax1 = fig.add_subplot(2, 1, 1)  
            ax1.set_title('Primary RV convergence')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Radial Velocity [km/s]')
            ax1.plot(nit, a[:,0], 'b-')
            ax2 = fig.add_subplot(2, 1, 2, sharex=ax1,sharey=ax1)  
            ax2.set_title('Secondary RV convergence')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Radial Velocity [km/s]')
            ax2.plot(nit, a[:,2], 'r-')
            fig.tight_layout()
            yours = input()
            plt.close()
            if yours == 'q' or yours == 'Q':
                break
        f2.write(str(a[len(a)-1,0])+'\t'+str(a[len(a)-1,2])+'\n')
    f2.close()
################################################################
################################################################
################################################################
def setrvs(lis, ta='templateA', tb=None, wreg='4000-4090,4110-4320,4360-4850,4875-5290,5350-5900', drv=5, rvrange=200, keyjd='MJD-OBS', interac=True):
    global rva1,rva2,rvb1,rvb2,gcca,minvalA,minvalB,dimy,cgaussA,cgaussB,xga1,xga2,xgb1,xgb2,initfitA,initfitB
    global vra_aux,vrb_aux,no_rv,yga1,yga2,ygb1,ygb2,yours
    cgaussA=0
    cgaussB=0
    plt.ioff()
    VerifyWarning('ignore')
    larch=makelist(lis)
    ta = ta.replace('.fits','')
    tb = tb.replace('.fits','')
    aaux1 = os.path.isfile(ta+'.fits')
    baux1 = os.path.isfile(tb+'.fits')
    if ta !=None:
        if aaux1==False:
            print('Can not access to primary template spectrum')
            print('END')
        else:
            for k,img in enumerate(larch):
                no_rv=0
                print('\t·············································')
                print('\tSet RV for '+img)
                wimg,fimg = pyasl.read1dFitsSpec(img)
                wta, fta = pyasl.read1dFitsSpec(ta+'.fits')
                hdul = fits.open(img, 'update')
                jd=hdul[0].header[keyjd]
                if k==0:
                    gap=50
                    inta1=10
                    inta2=10
                    intb1=10
                    intb2=10
                    yga1=0
                    yga2=0
                    ygb1=0
                    ygb2=0
                    winf=max(wimg[0],wta[0])+gap
                    wsup=min(wimg[-1],wta[-1])-gap
                    delta = hdul[0].header['CDELT1']
                    new_disp_grid,fmask = setregion(wreg,delta,winf,wsup)
                    try:
                        obj1 = hdul[0].header['OBJECT']
                    except KeyError:
                        obj1='NoObject'
                    fout=open('rv_'+obj1+'.txt','w')
                    fout.write('rjd\tvrad\tsvrad\n---\t---\t---\n')
                aux_img = Spectrum1D(flux=fimg*u.Jy, spectral_axis=wimg*0.1*u.nm)
                aux2_img = spline3(aux_img, new_disp_grid*0.1*u.nm)
                fnew=aux2_img.flux.value
                stat = 'm'
                initfitA=True
                while stat == 'm':
                    minvalA=True
                    drva, cca = fxcor(new_disp_grid,fnew,wta,fta,rvmin=-rvrange,rvmax=rvrange,drv=drv)
                    if initfitA:
                        iamax=np.argmax(cca)
                        if iamax<=inta1:
                            inta1=iamax
                        elif iamax>=(len(cca)-inta2):
                            inta2=len(cca)-iamax
                    else:
                        anear1=drva.flat[np.abs(drva - min(xga1,xga2)).argmin()]
                        anear2=drva.flat[np.abs(drva - max(xga1,xga2)).argmin()]
                        anew1=int(np.where(drva == anear1)[0])
                        anew2=int(np.where(drva == anear2)[0])
                        xa=drva[anew1:anew2]
                        ya=cca[anew1:anew2]
                        iamax=np.argmax(ya)+anew1
                        inta1=iamax-anew1
                        inta2=anew2-iamax
                    try:
                        xa=drva[iamax-inta1:iamax+inta2]
                        ya=cca[iamax-inta1:iamax+inta2]
                        ma = np.sum(xa * (ya-(yga1+yga2)/2)) / np.sum(ya-(yga1+yga2)/2)
                        siga = np.sqrt(np.abs(np.sum((ya-(yga1+yga2)/2) * (xa - ma)**2) / np.sum(ya-(yga1+yga2)/2)))
                        pa1,pa2 = curve_fit(Gauss, xa, ya-(yga1+yga2)/2, p0=[np.max((ya-(yga1+yga2)/2)), ma, siga])
                        yagauss=Gauss(xa, *pa1)+(yga1+yga2)/2
                        vra=pa1[1]
                        e_vra=np.sqrt(np.diag(pa2))[1]
                        shga=True
                    except Exception:
                        shga=False
                        inta1=10
                        inta2=10
                        yga1=0
                        yga2=0
                    if k==0 or interac==True:
                        plt.close()
                        fig = plt.figure(figsize=[8,8])
                        dimy=0
                        plt.title('Cross-correlation for '+img+' (primary)')
                        plt.xlabel('Radial Velocity [km/s]')
                        plt.ylabel('Correlation')
                        plt.plot(drva, cca, color='blue')
                        if shga:
                            plt.plot(xa, ya, color='black',marker='.',linestyle='')
                            plt.plot(xa, yagauss, color='green', label='fit',linestyle='--')
                        else:
                            plt.legend(('No fit'))
                        plt.tight_layout()
                        fig.canvas.mpl_connect('button_press_event', onclick2)
                        fig.canvas.mpl_connect('key_press_event', on_key)
                        print('\t..........................................')
                        print('\t:        Press any key to refit          :')
                        print('\t:                                        :')
                        print('\t:     Press Y to save and continue       :')
                        print('\t:   Press N to process next spectrum     :') 
                        print('\t··········································')
                        plt.show()
                        if yours == 'y' or yours == 'Y':
                            fsave = True
                            print('\t\t\t\t\tDone!')
                            plt.close()
                            break
                        elif yours == 'n' or yours == 'N' or yours == 'q' or yours == 'Q':
                            fsave = False
                            plt.close()
                            break
                    else:
                        fsave = True
                        print('\t\t\t\t\tDone!')
                        break
                if fsave:
                    hdul[0].header['VRA'] = round(vra,4)
                    fout.write(str(jd)+'\t'+str(round(pa1[1],3))+'\t'+str(round(e_vra,3))+'\n')
                    print(img+', VRA: '+str(round(vra,2))+' km/s')
                    hdul.flush(output_verify='ignore')
                    hdul.close(output_verify='ignore')
            fout.close()
    if tb !=None:
        if baux1==False:
            print('Can not access to secondary template spectrum')
            print('END')
        else:
            for k,img in enumerate(larch):
                no_rv=0
                print('\t·············································')
                print('\tSet RV for '+img)
                wimg,fimg = pyasl.read1dFitsSpec(img)
                wtb, ftb = pyasl.read1dFitsSpec(tb+'.fits')
                hdul = fits.open(img, 'update')
                if k==0:
                    #gap expresa un margen en angstroms
                    gap=50
                    intb1=10
                    intb2=10
                    intb1=10
                    intb2=10
                    ygb1=0
                    ygb2=0
                    ygb1=0
                    ygb2=0
                    delta = hdul[0].header['CDELT1']
                    winf=max(wimg[0],wtb[0])+gap
                    wsup=min(wimg[-1],wtb[-1])-gap
                    delta = hdul[0].header['CDELT1']
                    new_disp_grid=np.arange(winf,wsup,delta)
                aux_img = Spectrum1D(flux=fimg*u.Jy, spectral_axis=wimg*0.1*u.nm)
                aux2_img = spline3(aux_img, new_disp_grid*0.1*u.nm)
                fnew=aux2_img.flux.value
                stat = 'm'
                initfitB=True
                while stat == 'm':
                    minvalB=True 
                    drvb, ccb = fxcor(new_disp_grid,fnew,wtb,ftb,rvmin=-rvrange,rvmax=rvrange,drv=drv)
                    if initfitB:
                        ibmax=np.argmax(ccb)
                        if ibmax<=intb1:
                            intb1=ibmax
                        elif ibmax>=(len(ccb)-intb2):
                            intb2=len(ccb)-ibmax
                    else:
                        bnear1=drvb.flat[np.abs(drvb - min(xgb1,xgb2)).argmin()]
                        bnear2=drvb.flat[np.abs(drvb - max(xgb1,xgb2)).argmin()]
                        bnew1=int(np.where(drvb == bnear1)[0])
                        bnew2=int(np.where(drvb == bnear2)[0])
                        xb=drvb[bnew1:bnew2]
                        yb=ccb[bnew1:bnew2]
                        ibmax=np.argmax(yb)+bnew1
                        intb1=ibmax-bnew1
                        intb2=bnew2-ibmax
                    try:
                        xb=drvb[ibmax-intb1:ibmax+intb2]
                        yb=ccb[ibmax-intb1:ibmax+intb2]
                        mb = np.sum(xb * (yb-(ygb1+ygb2)/2)) / np.sum(yb-(ygb1+ygb2)/2)
                        sigb = np.sqrt(np.abs(np.sum((yb-(ygb1+ygb2)/2) * (xb - mb)**2) / np.sum(yb-(ygb1+ygb2)/2)))
                        pb1,pb2 = curve_fit(Gauss, xb, yb-(ygb1+ygb2)/2, p0=[np.max((yb-(ygb1+ygb2)/2)), mb, sigb])
                        ybgauss=Gauss(xb, *pb1)+(ygb1+ygb2)/2
                        vrb=pb1[1]
                        #e_vrb=np.sqrt(np.diag(pb2))[1]
                        shgb=True
                    except Exception:
                        shgb=False
                        intb1=10
                        intb2=10
                        ygb1=0
                        ygb2=0
                    if k==0 or interac==True:
                        plt.close()
                        fig = plt.figure(figsize=[8,8])
                        fy1=str(fig).find('x')
                        fy2=len(str(fig))
                        dimy=int(str(fig)[fy1+1:fy2-1])/2
                        plt.title('Cross-correlation for '+img+' (secondary)')
                        plt.xlabel('Radial Velocity [km/s]')
                        plt.ylabel('Correlation')
                        plt.plot(drvb, ccb, color='red')
                        if shgb:
                            plt.plot(xb, yb, color='black',marker='.',linestyle='')
                            plt.plot(xb, ybgauss, color='green', label='fit',linestyle='--')
                        else:
                            plt.legend(('No fit'))
                        plt.tight_layout()
                        fig.canvas.mpl_connect('button_press_event', onclick2)
                        fig.canvas.mpl_connect('key_press_event', on_key)
                        print('\t..........................................')
                        print('\t:        Press any key to refit          :')
                        print('\t:                                        :')
                        print('\t:     Press Y to save and continue       :')
                        print('\t:   Press N to process next spectrum     :') 
                        print('\t··········································')
                        plt.show()
                        if yours == 'y' or yours == 'Y':
                            fsave = True
                            print('\t\t\t\t\tDone!')
                            plt.close()
                            break
                        elif yours == 'n' or yours == 'N' or yours == 'q' or yours == 'Q':
                            fsave = False
                            plt.close()
                            break
                    else:
                        fsave = True
                        print('\t\t\t\t\tDone!')
                        break
                if fsave:
                    hdul[0].header['VRB'] = round(vrb,2)
                    print(img+', VRB: '+str(round(vrb,1))+' km/s')
                    hdul.flush(output_verify='ignore')
                    hdul.close(output_verify='ignore')
################################################################
################################################################
################################################################
def spbina(lis, spa='A', spb='B', nit=5, frat=0.8, reject=True,q=None,vgamma=None,obspha=False,showtit=True):
    if showtit:
        print('')
        print('\t  Running SPBINA')
        print('')
    spline3 = SplineInterpolatedResampler()
    VerifyWarning('ignore')
    larch=makelist(lis)
    nimg=len(larch)
    haux = fits.open(larch[0], 'update')
    delta = haux[0].header['CDELT1']
    xwmin=[]
    xwmax=[]
    for i in range(nimg):
        hx = fits.open(larch[i], 'update')
        xdel = hx[0].header['CDELT1']
        xw0 = hx[0].header['CRVAL1']
        xn = hx[0].header['NAXIS1']
        xw1=xw0+xdel*(xn-1)
        xwmin.append(xw0)
        xwmax.append(xw1)
    new_disp_grid=np.arange(np.max(xwmin),np.min(xwmax),delta)
    nwvl = len(new_disp_grid)
    baux1 = os.path.isfile(spb+'.fits')
#STEP 1: create B.fits
    if baux1==False:
        fa = 1.0/(1.0 + frat)
        fb = frat*fa
        tmp_matrix = np.zeros(shape=(nimg,nwvl))
        cont=0
        for img in larch:
            wimg,fimg = pyasl.read1dFitsSpec(img)
            aux_img = Spectrum1D(flux=fimg*u.Jy, spectral_axis=wimg*0.1*u.nm)
            aux2_img = spline3(aux_img, new_disp_grid*0.1*u.nm)
            tmp = aux2_img.flux.value
            tmp_matrix[cont] = splineclean(tmp)
            cont+=1
        f_mean = np.zeros(nwvl)
        for i in range(nwvl):
            f_mean[i] = np.mean(tmp_matrix[:,i])
        f_cont = continuum(new_disp_grid, f_mean,type='fit',graph=False)
        f_cont=splineclean(f_cont)
        B = f_cont* fb
    else:
        waux, faux = pyasl.read1dFitsSpec(spb+'.fits')
        aux_B = Spectrum1D(flux=faux*u.Jy, spectral_axis=waux*0.1*u.nm)
        aux2_B = spline3(aux_B, new_disp_grid*0.1*u.nm)
        tmp2 = aux2_B.flux.value
        B = splineclean(tmp2)
#STEP 2: obs - B.fits
    obs_matrix = np.zeros(shape=(nimg,nwvl))
    dsA_matrix = np.zeros(shape=(nimg,nwvl))
    dsB_matrix = np.zeros(shape=(nimg,nwvl))
    za_matrix = np.zeros(shape=(nimg,nwvl))
    zb_matrix = np.zeros(shape=(nimg,nwvl))
    cont=0
    vra_array = np.zeros(nimg)
    vrb_array = np.zeros(nimg)
    for img in larch:
        hdul = fits.open(img, 'update')
        vra = hdul[0].header['VRA']
        hdul.close(output_verify='ignore')
        if q==None:
            vrb = hdul[0].header['VRB']
        elif q>0:
            vrb = vgamma - (vra-vgamma)/q
        vra_array[cont]=vra
        vrb_array[cont]=vrb
        wimg,fimg = pyasl.read1dFitsSpec(img)
        #doppler correction for B.fits
        wlprime_B = new_disp_grid * np.sqrt((1.+vrb/299792.458)/(1.-vrb/299792.458))
        aux_sb = Spectrum1D(flux=B*u.Jy, spectral_axis=wlprime_B *0.1*u.nm)
        aux2_sb = spline3(aux_sb, new_disp_grid*0.1*u.nm)
        fb_dop = aux2_sb.flux.value
        #Replace np.nan values for the nearest element
        aux_img = Spectrum1D(flux=fimg*u.Jy, spectral_axis=wimg*0.1*u.nm)
        aux2_img = spline3(aux_img, new_disp_grid*0.1*u.nm)
        dsB = aux2_img.flux.value - fb_dop
        dsB_matrix[cont] = splineclean(dsB)
        obs_matrix[cont] = splineclean(aux2_img.flux.value)
        cont+=1
        wei=np.mean(obs_matrix,axis=1)/np.max(np.mean(obs_matrix,axis=1))
#STEP 3: calculate A.fits
    for i in range(nit):
        if showtit==True and i==0:
            bar3 = ChargingBar('Calculating spectra:', max=nit)
        for j in range(nimg):
            wlprime_A = new_disp_grid * np.sqrt((1.-vra_array[j]/299792.458)/(1.+vra_array[j]/299792.458))
            aux_sa = Spectrum1D(flux=dsB_matrix[j] *u.Jy, spectral_axis=wlprime_A *0.1*u.nm)
            aux2_sa = spline3(aux_sa, new_disp_grid*0.1*u.nm)
            fa_dop = aux2_sa.flux.value
            #Replace np.nan values for the nearest element
            za_matrix[j] = splineclean(fa_dop)
        if reject:
            A = sfcomb(za_matrix,wei)
        else:
            A = np.average(za_matrix,axis=0,weights=wei)
#STEP 4: obs - A.fits
#STEP 5: calculate B.fits
        for j in range(nimg):
            wlprime_A = new_disp_grid * np.sqrt((1.+vra_array[j]/299792.458)/(1.-vra_array[j]/299792.458))
            aux_sa = Spectrum1D(flux=A*u.Jy, spectral_axis=wlprime_A *0.1*u.nm)
            aux2_sa = spline3(aux_sa, new_disp_grid*0.1*u.nm)
            fa_dop = aux2_sa.flux.value
            fa_dop =  splineclean(fa_dop)
            dsA_matrix[j] = obs_matrix[j] - fa_dop
            wlprime_B = new_disp_grid * np.sqrt((1.-vrb_array[j]/299792.458)/(1.+vrb_array[j]/299792.458))
            aux_sb = Spectrum1D(flux=dsA_matrix[j]*u.Jy, spectral_axis=wlprime_B*0.1*u.nm)
            aux2_sb = spline3(aux_sb, new_disp_grid*0.1*u.nm)
            fb_dop = aux2_sb.flux.value
            zb_matrix[j] = splineclean(fb_dop)
        if reject:
            B = sfcomb(zb_matrix,wei)
        else:
            B = np.average(zb_matrix,axis=0,weights=wei)
#STEP 6: obs - B.fits
        for j in range(nimg):
            wlprime_B = new_disp_grid * np.sqrt((1.+vrb_array[j]/299792.458)/(1.-vrb_array[j]/299792.458))
            aux_sb = Spectrum1D(flux=B*u.Jy, spectral_axis=wlprime_B *0.1*u.nm)
            aux2_sb = spline3(aux_sb, new_disp_grid*0.1*u.nm)
            fb_dop = aux2_sb.flux.value
            dsB_matrix[j] = obs_matrix[j] - splineclean(fb_dop)
        if showtit:
            bar3.next()
    if showtit:
        bar3.finish()
    pyasl.write1dFitsSpec(spa+'.fits', A, wvl=new_disp_grid, clobber=True)
    pyasl.write1dFitsSpec(spb+'.fits', B, wvl=new_disp_grid, clobber=True)
    if obspha:
        for i,img in enumerate(larch):
            pyasl.write1dFitsSpec('ds-B_'+img, dsB_matrix[i], wvl=new_disp_grid, clobber=True)
            copyheader(img,'ds-B_'+img)
            pyasl.write1dFitsSpec('ds-A_'+img, dsA_matrix[i], wvl=new_disp_grid, clobber=True)
            copyheader(img,'ds-A_'+img)
################################################################
################################################################
################################################################
def splot(file,xmin='INDEF',xmax='INDEF',ymin='INDEF',ymax='INDEF', scale= 1., markpix=False, newfig=True, color='r'):
    plt.ion()
    w,f = pyasl.read1dFitsSpec(file)
    if newfig:
        plt.figure(figsize=[20,10])
    if xmin=='INDEF':
        x1=np.min(w)
    else:
        x1=xmin
    if xmax=='INDEF':
        x2=np.max(w)
    else:
        x2=xmax
    if ymin=='INDEF':
        y1=np.min(f)*scale
    else:
        y1=ymin
    if ymax=='INDEF':
        y2=np.max(f)*scale
    else:
        y2=ymax*scale
    plt.axis([x1,x2,y1,y2])  
    plt.ylabel('Flux')
    plt.xlabel('Wavelength')
    plt.title(file)
    plt.plot(w,f*scale,marker='',color=color,linewidth=1)
    if markpix:
        plt.plot(w,f*scale,marker='.',markersize=2,color='black',linestyle='')
    plt.tight_layout()
################################################################
################################################################
################################################################
def vgrid(lis, lit, svmin=-1, svmax=1, step=0.1, qmin=0.02, qmax=0.5, deltaq=0.01, wreg='4000-4090,4110-4320,4360-4850,4875-5290,5350-5900',nproc=6):
    if os.path.isdir('CC')==False:
        os.mkdir('CC')
    instanteInicial = datetime.now()
    svrange=np.arange(svmin,svmax+step,step) 
    plt.ion()
    print('\n\t  Running VGRID (v0.1)\n')
    VerifyWarning('ignore')
    larch=makelist(lis)
    filet=open(lit,'r')
    ltemp=[]
    for tx in filet:
        ltemp.append(tx.rstrip('\n'))
    filet.close()
    q_array=np.arange(round(qmin,7),round(qmax+deltaq,7),round(deltaq,7))
    #Create fading mask
    dlist=[]
    hobs = fits.open(larch[0], 'update')
    d1 = hobs[0].header['CDELT1']
    dlist.append(d1)
    try:
        obj1 = hobs[0].header['OBJECT']
        obj1=obj1.replace(' ','')
    except KeyError:
        obj1='NoObject'
    hobs.close(output_verify='ignore')
    htmp = fits.open(ltemp[0], 'update')
    d2 = htmp[0].header['CDELT1']
    dlist.append(d2)
    htmp.close(output_verify='ignore')
    waux1,faux1 = pyasl.read1dFitsSpec(larch[0])
    waux2,faux2 = pyasl.read1dFitsSpec(ltemp[0])
    gap=50
    winf=max(waux1[0],waux2[0])+gap
    wsup=min(waux1[-1],waux2[-1])-gap
    new_disp_grid,fmask = setregion(wreg,np.max(dlist),winf,wsup)
    #Save q_array in a file
    qout = open(obj1+'_q.txt', 'w')
    for j,xq in enumerate(q_array):
        qout.write(str(round(xq,len(str(deltaq+1))-2))+'\n')
    qout.close()   
    #Save xvg in a file
    vgout = open(obj1+'_vg.txt', 'w')
    if math.modf(step)[0] == 0:
        nrd=0
    else:
        nrd=len(str(step))-str(step).find('.')-1
    for j,xvg in enumerate(svrange):
        if nrd == 0:
            vgout.write(str(int(xvg))+'\n')
        else:
            vgout.write(str(round(xvg,nrd))+'\n')
    vgout.close()
    #Load templates and create array for temperatures
    vector_t=np.zeros(len(ltemp))
    tt_array=np.zeros(len(ltemp))
    matrix_tmp=np.zeros(shape=(len(ltemp),len(new_disp_grid)))
    tout = open(obj1+'_Teff.txt', 'w')
    bar2 = ChargingBar('Loading templates:', max=len(ltemp))
    for k,tmp in enumerate(ltemp):
        htmp = fits.open(tmp, 'update')
        teff= htmp[0].header['TEFF']
        vector_t[k]=teff
        htmp.close(output_verify='ignore')
        wt1,ft1 = pyasl.read1dFitsSpec(tmp)
        temp_cont = continuum(wt1, ft1, type='diff', lo=2.5,hi=6, graph=False)
        aux_tmp1 = Spectrum1D(flux=temp_cont*u.Jy, spectral_axis=wt1*0.1*u.nm)
        aux2_tmp1 = spline3(aux_tmp1, new_disp_grid*0.1*u.nm)
        template1=splineclean(aux2_tmp1.flux.value*fmask)
        matrix_tmp[k]=template1
        tt_array[k]=np.mean(template1**2)
        tout.write(str(int(teff))+'\n')
        bar2.next()
    bar2.finish()
    print('')
    #Execute FIND2C analysis for vgamma grid
    bar0 = ChargingBar('Calculating syst. rv:', max=len(svrange))
    for vgamma in svrange:
        vgindex=str(round(vgamma,nrd))
        check1=os.path.isfile('CC/'+obj1+'_CC_'+'vg_'+vgindex+'.txt')
        if check1 ==False:
            for xq in q_array:
                aux1=str(round(xq,len(str(deltaq+1))-2)).replace('.','')
                aux2 = os.path.isfile('B'+aux1+'.fits')
                if aux2:
                    os.remove('B'+aux1+'.fits')
                aux3 = os.path.isfile('A'+aux1+'.fits')
                if aux3:
                    os.remove('A'+aux1+'.fits') 
            pool=Pool(processes=nproc)
            qa2=np.array_split(q_array,nproc)
            pres=[pool.apply_async(qparallel, args= [chk,lis,larch,'A','B',deltaq,vgamma]) for chk in qa2]
            pool.close()
            #pool.join()
            pres = [chk.get() for chk in pres]
            pool.terminate()
            matrix_sq=np.zeros(shape=(len(q_array),len(new_disp_grid)))
            #Load calculated B_q spectra
            for j,xq in enumerate(q_array):
                aux1=str(round(xq,len(str(deltaq+1))-2)).replace('.','')
                wimg,fimg = pyasl.read1dFitsSpec('B'+aux1+'.fits')
                spec_cont=continuum(wimg, fimg, type='diff',lo=2.5,hi=3.5, graph=False)
                aux_img = Spectrum1D(flux=spec_cont*u.Jy, spectral_axis=wimg*0.1*u.nm)
                aux2_img = spline3(aux_img, new_disp_grid*0.1*u.nm)
                matrix_sq[j] = splineclean(aux2_img.flux.value)*fmask
            #Load calculated tt values from templates
            ccout = open('CC/'+obj1+'_CC_'+'vg_'+vgindex+'.txt', 'w')
            for k,tt in enumerate(tt_array):
                for l,xq in enumerate(q_array):
                    bb=np.mean(matrix_sq[l]**2)
                    tb=np.mean(matrix_tmp[k]*matrix_sq[l])
                    cc=tb/(np.sqrt(bb)*np.sqrt(tt))
                    ccout.write(str(round(cc,4))+'\t')
                ccout.write('\n')
            ccout.close()
        bar0.next()
        #Clean A.fits and B.fits files
        for xq in q_array:
            aux1=str(round(xq,len(str(deltaq+1))-2)).replace('.','')
            aux2 = os.path.isfile('B'+aux1+'.fits')
            if aux2:
                os.remove('B'+aux1+'.fits')
            aux3 = os.path.isfile('A'+aux1+'.fits')
            if aux3:
                os.remove('A'+aux1+'.fits') 
    bar0.finish()
    instanteFinal = datetime.now()
    tiempo = instanteFinal - instanteInicial
    print('\nTotal Time Processing:',tiempo.seconds,'s')
################################################################
################################################################
################################################################
def vexplore(obj):
    plt.ioff()
    aaux1=os.path.isfile(obj+'_Teff.txt')
    aaux2=os.path.isfile(obj+'_q.txt') 
    aaux3=os.path.isfile(obj+'_vg.txt') 
#read file for temperature effective array
    if aaux1:
        t_array=np.loadtxt(obj+'_Teff.txt')
    else:
        print('Can not access to Teff file')
        print('END')
#read file for mass ratio array
    if aaux2:
        q_array=np.loadtxt(obj+'_q.txt')
    else:
        print('Can not access to q file')
        print('END')
#read file for systemic velocity array
    if aaux3:
        vg_array=np.loadtxt(obj+'_vg.txt')
    else:
        print('Can not access to vgamma file')
        print('END')
    cc_list=os.listdir('CC/')
    if len(cc_list) == 0:
        print('CC/ folder is empty')
        print('END')
        aaux4=False
    else:
        aaux4=True
#read file for cross correlation function
    if aaux1 and aaux2 and aaux3 and aaux4:
        cc_wsv=np.zeros(shape=(len(vg_array),len(t_array),len(q_array)))
        lsup=[]
        linf=[]
        imed=int(len(vg_array)/2)
        step=round(vg_array[imed+1]-vg_array[imed],8)
        if math.modf(step)[0] == 0:
            nrd=0
        else:
            nrd=len(str(step))-str(step).find('.')-1
        for i,vgamma in enumerate(vg_array):
            if nrd == 0:
                vgindex=str(int(vgamma))
            else:
                vgindex=str(round(vgamma,nrd))
            cc_wsv[i] = np.loadtxt('CC/'+obj+'_CC_'+'vg_'+vgindex+'.txt')
            lsup.append(cc_wsv[i])
            linf.append(cc_wsv[i])
#3D surface graphic
        fig=plt.figure(figsize=[8,6])
        ax = Axes3D(fig)
        ax.set_autoscalez_on(False)
        ax.set_zlim3d(bottom=np.min(linf), top=np.max(lsup))
        ax.set_xlabel("Mass ratio ($q$)", fontsize=10)
        ax.set_ylabel("Temp. [1000 K]", fontsize=10)
        ax.set_title('vgamma = '+str(round(vg_array[0],len(str(step+1))-2))+' km/s',y=1)
        X, Y = np.meshgrid(q_array, t_array/1000)
        ax.plot_surface(X, Y, cc_wsv[0], rstride=1, cstride=1, cmap='jet',vmin=np.min(linf), vmax=np.max(lsup))
        class Index:
            ind = 0
            def next(self, event):
                self.ind += 1
                i=self.ind  % len(vg_array)
                ax.cla()
                ax.set_zlim3d(bottom=np.min(linf), top=np.max(lsup))
                ax.plot_surface(X, Y, cc_wsv[i], rstride=1, cstride=1, cmap='jet',vmin=np.min(linf), vmax=np.max(lsup))
                ax.set_title('vgamma = '+str(round(vg_array[i],len(str(step+1))-2))+' km/s',y=1)
                ax.set_xlabel("Mass ratio ($q$)", fontsize=10)
                ax.set_ylabel("Temp. [1000 K]", fontsize=10)
                plt.draw()
        
            def prev(self, event):
                self.ind -= 1 % len(vg_array)
                i=self.ind % len(vg_array)
                ax.cla()
                ax.set_zlim3d(bottom=np.min(linf), top=np.max(lsup))
                ax.plot_surface(X, Y, cc_wsv[i], rstride=1, cstride=1, cmap='jet',vmin=np.min(linf), vmax=np.max(lsup))
                ax.set_title('vgamma = '+str(round(vg_array[i],len(str(step+1))-2))+' km/s',y=1)
                ax.set_xlabel("Mass ratio ($q$)", fontsize=10)
                ax.set_ylabel("Temperature [x1000 K]", fontsize=10)
                plt.draw()
        callback = Index()
        axprev = plt.axes([0.05, 0.88, 0.1, 0.075])
        axnext = plt.axes([0.85, 0.88, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(callback.next)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(callback.prev)
        plt.show()
################################################################
################################################################
################################################################
# INTERNAL FUNCTIONS
#Parallel process
def qparallel(q_array,lis,larch,spa,spb,deltaq,vgamma):
    for xq in q_array:
        aux1=str(round(xq,len(str(deltaq+1))-2)).replace('.','')
        aux2 = os.path.isfile(spb+aux1+'.fits')
        if aux2==False:
            spbina(lis, spa=spa+aux1, spb=spb+aux1, nit=1,q=xq,vg=vgamma,showtit=False)

def onclick1(event):
    global vra_aux,vrb_aux,no_rv
    if event.button==1 and event.inaxes is not None:
        if no_rv==0:
            no_rv+=1
            vra_aux=event.xdata
            print('RV for primary component: '+str(int(event.xdata))+' km/s')
        elif no_rv==1:
            no_rv+=1
            vrb_aux=event.xdata
            print('RV for secondary component: '+str(int(event.xdata))+' km/s')

def onclick2(event):
    global rva1,rva2,rvb1,rvb2,minvalA,minvalB,sgaussA1,cgaussA,cgaussB,xga1,xga2,xgb1,xgb2,initfitA,initfitB,yga1,yga2,ygb1,ygb2
    if event.button==3 and event.inaxes is not None:
        if event.y > dimy and minvalA==True:
            initfitA=True
            rva1=event.xdata
            minvalA=False
            event.inaxes.axvline(x=rva1, color='black', linestyle='-')
            print('Set lower limit for A: '+str(int(rva1))+' km/s')
        elif event.y > dimy and minvalA==False:
            initfitA=True
            rva2=event.xdata
            minvalA=True
            event.inaxes.axvline(x=rva2, color='black', linestyle='-')
            print('Set upper limit for A: '+str(int(rva2))+' km/s')
        elif event.y < dimy and minvalB==True:
            initfitB=True
            rvb1=event.xdata
            minvalB=False
            event.inaxes.axvline(x=rvb1, color='black', linestyle='-')
            print('Set lower limit for B: '+str(int(rvb1))+' km/s')
        elif event.y < dimy and minvalB==False:
            initfitB=True
            rvb2=event.xdata
            minvalB=True
            event.inaxes.axvline(x=rvb2, color='black', linestyle='-')
            print('Set upper limit for B: '+str(int(rvb2))+' km/s')
    if event.button==1 and event.inaxes is not None:
        if event.y > dimy and cgaussA==0:
            cgaussA=1
            xga1=event.xdata
            yga1=event.ydata
            event.inaxes.axvline(x=xga1, color='green', linestyle='--')
            print('Minimum gaussian fit for A: '+str(int(event.xdata))+' km/s')
        elif event.y > dimy and cgaussA==1:
            cgaussA=0
            xga2=event.xdata
            yga2=event.ydata
            initfitA=False
            event.inaxes.axvline(x=xga2, color='green', linestyle='--')
            print('Maximum gaussian fit for A: '+str(int(event.xdata))+' km/s')
        elif event.y < dimy and cgaussB==0:
            cgaussB=1
            xgb1=event.xdata
            ygb1=event.ydata
            event.inaxes.axvline(x=xgb1, color='green', linestyle='--')
            print('Minimum gaussian fit for B: '+str(int(event.xdata))+' km/s')
        elif event.y < dimy and cgaussB==1:
            cgaussB=0
            xgb2=event.xdata
            ygb2=event.ydata
            initfitB=False
            event.inaxes.axvline(x=xgb2, color='green', linestyle='--')
            print('Maximum gaussian fit for B: '+str(int(event.xdata))+' km/s')
    event.canvas.draw()

def on_key(event):
    global yours
    print('You pressed: ', event.key)
    yours=event.key
    plt.close()

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def makelist(lis):
    aux1=lis.find('@')
    if aux1 == -1:
        path=os.getcwd()
        aux2=glob.glob(path+'/'+lis)
        newlist=[]
        for i in range(len(aux2)):
            newlist.append(aux2[i].replace(path+'/',''))
    else:
        aux2=lis.replace('@','')
        f=open(aux2,'r')
        aux3=f.readlines()
        f.close()
        newlist=[]
        for i in range(len(aux3)):
            newlist.append(aux3[i].rstrip('\n'))
    return(newlist)

@jit(nopython=True)
def sfcomb(aflux,wei,nit=5,sigma=3):
    nwvl=aflux.shape[1]
    s_mean = []
    for w in range(nwvl):
        fdata = aflux[:,w]/wei
        weights=wei
        for i in range(nit):
            fmean = np.sum(fdata*weights)/np.sum(weights)
            fsig=np.sqrt(np.sum((fdata-fmean)**2*weights)/np.sum(weights))
            faux=fdata
            cont=0
            for j,fj in enumerate(faux):
                if fj < (fmean-sigma*fsig) or fj > (fmean+sigma*fsig):
                    fdata=np.delete(fdata, j-cont)
                    weights=np.delete(weights,j-cont)
                    cont+=1
        fmean = np.sum(fdata*weights)/np.sum(weights)
        fsig=np.sqrt(np.sum((fdata-fmean)**2)/np.sum(weights))
        s_mean.append(fmean)
    return(np.array(s_mean))
            
def setregion(wreg,delta,winf,wsup,amort=0.1):
    reg1=wreg.split(',')
    reg2=[]
    for i,str1 in enumerate(reg1):
        reg2.append([int(str1.split('-')[0]),int(str1.split('-')[1])])
    reg3=[]
    stat1=True
    for j,wvx in enumerate(reg2):
        x1=wvx[0]
        x2=wvx[1]
        if stat1:
            if x1>=winf:
                reg3.append(wvx)
                stat1=False
            elif x1<winf and x2>winf:
                wvx[0]=winf
                reg3.append(wvx)
                stat1=False
            elif x1<winf and x2<=winf:
                stat1=True
        else:
            if x1>wsup and x2>wsup:
                break
            elif x1<wsup and x2>=wsup:
                wvx[1]=wsup
                reg3.append(wvx)
            elif x1<wsup and x2<wsup:
                reg3.append(wvx)
    wvl=np.arange(reg3[0][0],reg3[-1][1]+delta,delta)
    f=np.zeros(len(wvl))
    for k,interv in enumerate(reg3):
        x1=interv[0]
        x2=interv[1]
        i1 = np.abs(wvl - x1).argmin(0)
        i2 = np.abs(wvl - x2).argmin(0)
        am2=amort*(x2-x1)
        xarr=wvl[i1:i2]
        mask=np.zeros(len(xarr))
        for k,w in enumerate(xarr):
            if w<=(x1+am2):
                mask[k]=np.sin(np.pi*(w-x1)/(2*am2))
            elif w>(x1+am2) and w<(x2-am2):
                mask[k]=1
            else:
                mask[k]=np.cos(np.pi*(w-x2+am2)/(2*am2))
        f[i1:i2]=mask
    return(wvl,f)

def continuum(w,f, order=12, type='fit', lo=2, hi=3, nit=20, graph=True):
    w_cont=w.copy()
    f_cont=f.copy()
    sigma0=np.std(f_cont)
    wei=~np.isnan(f_cont)*1
    i=1
    nrej1=0
    while i < nit:
        c0=np.polynomial.chebyshev.Chebyshev.fit(w_cont,f_cont,order,w=wei)(w_cont)
        resid=f_cont-c0
        sigma0=np.sqrt(np.average((resid)**2, weights=wei))
        wei = 1*np.logical_and(resid>-lo*sigma0,resid<sigma0*hi)
        nrej=len(wei)-np.sum(wei)
        if nrej==nrej1:
            break
        nrej1=nrej
        i=i+1
    s1=Spectrum1D(flux=c0*u.Jy, spectral_axis=w_cont*0.1*u.nm) 
    c1= fit_continuum(s1, model=Chebyshev1D(order),fitter=LinearLSQFitter())
    if type=='fit':
        fout=c1(w*0.1*u.nm).value
    elif type=='ratio':
        fout=f_cont/c1(w*0.1*u.nm).value
    elif type=='diff':
        fout=f_cont-c1(w*0.1*u.nm).value
    if graph:
        fig = plt.figure(figsize=[20,10])
        ngrid = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[5, 1])
        ax1 = fig.add_subplot(ngrid[0])
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.ylabel('Flux')
        ax1.plot(w,f,color='gray')
        ax1.plot(w_cont,f_cont,color='blue',linestyle='',marker='.',markersize=2)
        ax1.plot(w_cont,c1(w_cont*0.1*u.nm).value,c='r',linestyle='--')
        ax2 = fig.add_subplot(ngrid[1], sharex=ax1,sharey=ax1)
        plt.xlabel('Wavelength [nm]')
        ax2.plot(w,(f-c1(w*0.1*u.nm).value),color='gray',linestyle='',marker='.',markersize=1)
        ax2.plot(w_cont,(f_cont-c1(w_cont*0.1*u.nm).value),color='blue',linestyle='',marker='.',markersize=1)
        ax2.axhline(y=0, color='red', linestyle='--',linewidth=1)
        plt.tight_layout()
        plt.show()
    return(fout)
    
def fxcor(w, f, wt, ft, rvmin, rvmax, drv):
    f1=continuum(w=w, f=f, type='diff', graph=False)
    ft1=continuum(w=wt, f=ft, type='diff', graph=False)
    drvs = np.arange(rvmin, rvmax, drv)
    cc = np.zeros(len(drvs))
    for i, xrv in enumerate(drvs):
        flux_i  = sci.interp1d(wt*np.sqrt((1.+xrv/299792.458)/(1.-xrv/299792.458)), ft1)
        cc[i] = np.sum(f1 * flux_i(w))
    cc = cc / (np.sqrt(np.sum(np.power(f1,2)))*(np.sqrt(np.sum(np.power(ft1,2)))))
    return drvs, cc

@jit(nopython=True)
def splineclean(fspl):
    if np.isnan(fspl[0]):
        cinit=1
        while np.isnan(fspl[cinit]):
            cinit+=1
        finit=fspl[cinit]
        fspl[0:cinit+1]=finit
    if np.isnan(fspl[-1]):
        cend=-2
        while np.isnan(fspl[cend]):
            cend-=1
        fend=fspl[cend]
        fspl[cend:len(fspl)]=fend
    return(fspl)

def copyheader(img1,imgout):
    hdul = fits.open(img1, 'update')
    hnorm =  fits.open(imgout, 'update')
    listk=('CDELT1','CTYPE1','BUNIT','ORIGIN','DATE','TELESCOP','INSTRUME',
           'OBJECT','RA','DEC','EQUINOX','RADECSYS','EXPTIME','MJD-OBS','DATE-OBS','UTC','LST',
           'PI-COI','CTYPE1','CTYPE2','ORIGFILE','UT','ST','AIRMASS','VRA','VRB')
    for i,k in enumerate(listk):
        try:
            hnorm[0].header[k] = hdul[0].header[k]
        except KeyError:
            print('Keyword '+k+' not found')
    hnorm.flush(output_verify='ignore')
    hnorm.close(output_verify='ignore')
    hdul.close(output_verify='ignore')

def boxcar(lis,sigma):
    VerifyWarning('ignore')
    larch=makelist(lis)
    for img in larch:
        hdul = fits.open(img, 'update')
        hdul[0].header.remove('comment')
        hdul[0].header.remove('comment')
        wimg,fimg = pyasl.read1dFitsSpec(img)
        aux_img = Spectrum1D(flux=fimg*u.Jy, spectral_axis=wimg*0.1*u.nm)
        newsp=gaussian_smooth(aux_img,sigma)
        pyasl.write1dFitsSpec('gfil'+img,newsp.flux.value, newsp.wavelength.value, header=hdul[0].header, clobber=True)
        hdul.close()

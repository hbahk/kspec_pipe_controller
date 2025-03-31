#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Hyeonguk Bahk (bahkhyeonguk@gmail.com)
# @Date: 2025-03-31
# @Filename: kspec_make_tlm.py
# @Description:
# This file contains `_make_tlm.py` of `twodfdr` package (@88c4e7f9),with small
# adjustments to make it work with the future KSPEC instrument.

import cpl
from twodfdr import _twodfdr, config, args
from twodfdr.io import Tdfio, TdfioMode, TdfioType, TdfioInstType, TdfioKeyType
from twodfdr.modules.arc import Arc
from twodfdr.modules.tlm import TLM 
from twodfdr.modules.utils import Utils
from twodfdr.modules.sigma_profile import SigmaProfile
from twodfdr.modules.external_psf import ExternalPsf 

from pathlib import Path

from enum import Enum, IntEnum

from numpy import array
import numpy as np
import re

from kspec_inst import KspecTdfioInstType

def kspec_make_tlm(params,image,ofname=None,odir=None):
    """
    Generate a tramline map from an image file from specifications entirely given in the args.
    This routine implements the make_tlm and make_tlm_from_im functionality entirely from Python.

    Parameters
    ----------
    params : cpl.ui.ParameterList
        The method arguments.
    image : str
        The filename of the image frame for which to generate tramline map. 
    ofname : str, optional
        The output tramline map file name. This file is created, filled with tramline map, wavelength and optional sigma profile, and then closed by this routine. The default value of `ofname` is to add the suffix 'tlm.fits'.
    odir: str, optional
        The directory to write the output files.

    Notes
    -----
    This can be considered a template for extending or modifying make_tlm from Python.

    Returns
    -------
    str
        To be determined
    """
    status = 0
    im_im = Tdfio.suffix(image,"im") 
    im_ex = Tdfio.suffix(image,"ex") 
    im_red = Tdfio.suffix(image,"red") 
    tlm = Tdfio.suffix(image,"tlm")

    # Add parameters necessary to run the recipe 
    params.append(cpl.ui.ParameterValue(name="OBJECT_FILENAME", context="",default=image,description="Object filename."))
    params.append(cpl.ui.ParameterValue(name="IMAGE_FILENAME", context="",default=im_im,description="Image filename."))
    params.append(cpl.ui.ParameterValue(name="EXTRAC_FILENAME", context="",default=im_ex,description="Extract filename."))
    params.append(cpl.ui.ParameterValue(name="OUTPUT_FILENAME", context="",default=im_red,description="Output filename."))

    config.TdfConfig.oval(params,"TLMAP_FILENAME",ofname,tlm,"Tramline map filename.")
    config.TdfConfig.oval(params,"OUTDIR_NAME",odir,"NONE","Output directory.")

    ta = args.TdfArgs(params=params)

    recipe_name = "kspec_make_tlm"

    # --Start reduce_fflat.F95 section--

    # This section does the work that is ordinarily done in reduce_fflat
    # since 2dfdr doesn't call make_tlm directly, it calls reduce_fflat with
    # MAKE_TLM_ONLY == 1 and DO_REDFL == 0   

    # Call up any environmental initialisations defined in the parameters
    Utils.init_from_args(ta)

    # If an output directory is specified in the arguments then set this directory.
    out_dirname = ta.getc("OUT_DIRNAME","NONE")
    Utils.outfile_setdir(out_dirname)
    cpl.core.Msg.info(recipe_name,"OUT_DIRNAME={}".format(out_dirname.rstrip()))

    # Produce im(age) frame if a tramline map is requested
    # TODO: make these robust separate Python functions ....
    _twodfdr.make_im__(args=ta.sid,status=status)

    # produce ex(tracted) frame
    # TODO: make these robust separate Python functions ....
    # NOT NEEDED
    # _twodfdr.make_ex(args=ta.sid,status=status)

    # --End reduce_fflat.F95 section--
    
    # Get name of the input image frame
    im_fname = ta.getc("IMAGE_FILENAME")


    # Get the name of the tramline map filename
    tlm_fname = TLM.get_tfname(ta)

    cpl.core.Msg.info(recipe_name,"Generating new tramline map from image. . .")

    # Open image file for read only
    timg = Tdfio.open(fname=im_fname,fmode=TdfioMode.R,ftype=TdfioType.STD)

    # We implement the function using only functions with Python bindings
    # Here, we are doing things the long way, following MAKE_TLM_FROM_IM in make_tlm.F95
    # cpl.core.Msg.info(recipe_name,"Generating TWODFDR tramline map")

    # Get code for instrument that created this file
    instcode = timg.instrument

    # Create empty tramline map file
    tram = Tdfio.open(fname=tlm_fname,fmode=TdfioMode.W,ftype=TdfioType.TLM)

    # Call routine appropriate for instrument to create TLM primary & WAVELA 
    # All routines write tramline map into Primary HDU and predicted wavelength
    # array into 'WAVELA' HDU of TLM file TLM_ID
    if instcode == TdfioInstType.GENERIC or instcode == TdfioInstType.GENERIC_IR:
        TLM.generic(timg,tram,ta)
    else:
        # We call the old TLM.twodf solely for the predicted
        # wavelength model it generates. For the tramline data itself 
        # we use the TLM.other method       
        if instcode == TdfioInstType.TWODF:
            TLM.twodf(timg,tram)

        TLM.other(timg,tram,instcode,ta)

    # Read tramline data into array TLM
    # npix, nfib = tram.get_size
    tlma = tram.tlmap_read()

    # If flat and create sigma profile option requested
    # then create sigma profile and add to tramline map FITS file
    # The ndf_class here is read from the timg
    # (in Fortran this happens in MAKE_TLM_OTHER (TLM.other above)
    # to populate the CLASS variable)
    if ta.getl('SIGPROF',False) and timg.ndf_class[:5] == 'MFFFF':
        cpl.core.Msg.info(recipe_name,"Extracting Sigma Profile...")
        # First get image data into array
        im_a = timg.read()
        
        # Get which PSF profile to use
        psf_type = ta.getc('PSF_TYPE','GAUSS') 
        ExternalPsf.set_type(psf_type)

        if psf_type == 'C2G2L':
            status = 0
            ExternalPsf.parse(ta,'MFFFF',status)
        
        # Extract sigma profile
        spa = SigmaProfile.sigma_profile(im_a,tlma,5,200,False)

        # Write sigma profile array to tramline map as new HDU
        tram.tsig_write(spa)

    # Copy 'UTDATE' and 'UTSTART' key headers from image into tramline map
    (utdate,dcomment) = timg.key_read('UTDATE',TdfioKeyType.STR)
    (utstart,scomment) = timg.key_read('UTSTART',TdfioKeyType.STR)
    tram.key_write('UTDATE',TdfioKeyType.STR,utdate,dcomment)
    tram.key_write('UTSTART',TdfioKeyType.STR,utstart,scomment)

    # Write tramline map data into TLM file
    tram.tlmap_write(tlma)

    # Set a "REDUCED" status for any external tool to query
    tram.setred()

    # Write the 2dfdr release version as a keyword for any
    # external tool to query.
    tram.stamp_version()

    # Now close it
    tram.close()

    # Close the image frame
    timg.close()

    # Close the args
    ta.close()
    
    # Report tramline map creation complete
    cpl.core.Msg.info(recipe_name,"Generated tramline map {}".format(tlm_fname.rstrip()))
    
def _make_tlm_for_kspec(image,tlm,instcode,targs):
    """
    Make a tramline map for the K-SPEC instruments. Also create and fill the wavelength HDU using the PREDICT_WAVELEN routine.

    Parameters
    ----------
    image : Tdfio
        Tdfio object for the input image frame. Opened for read-only on entry and return.
    tlm : Tdfio
        Tdfio object for the output tramline map frame. The tramline and wavelength data are written to this file on return. File is open on entry and return.
    instcode : TdfioInstType
        The instrument type in the format of the TdfioInstType enum. Retrievable from the instrument property of Tdfio objects.
    targs : TdfArgs
        The method arguments.
    
    Notes
    -----
    Uses routine "LOCATE_TRACES" to find and provide quadratic trace lines
    found within the image data of IM_ID. Then maps these traces to
    the fibers of the instrument using instrument specific "Identify Fiber"
    routine. Calculates an estimate of the FWHM of the fiber PSFs via the
    EXTRACT_PROFILE_SLICE routine. Creates a wavelength array using the
    PREDICT_WAVELEN() routine. All derived data is written to the TLM_ID
    data object.
    Note: This routine has evolved over a long history and there are
    still a few aspects that are present for historical reasons and
    may need updating in the future.
    """
    status = 0
    # TODO: turn tlm_id into a proper Tdfio object and return it?
    _twodfdr.make_tlm_other_mod.make_tlm_other(image.fid,tlm.fid,instcode,targs.sid,status)
    

def _predict_wavelen_for_kspec(image, tlm, instcode, targs)
    """
    Compute wavelength at the centre of each pixel for K-SPEC instrument.

    Parameters
    ----------
    image : Tdfio
        The input image frame.
    tlm : Tdfio
        The tramline map frame.
    instcode : TdfioInstType
        The instrument type.
    targs : TdfArgs
        The method arguments.

    Returns
    -------
    wavelen : ndarray
        The predicted wavelength at the pixel centres of each fibre.

    Notes
    -----
Purpose:
   Compute wavelength at the centre of each pixel for 6dF, AAOmega MOS and
   spiral IFU, HERMES or SAMI instruments.
Description:
   Calculation is done using various instrument parameters, certain FITS
   header keyword values and SDS arguments.
   Notice the accuracy of the returned wavelengths is NOT well understood.
   It is assumed NOT to be critical as they are normally calibrated later
   by the arc frame.
      ! arguments
   INTEGER,INTENT(IN) :: IM_ID          ! Image frame identifier. Used only to
                                        ! obtain certain FITS header values
   INTEGER,INTENT(IN) :: NX             ! Number of spectral pixels
   INTEGER,INTENT(IN) :: NF             ! Number of fibres in tram-line map
   INTEGER,INTENT(IN) :: NY             ! Number of spatial pixels
   REAL,   INTENT(IN) :: TLMA(NX,NF)    ! Tramline map data
   REAL,   INTENT(OUT):: WAVELEN(NX,NF) ! Predicted Wavelength (nanometres) at
                                        ! the pixel centres of each fibre
   INTEGER,INTENT(IN) :: ARGS           ! SDS identifier of structure containing
                                        ! method arguments, see above
   INTEGER,INTENT(INOUT):: STATUS       ! Global status
   """
    wavelen = None
    return wavelen
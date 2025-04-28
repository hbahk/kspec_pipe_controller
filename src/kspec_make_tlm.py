#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Hyeonguk Bahk (bahkhyeonguk@gmail.com)
# @Date: 2025-03-31
# @Filename: kspec_make_tlm.py
# @Description:
# This file contains `_make_tlm.py` of `twodfdr` package (@88c4e7f9),with small
# adjustments to make it work with the future KSPEC instrument.

import numpy as np
import cpl
from twodfdr import _twodfdr, config, args
from twodfdr.io import Tdfio, TdfioMode, TdfioType, TdfioInstType, TdfioKeyType
from twodfdr.modules.tlm import TLM
from twodfdr.modules.utils import Utils
from twodfdr.modules.sigma_profile import SigmaProfile
from twodfdr.modules.external_psf import ExternalPsf


def kspec_make_tlm(params, image, ofname=None, odir=None):
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
    im_im = Tdfio.suffix(image, "im")
    im_ex = Tdfio.suffix(image, "ex")
    im_red = Tdfio.suffix(image, "red")
    tlm = Tdfio.suffix(image, "tlm")

    # Add parameters necessary to run the recipe
    params.append(
        cpl.ui.ParameterValue(
            name="OBJECT_FILENAME",
            context="",
            default=image,
            description="Object filename.",
        )
    )
    params.append(
        cpl.ui.ParameterValue(
            name="IMAGE_FILENAME",
            context="",
            default=im_im,
            description="Image filename.",
        )
    )
    params.append(
        cpl.ui.ParameterValue(
            name="EXTRAC_FILENAME",
            context="",
            default=im_ex,
            description="Extract filename.",
        )
    )
    params.append(
        cpl.ui.ParameterValue(
            name="OUTPUT_FILENAME",
            context="",
            default=im_red,
            description="Output filename.",
        )
    )

    config.TdfConfig.oval(
        params, "TLMAP_FILENAME", ofname, tlm, "Tramline map filename."
    )
    config.TdfConfig.oval(params, "OUTDIR_NAME", odir, "NONE", "Output directory.")

    ta = args.TdfArgs(params=params)

    recipe_name = "kspec_make_tlm"

    # --Start reduce_fflat.F95 section--

    # This section does the work that is ordinarily done in reduce_fflat
    # since 2dfdr doesn't call make_tlm directly, it calls reduce_fflat with
    # MAKE_TLM_ONLY == 1 and DO_REDFL == 0

    # Call up any environmental initialisations defined in the parameters
    Utils.init_from_args(ta)

    # If an output directory is specified in the arguments then set this directory.
    out_dirname = ta.getc("OUT_DIRNAME", "NONE")
    Utils.outfile_setdir(out_dirname)
    cpl.core.Msg.info(recipe_name, "OUT_DIRNAME={}".format(out_dirname.rstrip()))

    # Produce im(age) frame if a tramline map is requested
    # TODO: make these robust separate Python functions ....
    _twodfdr.make_im__(args=ta.sid, status=status)

    # produce ex(tracted) frame
    # TODO: make these robust separate Python functions ....
    # NOT NEEDED
    # _twodfdr.make_ex(args=ta.sid,status=status)

    # --End reduce_fflat.F95 section--

    # Get name of the input image frame
    im_fname = ta.getc("IMAGE_FILENAME")

    # Get the name of the tramline map filename
    tlm_fname = TLM.get_tfname(ta)

    cpl.core.Msg.info(recipe_name, "Generating new tramline map from image. . .")

    # Open image file for read only
    timg = Tdfio.open(fname=im_fname, fmode=TdfioMode.R, ftype=TdfioType.STD)

    # We implement the function using only functions with Python bindings
    # Here, we are doing things the long way, following MAKE_TLM_FROM_IM in make_tlm.F95
    # cpl.core.Msg.info(recipe_name,"Generating TWODFDR tramline map")

    # Get code for instrument that created this file
    instcode = timg.instrument

    # Create empty tramline map file
    tram = Tdfio.open(fname=tlm_fname, fmode=TdfioMode.W, ftype=TdfioType.TLM)

    # Call routine appropriate for instrument to create TLM primary & WAVELA
    # All routines write tramline map into Primary HDU and predicted wavelength
    # array into 'WAVELA' HDU of TLM file TLM_ID
    TLM.other(timg, tram, instcode, ta)

    # Read tramline data into array TLM
    # npix, nfib = tram.get_size
    tlma = tram.tlmap_read()

    # If flat and create sigma profile option requested
    # then create sigma profile and add to tramline map FITS file
    # The ndf_class here is read from the timg
    # (in Fortran this happens in MAKE_TLM_OTHER (TLM.other above)
    # to populate the CLASS variable)
    if ta.getl("SIGPROF", False) and timg.ndf_class[:5] == "MFFFF":
        cpl.core.Msg.info(recipe_name, "Extracting Sigma Profile...")
        # First get image data into array
        im_a = timg.read()

        # Get which PSF profile to use
        psf_type = ta.getc("PSF_TYPE", "GAUSS")
        ExternalPsf.set_type(psf_type)

        if psf_type == "C2G2L":
            status = 0
            ExternalPsf.parse(ta, "MFFFF", status)

        # Extract sigma profile
        spa = SigmaProfile.sigma_profile(im_a, tlma, 5, 200, False)

        # Write sigma profile array to tramline map as new HDU
        tram.tsig_write(spa)

    # Copy 'UTDATE' and 'UTSTART' key headers from image into tramline map
    (utdate, dcomment) = timg.key_read("UTDATE", TdfioKeyType.STR)
    (utstart, scomment) = timg.key_read("UTSTART", TdfioKeyType.STR)
    tram.key_write("UTDATE", TdfioKeyType.STR, utdate, dcomment)
    tram.key_write("UTSTART", TdfioKeyType.STR, utstart, scomment)

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
    cpl.core.Msg.info(
        recipe_name, "Generated tramline map {}".format(tlm_fname.rstrip())
    )


def _make_tlm_for_kspec(image, tlm, instcode, targs):
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
    _twodfdr.make_tlm_other_mod.make_tlm_other(
        image.fid, tlm.fid, instcode, targs.sid, status
    )


def _predict_wavelen_for_kspec(image, tlm, instcode, targs):
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


def make_tlm_other(image, tram, instcode, args):
    """
    High-level Python implementation of the MAKE_TLM_OTHER subroutine.
    Generates a tramline map (TLM) for instruments like 6dF, AAOmega (MOS/IFU),
    HERMES, SAMI, KOALA, etc., by wrapping lower-level Fortran routines.

    Parameters
    ----------
    image : Tdfio
        Open Tdfio object for the input image (read-only).
    tram : Tdfio
        Open Tdfio object for the output tramline map (write mode).
    instcode : IntEnum or int
        Instrument code (e.g., TdfioInstType value) for the instrument.
    args : TdfArgs
        Parameters/arguments (configuration) object.

    Returns
    -------
    None
        (The tram Tdfio object is filled with the tramline map and related data.)
    """
    # Ensure we have internal IDs (Fortran handles) for image, output, and args
    im_id = int(image)  # Fortran ID for image frame (IM_ID)
    tlm_id = int(tram)  # Fortran ID for tramline map file (TLM_ID)
    args_id = int(args)  # Fortran ID for the ARGS structure

    status = np.int32(0)  # Fortran status (0 = DRS__OK)

    # Step 0: Read image data, variance data, and fiber types from input image
    # Get image dimensions
    n_pix = np.int32(0)  # will hold X-dimension (dispersion length)
    n_spatial = np.int32(0)  # will hold Y-dimension (spatial length)
    _twodfdr.tdfio_getsize(im_id, n_pix, n_spatial, status)
    
    # Allocate arrays for image and variance
    image_data = np.zeros((n_pix, n_spatial), dtype=np.float32)
    variance_data = np.zeros((n_pix, n_spatial), dtype=np.float32)
    
    # Read image and variance into arrays
    _twodfdr.tdfio_image_read(im_id, image_data, n_pix, n_spatial, status)
    _twodfdr.tdfio_var_read(im_id, variance_data, n_pix, n_spatial, status)
    
    # Read fiber type information for all possible fibers (returns actual count in n_fibers)
    max_fibers = (
        1000  # assume an upper bound for maximum number of fibers (MAX__NFIBRES)
    )
    fiber_types = np.empty(max_fibers, dtype="S1")  # 1-character codes for each fiber
    n_fibers = np.int32(0)
    _twodfdr.tdfio_fibres_read_types(
        im_id, np.int32(max_fibers), fiber_types, n_fibers, status
    )
    n_fibers = int(n_fibers)  # convert to Python int for easier use

    if status != 0:
        raise RuntimeError("Error reading image file (status={})".format(int(status)))

    # If instrument is HERMES, load instrument-specific coefficients (uses SPECTID header)
    if instcode == TdfioInstType.HERMES:
        spectrograph_id = image.key_read("SPECTID", TdfioKeyType.STR)[
            0
        ]  # SPECTID value
        # Initialize HERMES coefficients for this spectrograph (e.g., field distortions)
        _twodfdr.hermes_coeff_init(spectrograph_id, status)

    # Prepare tramline map array (n_pix x n_fibers), initially filled with 0
    tlm_array = np.zeros((int(n_pix), n_fibers), dtype=np.float32)

    # Determine which fibers are expected to have traces:
    # Use Fortran routine to classify fibers (Yes=has trace, No=definitely none, Maybe=potentially)
    fiber_trace_status = np.zeros(n_fibers, dtype=np.int32)
    _twodfdr.fibre_type_trace_status(
        instcode, fiber_types, np.int32(n_fibers), fiber_trace_status
    )
    
    # Count fiber categories for later logic
    yes_fibers = np.count_nonzero(fiber_trace_status == _twodfdr.tristate__yes)
    maybe_fibers = np.count_nonzero(fiber_trace_status == _twodfdr.tristate__maybe)
    no_fibers = np.count_nonzero(fiber_trace_status == _twodfdr.tristate__no)

    # Step 1: Locate fiber traces in the image
    # Allocate arrays for trace finding outputs
    trace_max = int(
        n_spatial
    )  # use image height as max possible traces (safe upper bound)
    trace_array = np.zeros((int(n_pix), trace_max), dtype=np.float32)  # TRACEA
    sigma_map = np.zeros(
        (int(n_pix), trace_max), dtype=np.float32
    )  # SIGMAP (for widths, initially empty)
    spatial_slice = np.zeros(
        int(n_spatial), dtype=np.float32
    )  # SPAT_SLICE (spatial profile)
    peak_positions = np.zeros(
        int(n_spatial), dtype=np.float32
    )  # PK_POSN (detected peak Y positions)
    n_traces = np.int32(0)
    
    # Set trace finding parameters from args
    sparse_fibs = args.getl(
        "SPARSE_FIBS", False
    )  # True if only a few fibers are present
    experimental_fit = args.getl(
        "TLM_FIT_RES", False
    )  # experimental fitting restrictions if poor data
    quick_and_dirty = args.getl(
        "QAD_PKSEARCH", False
    )  # force quick peak search method if True
    
    # Choose peak search method: 0 = standard (Echelle), 1 = alternative (all local peaks)
    pk_search_method = (
        1
        if quick_and_dirty
        or instcode in (TdfioInstType.AAOMEGA_IFU, TdfioInstType.AAOMEGA_KOALA)
        else 0
    )
    
    # Determine distortion modeling flag (DODIST): turn off for known highly distorted instruments
    do_distortion = True
    if (
        instcode == TdfioInstType.SPECTOR_HECTOR
        or instcode == TdfioInstType.AAOMEGA_HECTOR
    ):
        do_distortion = False
    # Invoke the trace finding routine (LOCATE_TRACES)
    _twodfdr.locate_traces(
        image_data,
        n_pix,
        n_spatial,
        trace_array,
        np.int32(trace_max),
        n_traces,
        np.int32(n_fibers),
        spatial_slice,
        peak_positions,
        np.int32(2),  # ORDER: use quadratic (2nd order) smoothing for traces by default
        sparse_fibs,
        experimental_fit,
        np.int32(pk_search_method),
        do_distortion,
        status,
    )
    
    n_traces = int(n_traces)  # convert to Python int
    if status != 0:
        raise RuntimeError("Trace location failed (status={})".format(int(status)))
    # At this point, trace_array[:,0:n_traces] contains the smoothed trace for each detected fiber-like peak,
    # spatial_slice contains a representative spatial profile, and peak_positions[0:n_traces] the Y positions of peaks.

    # Step 2: Estimate fiber profile widths (FWHM) using a spatial slice
    # Allocate peak heights and widths arrays for each found trace
    peak_heights = np.zeros(n_traces, dtype=np.float32)
    peak_widths = np.zeros(n_traces, dtype=np.float32)
    
    # Determine an approximate intensity at each peak position from the spatial slice
    for i in range(n_traces):
        # Fortran used 1-based indexing and int(pos+0.5); in Python, convert to 0-based index
        idx = int(np.floor(peak_positions[i] + 0.5))
        idx = max(idx, 0)
        if idx >= len(spatial_slice):
            idx = len(spatial_slice) - 1
        peak_heights[i] = spatial_slice[idx]
        
    # Set PSF type (point-spread function shape) from args and external PSF module
    psf_type = args.getc("PSF_TYPE", "GAUSS")
    ExternalPsf.set_type(psf_type)
    
    # If using complex PSF (e.g., 'C2G2L'), parse its parameters for this image's class
    image_class = image.ndf_class  # e.g., 'MFFFF' for flat field
    if psf_type.upper() == "C2G2L":
        parse_status = 0
        ExternalPsf.parse(args, image_class, parse_status)
        
    # Call the profile extraction routine to estimate sigma for each peak (assuming GAUSS or chosen PSF)
    _twodfdr.extract_profile_slice(
        spatial_slice,
        np.int32(n_spatial),
        peak_positions,
        peak_heights,
        np.int32(n_traces),
        peak_widths,
        False,
    )  # last False -> do not plot (as in Fortran .FALSE.)
    
    # Convert sigma to FWHM for Gaussian (FWHM = 2.35482 * sigma)
    peak_widths *= 2.35482
    
    # Compute median FWHM across all fibers (MWIDTH)
    median_width = float(np.median(peak_widths[:n_traces]))
    
    # For KOALA (and potentially SAMI), use wavelet-based block analysis to refine MWIDTH (if enabled)
    if instcode == TdfioInstType.AAOMEGA_KOALA:
        nblocks = 40  # KOALA has 40 IFU bundles
        mwidth2 = np.float32(0.0)
        _twodfdr.block_sigma_estimate(
            spatial_slice,
            np.int32(n_spatial),
            np.int32(n_fibers),
            np.int32(nblocks),
            mwidth2,
            status,
        )
        # Convert resulting sigma to FWHM
        mwidth2_val = float(mwidth2) * 2.35482
        if mwidth2_val >= 0.0:
            # If wavelet analysis produced a valid estimate, use it
            median_width = mwidth2_val

    # Step 3: Identify which trace corresponds to each fiber number
    # We will produce a mapping from fiber index (1..NF) to trace index (1..NTRACES).
    match_vector = np.zeros(
        2 * n_fibers, dtype=np.int32
    )  # fiber -> trace mapping (0 if no match)
    
    # Determine if we can skip complex identification:
    if (n_traces == yes_fibers) or (n_traces == yes_fibers + maybe_fibers):
        # If the number of found traces matches the expected fibers in use (including "maybe"),
        # assume a one-to-one mapping in order of position.
        # Sort peak positions to align with fiber numbering (usually fibers are ordered by position).
        sorted_idx = np.argsort(peak_positions[:n_traces])
        
        # Assign each found trace to the next fiber that should have a trace
        fiber_idx = 0
        for trace_idx in sorted_idx:
            # Skip fibers that officially have no trace (dead/parked) when assigning
            while (
                fiber_idx < n_fibers
                and fiber_trace_status[fiber_idx] == _twodfdr.tristate__no
            ):
                fiber_idx += 1
            if fiber_idx >= n_fibers:
                break
            # Assign this trace to fiber number (fiber_idx+1)
            match_vector[fiber_idx] = trace_idx + 1  # store 1-based trace index
            fiber_idx += 1
        # Any remaining fibers (if any) with no trace remain with 0 in match_vector.
    else:
        # Perform instrument-specific fiber identification using known fiber spacing or patterns
        if instcode in (
            TdfioInstType.TWODF,
            TdfioInstType.AAOMEGA,
        ):  # AAOmega MOS (2dF)
            # Mark known dead fibers for 2dF/AAOmega: read 'SOURCE' (plate ID) and get dead list
            plate_id = image.key_read("SOURCE", TdfioKeyType.STR)[0]
            dead_mask = np.zeros(
                (400,), dtype=bool
            )  # assume at most 400 fibers for 2dF
            _twodfdr.aaomega2df_deadfibres(
                plate_id, dead_mask, np.int32(dead_mask.size)
            )
            # Update fiber_types: label any dead fibers as 'F' (no trace) [oai_citation_attribution:21‡file-wxwp4zo9cpegs3cymqwzlr](file://file-WXwP4zo9CpeGs3CYMqWZLR#:~:text=CALL%20TDFIO_KYWD_READ_CHAR,implies%20gives%20no%20trace%20ENDDO)
            for fibno in range(min(n_fibers, dead_mask.size)):
                if dead_mask[fibno]:
                    fiber_types[fibno] = b"F"
            # Get nominal fiber-to-fiber gap values for this plate/spectrograph
            spectro_id = image.key_read("SPECTID", TdfioKeyType.STR)[0]
            gap_values = np.zeros(n_fibers, dtype=np.float32)
            _twodfdr.aaomega_nominal_gaps(spectro_id, gap_values, np.int32(n_fibers))
            # Identify fibers using expected gaps between traces [oai_citation_attribution:22‡file-wxwp4zo9cpegs3cymqwzlr](file://file-WXwP4zo9CpeGs3CYMqWZLR#:~:text=%21%20Now%20we%20identify%20the,NF%2CFIB_TYPEV%2CNPKS%2CPK_POSN%2CGAPV%2CMATCHV%2C%26%20MFIB_POSN%2CSTATUS)
            _twodfdr.ident_fibs_from_gaps(
                np.int32(n_fibers),
                fiber_types,
                np.int32(n_traces),
                peak_positions,
                gap_values,
                match_vector,
                np.zeros(n_fibers, dtype=np.float32),
                status,
            )

        elif instcode == TdfioInstType.AAOMEGA_IFU:
            # AAOmega/Spiral IFU: use known nominal gaps for IFU fibers [oai_citation_attribution:23‡file-wxwp4zo9cpegs3cymqwzlr](file://file-WXwP4zo9CpeGs3CYMqWZLR#:~:text=%21%20Get%20the%20nominal%20gap,IFU%20fibers.%20CALL%20AAOMEGAIFU_NOMINAL_GAPS%28GAPV%2CNF)
            gap_values = np.zeros(n_fibers, dtype=np.float32)
            _twodfdr.aaomegaifu_nominal_gaps(gap_values, np.int32(n_fibers))
            _twodfdr.ident_fibs_from_gaps(
                np.int32(n_fibers),
                fiber_types,
                np.int32(n_traces),
                peak_positions,
                gap_values,
                match_vector,
                np.zeros(n_fibers, dtype=np.float32),
                status,
            )

        elif instcode == TdfioInstType.AAOMEGA_KOALA:
            # KOALA IFU: should normally have one-to-one mapping; if not, log a warning [oai_citation_attribution:24‡file-wxwp4zo9cpegs3cymqwzlr](file://file-WXwP4zo9CpeGs3CYMqWZLR#:~:text=%21%20KOALA%20does%20not%20have,trace%20so%20give%20a%20warning)
            if abs(n_traces - yes_fibers) != 0:
                print(
                    "Warning: Unexpected trace count discrepancy for KOALA (found {}, expected {}).".format(
                        n_traces, yes_fibers
                    )
                )
            # Use nominal gaps for KOALA [oai_citation_attribution:25‡file-wxwp4zo9cpegs3cymqwzlr](file://file-WXwP4zo9CpeGs3CYMqWZLR#:~:text=%21%20Get%20the%20nominal%20gap,CALL%20KOALA_NOMINAL_GAPS%28GAPV%2CNF)
            gap_values = np.zeros(n_fibers, dtype=np.float32)
            _twodfdr.koala_nominal_gaps(gap_values, np.int32(n_fibers))
            _twodfdr.ident_fibs_from_gaps(
                np.int32(n_fibers),
                fiber_types,
                np.int32(n_traces),
                peak_positions,
                gap_values,
                match_vector,
                np.zeros(n_fibers, dtype=np.float32),
                status,
            )
            # Report any expected fibers that were not found [oai_citation_attribution:26‡file-wxwp4zo9cpegs3cymqwzlr](file://file-WXwP4zo9CpeGs3CYMqWZLR#:~:text=%21%20Announce%20any%20discrepencies%20found,MSG%29%2CSTATUS%29%20ENDIF)
            for fibno in range(n_fibers):
                if match_vector[fibno] == 0 and (
                    fiber_types[fibno] == b"P" or fiber_types[fibno] == b"S"
                ):
                    print(
                        f"--> Could not locate expected trace for fiber {fibno+1} (type {fiber_types[fibno].decode()})"
                    )

        elif instcode == TdfioInstType.AAOMEGA_SAMI:
            # SAMI multi-IFU: use special identification routine [oai_citation_attribution:27‡file-wxwp4zo9cpegs3cymqwzlr](file://file-WXwP4zo9CpeGs3CYMqWZLR#:~:text=%21%20SAMI%20had%20some%20very,0%20CALL%20IDENTAAOMEGASAMI_OLD%28SPAT_SLICE%2CNY%2CNPKS%2CPK_POSN%2CNF%2CFIB_TYPEV%2CMATCHV%29%20NBLOCKS%3D13)
            # (Replace bad values in spatial_slice with 0 to avoid issues)
            spatial_slice_clean = spatial_slice.copy()
            spatial_slice_clean[np.isnan(spatial_slice_clean)] = 0.0
            _twodfdr.identaaomegasami_old(
                spatial_slice_clean,
                np.int32(n_spatial),
                np.int32(n_traces),
                peak_positions,
                np.int32(n_fibers),
                fiber_types,
                match_vector,
            )
            # (If an updated IDENTAAOMEGASAMI exists, it could be used instead)

        elif instcode == TdfioInstType.HERMES:
            # HERMES: allow for "phantom" traces by using a robust gaps matching [oai_citation_attribution:28‡file-wxwp4zo9cpegs3cymqwzlr](file://file-WXwP4zo9CpeGs3CYMqWZLR#:~:text=%21%20with%20the%20assumption%20that,NF%2CFIB_TYPEV%2CNPKS%2CPK_POSN%2C%26%20PK_HGTS%2CGAPV%2CMATCHV%2CMFIB_POSN%2CSTATUS)
            spectro_id = image.key_read("SPECTID", TdfioKeyType.STR)[0]
            gap_values = np.zeros(n_fibers, dtype=np.float32)
            _twodfdr.hermes_nominal_gaps(spectro_id, gap_values, np.int32(n_fibers))
            _twodfdr.ident_fibs_from_gaps_given_phantoms(
                np.int32(n_fibers),
                fiber_types,
                np.int32(n_traces),
                peak_positions,
                peak_heights,
                gap_values,
                match_vector,
                np.zeros(n_fibers, dtype=np.float32),
                status,
            )

        elif instcode == TdfioInstType.SIXDF:
            # 6dF: use known fiber gaps for this instrument [oai_citation_attribution:29‡file-wxwp4zo9cpegs3cymqwzlr](file://file-WXwP4zo9CpeGs3CYMqWZLR#:~:text=CASE%28INST_6DF%29%20%21,CALL%20SIXDF_NOMINAL_GAPS%28GAPV%2CNF)
            gap_values = np.zeros(n_fibers, dtype=np.float32)
            _twodfdr.sixdf_nominal_gaps(gap_values, np.int32(n_fibers))
            _twodfdr.ident_fibs_from_gaps(
                np.int32(n_fibers),
                fiber_types,
                np.int32(n_traces),
                peak_positions,
                gap_values,
                match_vector,
                np.zeros(n_fibers, dtype=np.float32),
                status,
            )

        elif instcode == TdfioInstType.TAIPAN:
            # TAIPAN: uses an equidistant fiber position model (actually 150 fibers out of 300 slots) [oai_citation_attribution:30‡file-wxwp4zo9cpegs3cymqwzlr](file://file-WXwP4zo9CpeGs3CYMqWZLR#:~:text=CASE%28INST_TAIPAN%29%20%21) [oai_citation_attribution:31‡file-wxwp4zo9cpegs3cymqwzlr](file://file-WXwP4zo9CpeGs3CYMqWZLR#:~:text=%21%20Get%20the%20nominal%20gap,CALL%20TDFIO_KYWD_READ_CHAR%28IM_ID%2C%27SPECTID%27%2CSPECTID%2CCMT%2CSTATUS%29%20CALL%20TAIPAN_NOMINAL_FIBPOS%28SPECTID%2CARPV%2CNF)
            spectro_id = image.key_read("SPECTID", TdfioKeyType.STR)[0]
            archived_positions = np.zeros(n_fibers, dtype=np.float32)
            _twodfdr.taipan_nominal_fibpos(
                spectro_id, archived_positions, np.int32(n_fibers)
            )
            match_vector[:] = 0
            _twodfdr.ident_fibs_from_posns(
                np.int32(n_fibers),
                fiber_types,
                np.int32(n_traces),
                peak_positions,
                archived_positions,
                match_vector,
                np.zeros(n_fibers, dtype=np.float32),
                status,
            )
            print(
                f"Identified {np.count_nonzero(match_vector[:n_fibers] != 0)} fiber traces for TAIPAN."
            )

        else:
            # If instrument code not explicitly handled, default to gap-based matching if possible
            gap_values = np.zeros(n_fibers, dtype=np.float32)
            try:
                # Attempt to get nominal gaps if a routine exists for this instcode
                _twodfdr.aaomega_nominal_gaps(
                    image.key_read("SPECTID", TdfioKeyType.STR)[0],
                    gap_values,
                    np.int32(n_fibers),
                )
            except Exception:
                # Fallback: assume roughly uniform spacing if unknown instrument
                gap_values[:] = np.median(np.diff(np.sort(peak_positions[:n_traces])))
            _twodfdr.ident_fibs_from_gaps(
                np.int32(n_fibers),
                fiber_types,
                np.int32(n_traces),
                peak_positions,
                gap_values,
                match_vector,
                np.zeros(n_fibers, dtype=np.float32),
                status,
            )
    # End of identification

    # Step 4: Build the tramline map (TLMA) by assigning traces to fibers
    missing_traces = 0
    for fib_index in range(n_fibers):
        trace_no = match_vector[fib_index]
        if trace_no <= 0:
            # No trace identified for this fiber; leave tlm_array column as zeros (bad trace)
            missing_traces += 1
            continue
        trace_idx = trace_no - 1  # convert to 0-based index for Python
        tlm_array[:, fib_index] = trace_array[:, trace_idx]
        # If sigma_map data were produced (e.g., by a Gaussian fit across image), we would similarly assign it per fiber here.

    # Optionally, output fiber positions to an external file for reference (as Fortran does for debugging)
    # _twodfdr.IDENT_TO_EXTERNAL_FILE(np.int32(n_fibers), tlm_array[int(n_pix)//2, :], status)

    # Interpolate across any missing fibers to smooth the tramline map if needed
    sep_value = _twodfdr.fibre_sep(
        instcode
    )  # nominal fiber separation in pixels for this instrument
    if instcode == TdfioInstType.TAIPAN:
        # Use specialized interpolation for TAIPAN (non-uniform fiber spacing) [oai_citation_attribution:32‡file-wxwp4zo9cpegs3cymqwzlr](file://file-WXwP4zo9CpeGs3CYMqWZLR#:~:text=SEP%20%3D%20FIBRE_SEP,NX%2CNF%2CTLMA%2CMATCHV%2CARPV%29%20ELSE)
        _twodfdr.interpolate_tlms_taipan(
            np.int32(n_pix),
            np.int32(n_fibers),
            tlm_array,
            match_vector,
            archived_positions,
        )
    else:
        _twodfdr.interpolate_tlms(
            np.int32(n_pix),
            np.int32(n_fibers),
            tlm_array,
            match_vector,
            np.float32(sep_value),
        )

    # Step 5: Write results to the output tramline map FITS file
    # Write the tramline map primary data (TLMA array) into the tram file
    _twodfdr.tdfio_tlmap_write(
        tlm_id, tlm_array, np.int32(n_pix), np.int32(n_fibers), status
    )
    # Determine if we need to generate the wavelength array using PREDICT_WAVELEN
    sc_corrector = args.getl("TLMGAUSSFIT", False)  # corresponds to SC_CORRECTOR
    prf_corrector = args.getl("PRFGAUSSFIT", False)  # corresponds to PRF_CORRECTOR
    if sc_corrector or prf_corrector:
        # Allocate wavelength array and compute predicted wavelengths [oai_citation_attribution:33‡file-wxwp4zo9cpegs3cymqwzlr](file://file-WXwP4zo9CpeGs3CYMqWZLR#:~:text=ALLOCATE,RETURN)
        wave_array = np.zeros((int(n_pix), n_fibers), dtype=np.float32)
        _twodfdr.predict_wavelen( #TODO: change to _predict_wavelen_for_kspec
            im_id,
            np.int32(n_pix),
            np.int32(n_fibers),
            np.int32(n_spatial),
            tlm_array,
            wave_array,
            args_id,
            status,
        )
        if status != 0:
            raise RuntimeError(
                "Wavelength prediction failed (status={})".format(int(status))
            )
        _twodfdr.tdfio_wave_write(
            tlm_id, wave_array, np.int32(n_pix), np.int32(n_fibers), status
        )
    # Write the median FWHM (MWIDTH) to the tram file header [oai_citation_attribution:34‡file-wxwp4zo9cpegs3cymqwzlr](file://file-WXwP4zo9CpeGs3CYMqWZLR#:~:text=%21%20Write%20FWHM%20estimate%20to,Median%20tramline%20width%20FWHM%3D%27%2CMWIDTH)
    _twodfdr.tdfio_kywd_write_real(
        tlm_id, "MWIDTH", np.float32(median_width), "Median spatial FWHM", status
    )

    # If a full sigma profile correction was requested (SC or PRF), we could write the sigma map:
    if sc_corrector or prf_corrector:
        # In PRF_GAUSSFIT mode, we refine the sigma for each trace across all X using SVARPRO_FIT [oai_citation_attribution:35‡file-wxwp4zo9cpegs3cymqwzlr](file://file-WXwP4zo9CpeGs3CYMqWZLR#:~:text=SIGMAP%28IDX1%2C1%3ANTRACES%29%3DPK_WDTS%281%3ANTRACES%29%2F2)
        if prf_corrector:
            # Initialize sigma_map for all traces as constant (peak_widths) across X, then fit
            for j in range(n_traces):
                sigma_map[:, j] = (
                    peak_widths[j] / 2.35482
                )  # back to sigma as initial guess
            _twodfdr.svarpro_fit(
                image_data,
                np.int32(n_pix),
                np.int32(n_spatial),
                sigma_map[:, :n_traces],
                status,
            )
        # Map sigma data from trace order to fiber order and write as TSIG HDU [oai_citation_attribution:36‡file-wxwp4zo9cpegs3cymqwzlr](file://file-WXwP4zo9CpeGs3CYMqWZLR#:~:text=%21%20First%20of%20all%20Remap,numbers%20from%20NF%20to%201) [oai_citation_attribution:37‡file-wxwp4zo9cpegs3cymqwzlr](file://file-WXwP4zo9CpeGs3CYMqWZLR#:~:text=CALL%20TDFIO_TSIG_WRITE)
        for fib_index in range(n_fibers):
            trace_no = match_vector[fib_index]
            if trace_no <= 0:
                sigma_map[:, fib_index] = (
                    _twodfdr.val__badr
                )  # fill missing fiber with bad value constant
            else:
                sigma_map[:, fib_index] = sigma_map[:, trace_no - 1]
        _twodfdr.tdfio_tsig_write(
            tlm_id, sigma_map[:, :n_fibers], np.int32(n_pix), np.int32(n_fibers), status
        )

    # (Any additional keywords like UTDATE/UTSTART could be copied from image to tram if needed,
    # as done in the original wrapper.)
    return

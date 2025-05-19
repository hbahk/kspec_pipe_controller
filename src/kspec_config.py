from twodfdr.config import TdfConfig, AAOmegaConfig

class Taipan(AAOmegaConfig):
    """
    Defines a Taipan configuration.

    Notes
    -----
    Taipan is an AAOmega-based instrument with specific requirements for wiggle correction
    and other parameters.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        # Override default parameters from taipan.idx
        self._p["USEBIASIM"].value = True
        self._p["SCATSUB"].value = "NONE"
        self._p["ARCFITORDER"].value = 4
        self._p["SKYFITORDER"].value = 2
        self._p["SKYSUB"].value = False
        self._p["THRUPUT"].value = True
        self._p["SKYSPRSMP"].value = True

        # Add Taipan specific parameters
        self.val("USETHPTFILE", True, "Use file THROUGHPUT.fits if one is available")
        self.val("PROP_BTHPUT", True, "If the calculated throughput of a fibre is bad then render all associated reduced spectra as bad")
        
        # Add Taipan specific experimental parameters
        self.val("DEWIGGLE", False, "Correct for wiggles in red Taipan Spectra.")
        self.val("DEWIGGLE_HARSH", False, "Harsh correction for wiggles in red Taipan Spectra.")
        self.val("DEWIGGLE_SMART", False, "Smart correction for wiggles in red Taipan Spectra.")

    def setup_combine(self):
        """
        Taipan Specific Combination Parameters.

        Notes
        -----
        Inherits from AAOmega combine parameters but can be customized if needed.
        """
        super().setup_combine()
        # Add any Taipan-specific combine parameters here if needed

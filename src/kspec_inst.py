from twodfdr.io import Tdfio, TdfioInstType, TdfioKeyType

class KspecTdfioInstType(TdfioInstType):
    """
    Class to handle the TDFIO instance type for Kspec.
    
    Attributes
    ----------
    inst_type : str
        The instrument type.
    inst_name : str
        The instrument name.
    inst_version : str
        The instrument version.
    """
    KSPEC = 21
    """KSPEC"""
    
    # the other instrument types can be removed in the future
    # as they will not be used in the Kspec project.
    TWODF = 1
    """2dF"""
    SIXDF = 2
    """6dF"""
    AAOMEGA_2DF = 3
    """2dF-AAOmega"""
    AAOMEGA_IFU = 4
    """IFU-AAOMega"""
    FMOS = 5
    """FMOS"""
    HERMES = 6
    """HERMES"""
    AAOMEGA_SAMI = 7
    """SAMI-AAOmega"""
    AAOMEGA_KOALA = 8
    """Koala-AAOmega"""
    GENERIC = 9
    """Generic"""
    GENERIC_IR = 10
    """Generic IR"""
    TAIPAN = 11
    """TAIPAN"""
    AAOMEGA_HECTOR= 12
    """Hector-AAOmega"""
    SPECTOR_HECTOR= 13
    """Spector-AAOmega"""

class KspecTdfio(Tdfio):
    """
    A subclass of Tdfio that adds Kspec-specific functionality.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
           
    @property
    def instrument(self):
        """
        Returns the code for the instrument that produced the current file.

        Notes
        -----
        A central database for instrument names. Used to tranlate the value
        given in the file's "INSTRUME" FITS keyword to the corresponding code
        (see TDFRED_PAR). The code is then normally used is a SELECT statement.

        Returns
        -------
        TdfioInstType
            The instrument type of the current file.
        """
        if not self.key_exists("INSTRUME"):
            raise RuntimeError("Missing INSTRUME keyword. Cannot determine instrument type.")

        (inst_str,inst_comment) = self.key_read("INSTRUME",TdfioKeyType.STR)
        inst_str = inst_str.upper()

        if len(inst_str) == 0:
            raise RuntimeError("INSTRUME keyword is empty. Cannot determine instrument type.")

        if inst_str == 'KSPEC':
            # KSPEC is not in TdfioInstType
            # remove this once KSPEC is fully integrated
            return KspecTdfioInstType.KSPEC
        else:
            return super().instrument

    

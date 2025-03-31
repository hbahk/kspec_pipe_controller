from twodfdr.io import TdfioInstType

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
    

    

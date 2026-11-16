"""
Centralized configuration for IATIR project.
Contains sensor parameters (HyTES, ASTER) and path definitions.
"""
import os

# Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SENSOR_CONFIG = {
    'hytes': {
        'name': 'hytes',
        'n_bands': 256,
        'wavelength_range': (7.5, 12.0),
        'calibration_file_wmx': os.path.join(PROJECT_ROOT, 'bibliografic_references', 'hytes', 'HyTES_WMX_v6.csv'),
        'calibration_file_srf': os.path.join(PROJECT_ROOT, 'bibliografic_references', 'hytes', 'HyTES_SRF_v6.csv'),
        'processed_params_file': os.path.join(PROJECT_ROOT, 'data', 'processed', 'hytes_real_parameters.csv'),
        'description': 'Hyperspectral Thermal Emission Spectrometer (256 bands)'
    },
    'aster': {
        'name': 'aster',
        'n_bands': 5,
        'band_indices': [11, 12, 13, 14, 15], # RTTOV SRF Channels 11-15 correspond to ASTER TIR Bands 10-14
        'band_names': ['Band 10', 'Band 11', 'Band 12', 'Band 13', 'Band 14'],
        'wavelength_range': (8.0, 12.0),
        'srf_dir': os.path.join(PROJECT_ROOT, 'bibliografic_references', 'aster'),
        # Fallback FWHMs if SRF files are missing (Approximate values in microns)
        'fallback_params': {
            'centers': [8.281, 8.633, 9.079, 10.662, 11.293],
            'fwhms': [0.35, 0.35, 0.35, 0.70, 0.70] 
        },
        'description': 'Advanced Spaceborne Thermal Emission and Reflection Radiometer (5 TIR bands)'
    }
}

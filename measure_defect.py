import numpy as np
from scipy import ndimage

def measure_defect(mask_array, pixel_size_mm=0.1):
    """
    Takes binary mask, returns defect measurements.
    pixel_size_mm: real-world size of each pixel (calibration needed per setup)
    """
    # Label connected regions
    labeled, num_features = ndimage.label(mask_array)
    
    measurements = []
    for i in range(1, num_features + 1):
        region = (labeled == i)
        area_pixels = region.sum()
        area_mm2 = area_pixels * (pixel_size_mm ** 2)
        
        # Bounding box for length estimation
        coords = np.where(region)
        height = (coords[0].max() - coords[0].min()) * pixel_size_mm
        width = (coords[1].max() - coords[1].min()) * pixel_size_mm
        max_length = max(height, width)
        
        measurements.append({
            'area_mm2': round(area_mm2, 2),
            'max_length_mm': round(max_length, 2),
            'height_mm': round(height, 2),
            'width_mm': round(width, 2),
            'area_pixels': int(area_pixels)
        })
    
    return measurements

def check_tolerance(measurements, tolerances):
    """
    Compare measurements against tolerance limits.
    Returns: list of (measurement, pass/fail, reason)
    """
    results = []
    for m in measurements:
        passed = True
        reasons = []
        
        if m['max_length_mm'] > tolerances.get('max_length_mm', float('inf')):
            passed = False
            reasons.append(f"Length {m['max_length_mm']}mm exceeds limit {tolerances['max_length_mm']}mm")
        
        if m['area_mm2'] > tolerances.get('max_area_mm2', float('inf')):
            passed = False
            reasons.append(f"Area {m['area_mm2']}mm² exceeds limit {tolerances['max_area_mm2']}mm²")
        
        results.append({
            'measurement': m,
            'passed': passed,
            'reasons': reasons if reasons else ['Within tolerance']
        })
    
    return results


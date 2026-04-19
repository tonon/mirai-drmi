from app.inference.geometry import lesion_area, lesion_perimeter, area_ratio
from app.inference.morphology import circularity, count_components
from app.inference.topology import centroid, distance_to_center, lesion_location
from app.inference.lesion_typing import infer_lesion_type


def analyze_lesion(mask, retina_mask=None):
    area = lesion_area(mask)
    perim = lesion_perimeter(mask)
    ar = area_ratio(mask, retina_mask)

    circ = circularity(area, perim)
    comps = count_components(mask)

    cent = centroid(mask)
    dist = distance_to_center(mask.shape, cent)
    loc = lesion_location(dist)

    pattern = infer_lesion_type(ar, circ, comps)

    return {
        "area_pixels": area,
        "area_ratio": ar,
        "perimeter": perim,
        "circularity": circ,
        "components": comps,
        "centroid": cent,
        "distance_to_center": dist,
        "location": loc,
        "lesion_pattern": pattern
    }
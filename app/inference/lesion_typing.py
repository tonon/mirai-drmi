def infer_lesion_type(area_ratio, circularity, components):
    if area_ratio < 0.01 and circularity > 0.7:
        return "Drusa dura (padrão compatível)"
    if area_ratio < 0.05 and circularity > 0.4:
        return "Drusa mole (padrão compatível)"
    if area_ratio >= 0.05 and circularity < 0.4:
        return "Atrofia geográfica (padrão compatível)"
    return "Padrão indeterminado"
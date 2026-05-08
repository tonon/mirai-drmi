import cv2
import numpy as np
from pathlib import Path

# ============================================================
# 1️⃣ Função de separação morfológica de drusas
# ============================================================

def separate_drusas(mask: np.ndarray) -> np.ndarray:
    """
    Aplica morfologia + watershed para separar drusas adjacentes.

    Entrada:
        mask: np.ndarray binária (0/1 ou 0/255)

    Saída:
        mask_sep: máscara binária (0/255) com drusas separadas
    """

    # Garantir máscara binária 0/255
    mask_bin = (mask > 0).astype(np.uint8) * 255

    # 🔹 Opening para remover conexões finas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=2)

    # 🔹 Transformada de distância
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)

    # 🔹 Normalização
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # 🔹 Picos (centros das drusas)
    _, sure_fg = cv2.threshold(dist_norm, 0.35, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    # 🔹 Fundo seguro
    sure_bg = cv2.dilate(opened, kernel, iterations=3)

    # 🔹 Região desconhecida
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 🔹 Componentes conectados (marcadores)
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 🔹 Watershed
    img_color = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    # 🔹 Máscara final: regiões > 1
    mask_sep = np.zeros_like(mask_bin)
    mask_sep[markers > 1] = 255

    return mask_sep


# ============================================================
# 2️⃣ Função principal de inferência
# ============================================================

def infer_image(image_path: Path) -> dict:
    """
    Pipeline completo de inferência:
    - leitura da imagem
    - segmentação (mock ou modelo real)
    - separação morfológica de drusas
    - cálculo de métricas
    - salvamento da máscara
    - retorno estruturado
    """

    # --------------------------------------------------------
    # 1️⃣ Ler imagem
    # --------------------------------------------------------
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Imagem inválida")

    h, w, _ = image.shape

    # --------------------------------------------------------
    # 2️⃣ Segmentação (⚠️ MOCK temporário)
    # ⚠️ AQUI você liga a U‑Net / ensemble depois
    # --------------------------------------------------------
    mask_model = np.zeros((h, w), dtype=np.uint8)

    # EXEMPLO: duas drusas artificiais separáveis
    cv2.circle(mask_model, (w // 2 - 40, h // 2), 18, 1, -1)
    cv2.circle(mask_model, (w // 2 + 40, h // 2), 22, 1, -1)

    # --------------------------------------------------------
    # 3️⃣ Separação morfológica das drusas
    # --------------------------------------------------------
    mask_refined = separate_drusas(mask_model)

    # --------------------------------------------------------
    # 4️⃣ Métricas globais (máscara refinada!)
    # --------------------------------------------------------
    area_pixels = int((mask_refined > 0).sum())
    area_ratio = area_pixels / (h * w)

    contours, _ = cv2.findContours(
        mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    perimeter = sum(cv2.arcLength(c, True) for c in contours)

    circularity = (4 * np.pi * area_pixels) / (perimeter ** 2 + 1e-6)

    components = len(contours)

    # --------------------------------------------------------
    # 5️⃣ Salvar máscara refinada
    # --------------------------------------------------------
    mask_img = mask_refined.astype("uint8")
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    mask_path = temp_dir / f"{image_path.stem}_mask.png"
    cv2.imwrite(str(mask_path), mask_img)

    # --------------------------------------------------------
    # 6️⃣ Retorno estruturado (contrato da API)
    # --------------------------------------------------------
    return {
        "image_id": image_path.stem,
        "predicted_class": "AMD",
        "lesion_pattern": "Drusa mole (compatível)",
        "area_pixels": area_pixels,
        "area_ratio": area_ratio,
        "perimeter": perimeter,
        "circularity": circularity,
        "components": components,
        "location": "central",
        "distance_to_center": 0.0,
        "mask_url": f"/temp/{mask_path.name}",
    }
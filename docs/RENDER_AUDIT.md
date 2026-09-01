# Object Rendering Color Issue - Summary for Paper A.10

## Original Design: Lighting-Independent Object Injection

PIN's object injection mechanism is **intentionally designed** to be independent of scene lighting:

1. **`NO_LIGHT_KEY`** (env.py L285): tells Habitat-sim not to apply scene lights to the injected object
2. **`force_flat_shading: true`** (object_config.json): forces flat shader, bypassing all lighting calculations

This ensures injected objects render with consistent colors regardless of the scene's lighting conditions. For the vast majority of objects this works correctly — their `baseColorFactor` matches their intended color.

## The Issue: One Problematic Instance

`force_flat_shading=true` causes the flat shader to only read `baseColorFactor` (a single RGBA constant), discarding GLB textures. For most objects this is fine (single-colored objects), but one specific instance had a mismatch:

- **Object**: `570b82c4391c49ddb1e471e6e55de9f4` (category: ball, adidas soccer ball)
- **Expected**: White + black/gold curved arcs (complex texture)
- **Rendered**: Solid red (its `baseColorFactor` happened to be red, texture was discarded by flat shader)

## Explored Alternative: Emissive Self-Illumination

We explored an alternative rendering approach (`fix_object_rendering.py`) using the emissive channel to achieve lighting independence while preserving textures:

- `force_flat_shading` -> `false`, copy `baseColorTexture` -> `emissiveTexture`, `emissiveFactor=[1,1,1]`, `baseColorFactor=[0,0,0,1]`
- PBR formula: `C_final = emissive × T_emissive + PBR(black) ≈ T_baseColor` (lighting-independent)

However, even with the emissive fix, this specific ball still rendered as solid white (texture detail lost). This confirmed the issue is a Habitat-sim renderer limitation for this GLB's texture format, not just a shading mode problem.

## Final Solution: Automated Detection + Removal

**We kept the original PIN injection approach** (`NO_LIGHT_KEY` + `force_flat_shading=true`) and used an automated scan to find and remove the anomalous instance.

Script: `scan_render_colors.py`

- Scans all captured episodes' RGB + semantic mask
- Extracts object pixel region, computes average RGB per viewpoint
- Flags anomalies using **color channel ratio** criteria (R >> G,B = RED_DOMINANT)
- Result: scanned 2,504 episodes, found **only 1 anomalous instance** (570b82c4 ball, 36 episodes)
- Also found 29 VERY_DARK objects, confirmed as legitimately dark-colored (not anomalies)

The 36 episodes containing this single anomalous ball instance were removed from the dataset:
- Loss: 36/2504 = **1.4%** of total episodes
- Ball category: 128 -> 92 episodes
- All other 17 categories and all other object instances unaffected

## Figure for Paper (A.10)

Three-panel comparison figure:

| Panel | Content | Source |
|-------|---------|--------|
| (a) Ground Truth | Original GLB texture (white + black/gold adidas ball) | GLB model viewer screenshot |
| (b) Rendered | Rendered in Habitat with `force_flat_shading=true` — solid red ball | `fig_render_before/original_flat_nolight.png` |
| (c) Detection | Color scan result showing the anomaly was automatically identified | `scan_render_colors.py` output |

Caption should explain:
- The original `force_flat_shading` + `NO_LIGHT_KEY` design ensures lighting-independent rendering, which works correctly for the vast majority of objects
- This specific ball instance's `baseColorFactor` did not match its intended appearance (complex texture discarded by flat shader)
- An automated color channel ratio scan identified it as the only anomaly across 2,504 episodes
- The affected 36 episodes (1.4%) were removed from the final dataset

## Key Files

| File | Purpose |
|------|---------|
| `scan_render_colors.py` | Automated RGB channel ratio scan to detect color anomalies |
| `fix_object_rendering.py` | Emissive self-illumination fix (explored but not used in final dataset) |

The one-off diagnostic scripts used to reach these conclusions (GLB material
inspection, per-shader comparison renders, texture extraction, and a risk scan over
all object GLBs) are not published: their findings are the write-up above, and they
were throwaway tools rather than part of the pipeline. The same applies to the
before/after comparison renders, which survive as panels (b) and (c) of the appendix
figure.

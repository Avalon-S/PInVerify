#!/usr/bin/env python3
"""
Preprocessing: an attempted fix for PIN object colors under Habitat-sim.

The problem:
  PIN injects objects with force_flat_shading=true and NO_LIGHT_KEY.
  Every Habitat shader path distorts the color in its own way:
    - the flat shader reads only baseColorFactor and discards textures
    - the PBR shader picks up scene lighting and environment reflections
    - the KHR_materials_unlit extension is unsupported in some Habitat-sim builds

The approach tried here: the emissive channel
  Copy baseColorTexture into emissiveTexture, set emissiveFactor=[1,1,1],
  and black out baseColorFactor to [0,0,0,1] so PBR contributes nothing.
  PBR then reduces to:
      finalColor = emissive × emissiveTexture + PBR(baseColor=black, lighting)
                 = [1,1,1] × texture + ~0
                 = texture  <- the object's own color
  The emissive channel is independent of all lighting and is core glTF, needing no extension.

Steps:
  Step 1: set force_flat_shading to false in the JSON, keeping the GLB material
  Step 2: the emissive move itself, baseColorTexture into emissiveTexture
  Steps 3-7: material tweaks that minimize the PBR contribution
    - baseColorFactor to [0,0,0,1]
    - metallicFactor → 0.0
    - drop metallicRoughnessTexture
    - roughnessFactor → 1.0
    - drop non-white vertex colors

Usage:
    pip install pygltflib

    # restore the originals first
    python3 fix_object_rendering.py --objects_dir data/datasets/pin/hm3d/v1/objects --restore

    # apply everything (the emissive route is the default)
    python3 fix_object_rendering.py --objects_dir data/datasets/pin/hm3d/v1/objects

    # single file, for testing
    python3 fix_object_rendering.py --objects_dir data/datasets/pin/hm3d/v1/objects \
        --filter 570b82c4391c49ddb1e471e6e55de9f4

    # preview the changes
    python3 fix_object_rendering.py --objects_dir data/datasets/pin/hm3d/v1/objects --dry_run

    # restore the originals
    python3 fix_object_rendering.py --objects_dir data/datasets/pin/hm3d/v1/objects --restore

    # diagnose only: print the material information
    python3 fix_object_rendering.py --objects_dir data/datasets/pin/hm3d/v1/objects --diagnose
"""

import os
import sys
import json
import struct
import shutil
import argparse
from pathlib import Path


# ============================================================
# Step 1: Fix JSON configs
# ============================================================

def fix_json_configs(objects_dir, dry_run=False, name_filter=None):
    """Set force_flat_shading to false in every object_config.json."""
    json_files = sorted(Path(objects_dir).glob("*.object_config.json"))
    if name_filter:
        json_files = [f for f in json_files if name_filter in f.stem]

    modified = 0
    already_ok = 0

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("force_flat_shading") == True:
            if dry_run:
                print(f"  [DRY] Would change force_flat_shading: true → false in {jf.name}")
            else:
                bak = jf.with_suffix(jf.suffix + ".bak")
                if not bak.exists():
                    shutil.copy2(jf, bak)
                data["force_flat_shading"] = False
                with open(jf, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"  Fixed: {jf.name} (force_flat_shading → false)")
            modified += 1
        else:
            already_ok += 1

    print(f"\n[JSON] Total: {len(json_files)}, Modified: {modified}, Already OK: {already_ok}")
    return modified


# ============================================================
# Helpers for vertex color analysis
# ============================================================

def _read_vertex_color_stats(gltf, accessor):
    """Read vertex-color statistics (mean RGB)."""
    bv_idx = accessor.bufferView
    if bv_idx is None:
        return None

    buffer_view = gltf.bufferViews[bv_idx]
    comp_sizes = {
        5120: ('b', 1), 5121: ('B', 1), 5122: ('h', 2),
        5123: ('H', 2), 5126: ('f', 4),
    }
    if accessor.componentType not in comp_sizes:
        return None

    fmt_char, comp_size = comp_sizes[accessor.componentType]
    type_sizes = {'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'SCALAR': 1}
    num_components = type_sizes.get(accessor.type, 4)
    stride = buffer_view.byteStride or (comp_size * num_components)
    offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)

    binary_blob = gltf.binary_blob()
    if binary_blob is None:
        return None

    r_sum, g_sum, b_sum = 0.0, 0.0, 0.0
    count = min(accessor.count, 5000)

    for i in range(count):
        pos = offset + i * stride
        values = []
        for c in range(min(num_components, 3)):
            val_bytes = binary_blob[pos + c * comp_size: pos + (c + 1) * comp_size]
            if len(val_bytes) < comp_size:
                return None
            val = struct.unpack(fmt_char, val_bytes)[0]
            if accessor.componentType == 5121:
                val = val / 255.0
            elif accessor.componentType == 5123:
                val = val / 65535.0
            values.append(val)

        if len(values) >= 3:
            r_sum += values[0]
            g_sum += values[1]
            b_sum += values[2]

    if count == 0:
        return None
    return (r_sum / count, g_sum / count, b_sum / count)


# ============================================================
# Step 2-7: Fix GLB materials
# ============================================================

def fix_glb_materials(objects_dir, dry_run=False, name_filter=None,
                      fix_emissive=True, fix_metallic=True,
                      fix_vtxcolor=True, fix_mr_texture=True, fix_roughness=True,
                      target_metallic=0.0, min_roughness=1.0):
    """
    Apply the full material change to a GLB:
      - the emissive move: baseColorTexture into emissiveTexture
      - baseColorFactor to [0,0,0,1]
      - metallicFactor → 0.0
      - drop metallicRoughnessTexture
      - roughnessFactor → 1.0
      - drop non-white vertex colors
    """
    try:
        from pygltflib import GLTF2
    except ImportError:
        print("[ERROR] pygltflib not installed. Run: pip install pygltflib")
        return 0

    glb_files = sorted(Path(objects_dir).glob("*.glb"))
    if name_filter:
        glb_files = [f for f in glb_files if name_filter in f.stem]

    modified = 0
    already_ok = 0
    errors = 0

    stats = {
        "emissive_set": 0,
        "metallic_fixed": 0,
        "mr_texture_removed": 0,
        "roughness_fixed": 0,
        "vtxcolor_removed": 0,
    }

    for gf in glb_files:
        try:
            gltf = GLTF2().load(str(gf))
        except Exception as e:
            print(f"  [ERROR] Failed to load {gf.name}: {e}")
            errors += 1
            continue

        file_needs_fix = False
        fix_details = []

        # ---- Check emissive (core fix) ----
        if fix_emissive:
            for i, mat in enumerate(gltf.materials):
                pbr = mat.pbrMetallicRoughness
                if pbr is None:
                    continue
                if pbr.baseColorTexture is not None:
                    # Has texture: check if already moved to emissive
                    if mat.emissiveTexture is None:
                        file_needs_fix = True
                        fix_details.append(
                            f"  ★ mat[{i}]: baseColorTexture → emissiveTexture (self-illuminating)")
                    # Also check if baseColorFactor needs zeroing
                    bcf = pbr.baseColorFactor or [1, 1, 1, 1]
                    if bcf[0] > 0.01 or bcf[1] > 0.01 or bcf[2] > 0.01:
                        file_needs_fix = True
                        fix_details.append(
                            f"  ★ mat[{i}]: baseColorFactor → [0,0,0,1] (zero PBR contribution)")
                else:
                    # No texture: move baseColorFactor to emissiveFactor
                    bcf = pbr.baseColorFactor or [1, 1, 1, 1]
                    ef = mat.emissiveFactor or [0, 0, 0]
                    if (bcf[0] > 0.01 or bcf[1] > 0.01 or bcf[2] > 0.01) and \
                       (ef[0] < 0.01 and ef[1] < 0.01 and ef[2] < 0.01):
                        file_needs_fix = True
                        fix_details.append(
                            f"  ★ mat[{i}]: baseColorFactor ({bcf[0]:.2f},{bcf[1]:.2f},{bcf[2]:.2f}) → emissiveFactor")

        # ---- Check metallic ----
        if fix_metallic:
            for i, mat in enumerate(gltf.materials):
                pbr = mat.pbrMetallicRoughness
                if pbr is None:
                    continue
                old_m = pbr.metallicFactor if pbr.metallicFactor is not None else 1.0
                if old_m > target_metallic + 0.01:
                    file_needs_fix = True
                    fix_details.append(f"  mat[{i}]: metallic {old_m:.2f} → {target_metallic:.2f}")

        # ---- Check metallicRoughnessTexture ----
        if fix_mr_texture:
            for i, mat in enumerate(gltf.materials):
                pbr = mat.pbrMetallicRoughness
                if pbr is None:
                    continue
                if pbr.metallicRoughnessTexture is not None:
                    file_needs_fix = True
                    fix_details.append(
                        f"  mat[{i}]: metallicRoughnessTexture (index={pbr.metallicRoughnessTexture.index}) → REMOVED"
                    )

        # ---- Check roughnessFactor ----
        if fix_roughness:
            for i, mat in enumerate(gltf.materials):
                pbr = mat.pbrMetallicRoughness
                if pbr is None:
                    continue
                old_r = pbr.roughnessFactor if pbr.roughnessFactor is not None else 1.0
                if old_r < min_roughness - 0.01:
                    file_needs_fix = True
                    fix_details.append(
                        f"  mat[{i}]: roughness {old_r:.2f} → {min_roughness:.2f}"
                    )

        # ---- Check vertex colors ----
        if fix_vtxcolor:
            for mesh in gltf.meshes:
                for prim in mesh.primitives:
                    color_acc_idx = getattr(prim.attributes, 'COLOR_0', None)
                    if color_acc_idx is not None:
                        accessor = gltf.accessors[color_acc_idx]
                        vtx_stats = _read_vertex_color_stats(gltf, accessor)
                        if vtx_stats:
                            r, g, b = vtx_stats
                            if not (r > 0.95 and g > 0.95 and b > 0.95):
                                file_needs_fix = True
                                fix_details.append(
                                    f"  COLOR_0: avg=({r:.3f},{g:.3f},{b:.3f}) → REMOVED"
                                )
                        else:
                            file_needs_fix = True
                            fix_details.append(f"  COLOR_0: present (unreadable) → REMOVED")

        if file_needs_fix:
            if dry_run:
                print(f"  [DRY] Would fix {gf.name}:")
                for d in fix_details:
                    print(f"    {d}")
            else:
                # Backup
                bak = gf.with_suffix(gf.suffix + ".bak")
                if not bak.exists():
                    shutil.copy2(gf, bak)

                # ★ Apply emissive self-illumination
                if fix_emissive:
                    try:
                        from pygltflib import TextureInfo
                    except ImportError:
                        TextureInfo = None

                    for mat in gltf.materials:
                        pbr = mat.pbrMetallicRoughness
                        if pbr is None:
                            continue

                        if pbr.baseColorTexture is not None:
                            # Copy baseColorTexture reference to emissiveTexture
                            if mat.emissiveTexture is None:
                                if TextureInfo is not None:
                                    mat.emissiveTexture = TextureInfo(
                                        index=pbr.baseColorTexture.index)
                                else:
                                    # Fallback: deep copy
                                    import copy
                                    mat.emissiveTexture = copy.deepcopy(
                                        pbr.baseColorTexture)
                                mat.emissiveFactor = [1.0, 1.0, 1.0]
                                stats["emissive_set"] += 1
                            # Zero out baseColor so PBR doesn't add color
                            pbr.baseColorFactor = [0.0, 0.0, 0.0, 1.0]
                        else:
                            # No texture: move baseColorFactor color to emissive
                            bcf = pbr.baseColorFactor or [1, 1, 1, 1]
                            if bcf[0] > 0.01 or bcf[1] > 0.01 or bcf[2] > 0.01:
                                mat.emissiveFactor = [float(bcf[0]), float(bcf[1]), float(bcf[2])]
                                pbr.baseColorFactor = [0.0, 0.0, 0.0, 1.0]
                                stats["emissive_set"] += 1

                # Apply metallic fix
                if fix_metallic:
                    for mat in gltf.materials:
                        pbr = mat.pbrMetallicRoughness
                        if pbr is None:
                            continue
                        old_m = pbr.metallicFactor if pbr.metallicFactor is not None else 1.0
                        if old_m > target_metallic + 0.01:
                            pbr.metallicFactor = target_metallic
                            stats["metallic_fixed"] += 1

                # Apply metallicRoughnessTexture removal
                if fix_mr_texture:
                    for mat in gltf.materials:
                        pbr = mat.pbrMetallicRoughness
                        if pbr is None:
                            continue
                        if pbr.metallicRoughnessTexture is not None:
                            pbr.metallicRoughnessTexture = None
                            stats["mr_texture_removed"] += 1

                # Apply roughnessFactor fix
                if fix_roughness:
                    for mat in gltf.materials:
                        pbr = mat.pbrMetallicRoughness
                        if pbr is None:
                            continue
                        old_r = pbr.roughnessFactor if pbr.roughnessFactor is not None else 1.0
                        if old_r < min_roughness - 0.01:
                            pbr.roughnessFactor = min_roughness
                            stats["roughness_fixed"] += 1

                # Apply vertex color removal
                if fix_vtxcolor:
                    for mesh in gltf.meshes:
                        for prim in mesh.primitives:
                            color_acc_idx = getattr(prim.attributes, 'COLOR_0', None)
                            if color_acc_idx is not None:
                                accessor = gltf.accessors[color_acc_idx]
                                vtx_stats = _read_vertex_color_stats(gltf, accessor)
                                should_remove = True
                                if vtx_stats:
                                    r, g, b = vtx_stats
                                    if r > 0.95 and g > 0.95 and b > 0.95:
                                        should_remove = False
                                if should_remove:
                                    prim.attributes.COLOR_0 = None
                                    stats["vtxcolor_removed"] += 1

                gltf.save(str(gf))
                print(f"  Fixed: {gf.name}")
                for d in fix_details:
                    print(f"    {d}")
            modified += 1
        else:
            already_ok += 1

    print(f"\n[GLB] Total: {len(glb_files)}, Modified: {modified}, "
          f"Already OK: {already_ok}, Errors: {errors}")
    if not dry_run:
        print(f"  ★ Emissive self-illumination set: {stats['emissive_set']} materials")
        print(f"  Metallic fixes: {stats['metallic_fixed']}")
        print(f"  MetallicRoughnessTexture removals: {stats['mr_texture_removed']}")
        print(f"  Roughness fixes: {stats['roughness_fixed']}")
        print(f"  VertexColor removals: {stats['vtxcolor_removed']}")
    return modified


# ============================================================
# Diagnose
# ============================================================

def diagnose_glb(glb_path):
    """Print one GLB's material details."""
    from pygltflib import GLTF2

    gltf = GLTF2().load(str(glb_path))
    name = Path(glb_path).stem

    issues = []

    # Materials
    for i, mat in enumerate(gltf.materials):
        pbr = mat.pbrMetallicRoughness
        if pbr is None:
            continue

        bcf = pbr.baseColorFactor or [1, 1, 1, 1]
        mf = pbr.metallicFactor if pbr.metallicFactor is not None else 1.0
        rf = pbr.roughnessFactor if pbr.roughnessFactor is not None else 1.0
        has_base_tex = pbr.baseColorTexture is not None
        has_mr_tex = pbr.metallicRoughnessTexture is not None
        has_emissive_tex = mat.emissiveTexture is not None
        ef = mat.emissiveFactor or [0, 0, 0]
        mat_unlit = bool(mat.extensions and "KHR_materials_unlit" in mat.extensions)

        print(f"  {name} mat[{i}]: bcf=({bcf[0]:.3f},{bcf[1]:.3f},{bcf[2]:.3f},{bcf[3]:.3f}) "
              f"metallic={mf:.2f} rough={rf:.2f} hasTex={has_base_tex} hasMRTex={has_mr_tex} "
              f"unlit={mat_unlit} name={mat.name!r}")
        print(f"    emissive: factor=({ef[0]:.2f},{ef[1]:.2f},{ef[2]:.2f}) "
              f"hasTex={has_emissive_tex}")

        # Check for issues
        if not has_emissive_tex and has_base_tex:
            issues.append(f"mat[{i}]: baseColorTexture not copied to emissiveTexture")
        if has_emissive_tex and (bcf[0] > 0.01 or bcf[1] > 0.01 or bcf[2] > 0.01):
            issues.append(f"mat[{i}]: has emissive but baseColorFactor not zeroed ({bcf[0]:.2f},{bcf[1]:.2f},{bcf[2]:.2f})")
        if mf > 0.01:
            issues.append(f"mat[{i}]: metallic={mf:.2f}")
        if has_mr_tex:
            issues.append(f"mat[{i}]: has metallicRoughnessTexture")

    # Vertex colors
    has_vtx_colors = False
    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            color_acc_idx = getattr(prim.attributes, 'COLOR_0', None)
            if color_acc_idx is not None:
                has_vtx_colors = True
                accessor = gltf.accessors[color_acc_idx]
                vtx_stats = _read_vertex_color_stats(gltf, accessor)
                if vtx_stats:
                    r, g, b = vtx_stats
                    print(f"  {name} COLOR_0: avg=({r:.3f},{g:.3f},{b:.3f}) "
                          f"count={accessor.count} type={accessor.type}")
                    if not (r > 0.9 and g > 0.9 and b > 0.9):
                        issues.append(f"COLOR_0 non-white avg=({r:.2f},{g:.2f},{b:.2f})")

    if not has_vtx_colors:
        print(f"  {name}: no vertex colors")

    if issues:
        print(f"  ** Issues: {'; '.join(issues)}")
    else:
        print(f"  ✓ OK (emissive self-illumination configured)")

    return issues


def diagnose_all(objects_dir, name_filter=None):
    """Diagnose material problems across every GLB."""
    try:
        from pygltflib import GLTF2
    except ImportError:
        print("[ERROR] pygltflib not installed. Run: pip install pygltflib")
        return

    glb_files = sorted(Path(objects_dir).glob("*.glb"))
    if name_filter:
        glb_files = [f for f in glb_files if name_filter in f.stem]

    problem_files = []
    for gf in glb_files:
        issues = diagnose_glb(str(gf))
        if issues:
            problem_files.append((gf.stem, issues))

    print(f"\n{'='*60}")
    print(f"Diagnosis complete: {len(glb_files)} GLBs scanned")
    print(f"Files with potential rendering issues: {len(problem_files)}")
    print(f"{'='*60}")

    if problem_files:
        no_emissive = [f for f, issues in problem_files if any("emissive" in i.lower() for i in issues)]
        has_metallic = [f for f, issues in problem_files if any("metallic" in i for i in issues)]
        has_vtxcolor = [f for f, issues in problem_files if any("COLOR_0" in i for i in issues)]

        print(f"\n  Missing emissive setup: {len(no_emissive)} files")
        print(f"  High metallic: {len(has_metallic)} files")
        print(f"  Non-white vertex colors: {len(has_vtxcolor)} files")

        if name_filter:
            print(f"\nDetailed issues for filter '{name_filter}':")
            for fname, issues in problem_files:
                print(f"  {fname}:")
                for issue in issues:
                    print(f"    - {issue}")


# ============================================================
# Restore
# ============================================================

def restore_backups(objects_dir):
    """Restore the originals from their .bak backups."""
    restored = 0
    for bak in Path(objects_dir).glob("*.bak"):
        orig = bak.with_suffix("")
        shutil.copy2(bak, orig)
        bak.unlink()
        print(f"  Restored: {orig.name}")
        restored += 1
    print(f"\nRestored {restored} files")
    return restored


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Fix PIN object rendering in Habitat-sim (emissive self-illumination + PBR fallback fixes)"
    )
    ap.add_argument("--objects_dir", type=str, required=True,
                    help="Path to objects directory (e.g. data/datasets/pin/hm3d/v1/objects)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Preview changes without modifying files")
    ap.add_argument("--restore", action="store_true",
                    help="Restore original files from .bak backups")
    ap.add_argument("--diagnose", action="store_true",
                    help="Only diagnose — print material info, don't modify anything")
    ap.add_argument("--filter", type=str, default=None,
                    help="Only process GLBs whose filename contains this string (e.g. object_id)")
    ap.add_argument("--metallic", type=float, default=0.0,
                    help="Target metallicFactor (default: 0.0)")
    ap.add_argument("--min_roughness", type=float, default=1.0,
                    help="Minimum roughnessFactor (default: 1.0, fully diffuse)")

    # Step toggles
    ap.add_argument("--skip_json", action="store_true",
                    help="Skip JSON config fix")
    ap.add_argument("--skip_emissive", action="store_true",
                    help="Skip emissive self-illumination (NOT recommended)")
    ap.add_argument("--skip_metallic", action="store_true",
                    help="Skip metallicFactor fix")
    ap.add_argument("--skip_vtxcolor", action="store_true",
                    help="Skip vertex color removal")
    ap.add_argument("--skip_mr_texture", action="store_true",
                    help="Skip metallicRoughnessTexture removal")
    ap.add_argument("--skip_roughness", action="store_true",
                    help="Skip roughnessFactor fix")
    args = ap.parse_args()

    if not os.path.isdir(args.objects_dir):
        print(f"[ERROR] Directory not found: {args.objects_dir}")
        sys.exit(1)

    if args.restore:
        print(f"Restoring backups in {args.objects_dir}...")
        restore_backups(args.objects_dir)
        return

    if args.diagnose:
        print(f"Diagnosing GLBs in {args.objects_dir}...")
        diagnose_all(args.objects_dir, args.filter)
        return

    mode = "DRY RUN" if args.dry_run else "FIXING"
    print(f"{'='*60}")
    print(f"[{mode}] Objects dir: {args.objects_dir}")
    if args.filter:
        print(f"  Filter: {args.filter}")
    print(f"  Steps: JSON={'skip' if args.skip_json else 'fix'}, "
          f"emissive={'skip' if args.skip_emissive else '★ SET'}, "
          f"metallic={'skip' if args.skip_metallic else 'fix'}, "
          f"mrTexture={'skip' if args.skip_mr_texture else 'remove'}, "
          f"roughness={'skip' if args.skip_roughness else f'min={args.min_roughness}'}, "
          f"vtxColor={'skip' if args.skip_vtxcolor else 'fix'}")
    print(f"{'='*60}")

    total_fixed = 0

    # Step 1: JSON
    if not args.skip_json:
        print(f"\n--- Step 1: Fix object_config.json (force_flat_shading → false) ---")
        total_fixed += fix_json_configs(args.objects_dir, args.dry_run, args.filter)

    # Steps 2-7: GLB
    all_glb_skipped = (args.skip_emissive and args.skip_metallic
                       and args.skip_vtxcolor and args.skip_mr_texture and args.skip_roughness)
    if not all_glb_skipped:
        steps = []
        if not args.skip_emissive:
            steps.append("★ emissive self-illumination")
        if not args.skip_metallic:
            steps.append("metallicFactor→0")
        if not args.skip_mr_texture:
            steps.append("remove metallicRoughnessTexture")
        if not args.skip_roughness:
            steps.append(f"roughness→{args.min_roughness}")
        if not args.skip_vtxcolor:
            steps.append("remove COLOR_0")
        print(f"\n--- Steps 2-7: Fix GLB ({', '.join(steps)}) ---")
        total_fixed += fix_glb_materials(
            args.objects_dir,
            dry_run=args.dry_run,
            name_filter=args.filter,
            fix_emissive=not args.skip_emissive,
            fix_metallic=not args.skip_metallic,
            fix_vtxcolor=not args.skip_vtxcolor,
            fix_mr_texture=not args.skip_mr_texture,
            fix_roughness=not args.skip_roughness,
            target_metallic=args.metallic,
            min_roughness=args.min_roughness,
        )

    print(f"\n{'='*60}")
    if args.dry_run:
        print(f"[DRY RUN] Would fix {total_fixed} files. Run without --dry_run to apply.")
    else:
        print(f"[DONE] Fixed {total_fixed} files. Backups saved as .bak")
        print(f"  To restore: python3 {__file__} --objects_dir {args.objects_dir} --restore")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

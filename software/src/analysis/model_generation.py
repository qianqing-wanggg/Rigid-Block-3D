# scripts to run
#1. 00_create_wall_beam_ground.py
#2. 00_create_mortar_surface_09.py
#3. 00_create_mortar_gmsh.sh
#4. 00_generate_contps.py
#5. 00_generate_ss_contp.py

import numpy as np
from stl import mesh
import os
import json
from typing import List, Tuple, Optional


def generate_cube_mesh(
    dimensions: Tuple[float, float, float],
    center: Tuple[float, float, float]
) -> mesh.Mesh:
    """
    Generate a 3D cube mesh with specified dimensions and center position.
    
    Args:
        dimensions: Tuple of (length_x, length_y, length_z)
        center: Tuple of (center_x, center_y, center_z)
    
    Returns:
        mesh.Mesh: The generated STL mesh object
    """
    # Define vertices of a unit cube (from -1 to +1)
    vertices = np.array([
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1]
    ])
    
    # Define the 12 triangles composing the cube
    faces = np.array([
        [0, 3, 1],
        [1, 3, 2],
        [0, 4, 7],
        [0, 7, 3],
        [4, 5, 6],
        [4, 6, 7],
        [5, 1, 2],
        [5, 2, 6],
        [2, 3, 6],
        [3, 7, 6],
        [0, 1, 5],
        [0, 5, 4]
    ])
    
    # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j], :]
    
    # Scale the cube (divide by 2 because base cube goes from -1 to +1)
    cube.x = cube.x * dimensions[0] / 2
    cube.y = cube.y * dimensions[1] / 2
    cube.z = cube.z * dimensions[2] / 2
    
    # Translate the cube to the specified center
    cube.x = cube.x + center[0]
    cube.y = cube.y + center[1]
    cube.z = cube.z + center[2]
    
    return cube


def generate_cubes_from_json(
    json_path: str,
    output_dir: Optional[str] = None,
    save_files: bool = True
) -> dict:
    """
    Generate cube meshes from a JSON geometry file.
    
    Args:
        json_path: Path to the geometry JSON file
        output_dir: Directory to save STL files (optional, defaults to same dir as JSON)
        save_files: Whether to save the STL files to disk
    
    Returns:
        dict: Dictionary containing the generated mesh objects with keys 'beam', 'ground', 'wall'
    """
    # Load geometry specifications
    with open(json_path, 'r') as f:
        geometry_file = json.load(f)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(json_path), 'wall')
    
    # Create output directory if saving files
    if save_files and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Dictionary to store generated meshes
    generated_meshes = {}
    
    # Define the objects to generate
    objects_to_generate = ['beam', 'ground', 'wall']
    
    for obj_name in objects_to_generate:
        # Extract dimensions and center from JSON
        dimensions = (
            float(geometry_file[f'{obj_name}_dim_x']),
            float(geometry_file[f'{obj_name}_dim_y']),
            float(geometry_file[f'{obj_name}_dim_z'])
        )
        center = tuple(geometry_file[f'{obj_name}_center'])
        
        # Generate the mesh
        cube_mesh = generate_cube_mesh(dimensions, center)
        generated_meshes[obj_name] = cube_mesh
        
        # Save to file if requested
        if save_files:
            output_path = os.path.join(output_dir, f'{obj_name}_origin.stl')
            cube_mesh.save(output_path)
            print(f"Saved {obj_name} mesh to: {output_path}")
    
    return generated_meshes


import numpy as np
import pymeshlab
import glob
from typing import List, Union, Optional, Dict
import os


def apply_rotation_to_mesh(
    ms: pymeshlab.MeshSet,
    rotation_deg: Tuple[float, float, float]
) -> None:
    """Apply rotation (Rx, Ry, Rz) to a MeshSet in place."""
    if rotation_deg[0] != 0:
        ms.compute_matrix_from_rotation(
            rotaxis='X axis', rotcenter='origin', angle=rotation_deg[0]
        )
        #ms.apply_matrix_freeze()

    if rotation_deg[1] != 0:
        ms.compute_matrix_from_rotation(
            rotaxis='Y axis', rotcenter='origin', angle=rotation_deg[1]
        )
        #ms.apply_matrix_freeze()

    if rotation_deg[2] != 0:
        ms.compute_matrix_from_rotation(
            rotaxis='Z axis', rotcenter='origin', angle=rotation_deg[2]
        )
        #ms.apply_matrix_freeze()


def apply_translation_to_mesh(
    ms: pymeshlab.MeshSet,
    translation: Tuple[float, float, float]
) -> None:
    """Apply translation (tx, ty, tz) to a MeshSet in place."""
    if any(t != 0 for t in translation):
        ms.compute_matrix_from_translation_rotation_scale(
            translationx =translation[0],
            translationy=translation[1],
            translationz=translation[2]
        )
        #ms.apply_matrix_freeze()


def compute_global_min_after_rotation(
    stone_mesh_paths: List[str],
    stone_transforms: Optional[Dict[str, dict]] = None
) -> Tuple[float, float, float]:
    """
    Compute the global minimum coordinates across all stones after applying rotation.

    Args:
        stone_mesh_paths: List of paths to stone mesh files
        stone_transforms: Optional dict mapping stone filename to transform settings

    Returns:
        Tuple of (min_x, min_y, min_z) across all stones after rotation
    """
    global_min = [float('inf'), float('inf'), float('inf')]

    for stone_path in stone_mesh_paths:
        if not os.path.exists(stone_path):
            continue

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(stone_path)

        # Apply rotation if specified
        stone_filename = os.path.basename(stone_path)
        if stone_transforms and stone_filename in stone_transforms:
            transform = stone_transforms[stone_filename]
            rotation_deg = transform.get('rotation', (0, 0, 0))
            apply_rotation_to_mesh(ms, rotation_deg)

        # Update global min
        bb = ms.current_mesh().bounding_box()
        min_coords = bb.min()
        global_min[0] = min(global_min[0], min_coords[0])
        global_min[1] = min(global_min[1], min_coords[1])
        global_min[2] = min(global_min[2], min_coords[2])

    return tuple(global_min)


def compute_stones_bounding_box(
    stone_mesh_paths: List[str],
    stone_transforms: Optional[Dict[str, dict]] = None
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute the overall bounding box of all stones after applying all transforms
    (rotation, auto-translate to origin, custom translation).

    Args:
        stone_mesh_paths: List of paths to stone mesh files
        stone_transforms: Optional dict mapping stone filename to transform settings

    Returns:
        Tuple of ((min_x, min_y, min_z), (max_x, max_y, max_z))
    """
    # First pass: compute global min after rotation (needed for auto-translate)
    auto_translate_offset = None
    if stone_transforms:
        any_auto_origin = any(
            t.get('auto_origin', False)
            for t in stone_transforms.values()
        )
        if any_auto_origin:
            global_min_after_rot = compute_global_min_after_rotation(stone_mesh_paths, stone_transforms)
            auto_translate_offset = (-global_min_after_rot[0], -global_min_after_rot[1], -global_min_after_rot[2])

    # Second pass: apply all transforms and compute bounding box
    global_min = [float('inf'), float('inf'), float('inf')]
    global_max = [float('-inf'), float('-inf'), float('-inf')]

    for stone_path in stone_mesh_paths:
        if not os.path.exists(stone_path):
            continue

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(stone_path)

        stone_filename = os.path.basename(stone_path)
        if stone_transforms and stone_filename in stone_transforms:
            transform = stone_transforms[stone_filename]

            # Step 1: Apply rotation
            rotation_deg = transform.get('rotation', (0, 0, 0))
            apply_rotation_to_mesh(ms, rotation_deg)

            # Step 2: Apply auto-translate to origin (if enabled for this stone)
            if transform.get('auto_origin', False) and auto_translate_offset is not None:
                apply_translation_to_mesh(ms, auto_translate_offset)

            # Step 3: Apply custom translation
            custom_translation = transform.get('translation', (0, 0, 0))
            apply_translation_to_mesh(ms, custom_translation)

        # Update global bounds
        bb = ms.current_mesh().bounding_box()
        min_coords = bb.min()
        max_coords = bb.max()
        for i in range(3):
            global_min[i] = min(global_min[i], min_coords[i])
            global_max[i] = max(global_max[i], max_coords[i])

    return (tuple(global_min), tuple(global_max))


def compute_wall_dimensions_from_stones(
    stone_mesh_paths: List[str],
    stone_transforms: Optional[Dict[str, dict]] = None,
    method: str = "bounding_box"
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute wall dimensions and center from stone meshes.

    Args:
        stone_mesh_paths: List of paths to stone mesh files
        stone_transforms: Optional dict mapping stone filename to transform settings
        method: Method to use:
            - "bounding_box": Use bounding box of all stones for all dimensions and center
            - "barycenter_yz": Use bounding box for X, barycenter bounds for Y and Z;
                               Center: X from vertex bbox center, Y/Z from barycenter bbox center

    Returns:
        Tuple of ((wall_dim_x, wall_dim_y, wall_dim_z), (wall_center_x, wall_center_y, wall_center_z))
    """
    if not stone_mesh_paths:
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    # Compute overall bounding box
    global_min, global_max = compute_stones_bounding_box(stone_mesh_paths, stone_transforms)

    # Vertex bounding box center
    vertex_center_x = (global_min[0] + global_max[0]) / 2.0
    vertex_center_y = (global_min[1] + global_max[1]) / 2.0
    vertex_center_z = (global_min[2] + global_max[2]) / 2.0

    if method == "bounding_box":
        # Option 1: All dimensions and center from bounding box
        wall_dim_x = global_max[0] - global_min[0]
        wall_dim_y = global_max[1] - global_min[1]
        wall_dim_z = global_max[2] - global_min[2]

        wall_center_x = vertex_center_x
        wall_center_y = vertex_center_y
        wall_center_z = vertex_center_z

    elif method == "barycenter_yz":
        # Option 2: X from bounding box, Y and Z from barycenter bounds
        # Center: X from vertex bbox, Y/Z from barycenter bbox
        wall_dim_x = global_max[0] - global_min[0]

        # First, compute auto-translate offset (needed for transforms)
        auto_translate_offset = None
        if stone_transforms:
            any_auto_origin = any(
                t.get('auto_origin', False)
                for t in stone_transforms.values()
            )
            if any_auto_origin:
                global_min_after_rot = compute_global_min_after_rotation(stone_mesh_paths, stone_transforms)
                auto_translate_offset = (-global_min_after_rot[0], -global_min_after_rot[1], -global_min_after_rot[2])

        # Compute barycenters of all stones after applying all transforms
        barycenters = []
        for stone_path in stone_mesh_paths:
            if not os.path.exists(stone_path):
                continue

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(stone_path)

            stone_filename = os.path.basename(stone_path)
            if stone_transforms and stone_filename in stone_transforms:
                transform = stone_transforms[stone_filename]

                # Step 1: Apply rotation
                rotation_deg = transform.get('rotation', (0, 0, 0))
                apply_rotation_to_mesh(ms, rotation_deg)

                # Step 2: Apply auto-translate to origin (if enabled for this stone)
                if transform.get('auto_origin', False) and auto_translate_offset is not None:
                    apply_translation_to_mesh(ms, auto_translate_offset)

                # Step 3: Apply custom translation
                custom_translation = transform.get('translation', (0, 0, 0))
                apply_translation_to_mesh(ms, custom_translation)

            # Compute barycenter after all transforms
            bb = ms.current_mesh().bounding_box()
            center = bb.center()
            barycenters.append(center)

        if barycenters:
            # Compute bounding box of barycenters
            bary_min_y = min(c[1] for c in barycenters)
            bary_max_y = max(c[1] for c in barycenters)
            bary_min_z = min(c[2] for c in barycenters)
            bary_max_z = max(c[2] for c in barycenters)

            wall_dim_y = bary_max_y - bary_min_y
            wall_dim_z = bary_max_z - bary_min_z

            # Center: X from vertex bbox, Y/Z from barycenter bbox
            wall_center_x = vertex_center_x
            wall_center_y = (bary_min_y + bary_max_y) / 2.0
            wall_center_z = (bary_min_z + bary_max_z) / 2.0
        else:
            wall_dim_y = global_max[1] - global_min[1]
            wall_dim_z = global_max[2] - global_min[2]
            wall_center_x = vertex_center_x
            wall_center_y = vertex_center_y
            wall_center_z = vertex_center_z
    else:
        raise ValueError(f"Unknown method: {method}")

    return ((wall_dim_x, wall_dim_y, wall_dim_z), (wall_center_x, wall_center_y, wall_center_z))


def preprocess_stone_mesh(
    mesh_path: str,
    rotation_deg: Tuple[float, float, float] = (0, 0, 0),
    auto_translate_offset: Optional[Tuple[float, float, float]] = None,
    translation: Tuple[float, float, float] = (0, 0, 0),
    output_path: Optional[str] = None
) -> pymeshlab.MeshSet:
    """
    Preprocess a stone mesh with sequential transformations.

    Order of operations:
    1. Rotation (Rx, Ry, Rz) around origin
    2. Auto-translate using pre-computed global offset (same for all stones)
    3. Custom translation (tx, ty, tz)

    Args:
        mesh_path: Input mesh file path
        rotation_deg: (Rx, Ry, Rz) rotation in degrees
        auto_translate_offset: Pre-computed offset to translate all stones by the same amount
            (typically negative of global min coords so wall min is at origin)
        translation: (tx, ty, tz) custom translation after auto-translate
        output_path: Optional path to save result

    Returns:
        pymeshlab.MeshSet with transformed mesh
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)

    # Step 1: Rotation (Rx, Ry, Rz)
    apply_rotation_to_mesh(ms, rotation_deg)

    # Step 2: Auto-translate using global offset (same for all stones)
    if auto_translate_offset is not None:
        apply_translation_to_mesh(ms, auto_translate_offset)

    # Step 3: Custom translation
    apply_translation_to_mesh(ms, translation)

    if output_path:
        ms.save_current_mesh(output_path)

    return ms


def merge_wall_components(
    beam_mesh_path: str,
    ground_mesh_path: str,
    wall_mesh_path: str,
    stone_mesh_paths: List[str],
    output_path: Optional[str] = None,
    process_stone_normals: bool = True,
    simplify_stones: bool = False,
    simplification_ratio: float = 0.2,
    stone_transforms: Optional[Dict[str, dict]] = None,
    temp_dir: str = "temp"
) -> pymeshlab.MeshSet:
    """
    Merge beam, ground, wall, and multiple stone meshes into a single combined mesh.
    
    Args:
        beam_mesh_path: Path to the beam mesh file
        ground_mesh_path: Path to the ground mesh file
        wall_mesh_path: Path to the wall mesh file
        stone_mesh_paths: List of paths to stone mesh files
        output_path: Optional path to save the merged mesh
        process_stone_normals: Whether to reorient and invert stone face normals
        simplify_stones: Whether to simplify stone meshes
        simplification_ratio: Target percentage for mesh simplification (0.0 to 1.0)
        stone_transforms: Optional dict mapping stone filename to transform settings
            Each entry: {'rotation': (rx, ry, rz), 'auto_origin': bool, 'translation': (tx, ty, tz)}

    Returns:
        pymeshlab.MeshSet: MeshSet containing the final merged mesh
    """

    # Initialize final MeshSet
    final_ms = pymeshlab.MeshSet()
    
    print("Loading structural components...")
    
    # Load wall
    final_ms.load_new_mesh(wall_mesh_path)
    print(f"  - Loaded wall from: {wall_mesh_path}")
    
    # # Load beam
    # final_ms.load_new_mesh(beam_mesh_path)
    # print(f"  - Loaded beam from: {beam_mesh_path}")
    
    # # Load ground
    # final_ms.load_new_mesh(ground_mesh_path)
    # print(f"  - Loaded ground from: {ground_mesh_path}")
    
    # Process and add stones
    print(f"\nProcessing {len(stone_mesh_paths)} stones...")

    # Check if auto_origin is enabled for any stone and compute global min
    auto_translate_offset = None
    if stone_transforms:
        # Check if any stone has auto_origin enabled
        any_auto_origin = any(
            t.get('auto_origin', False)
            for t in stone_transforms.values()
        )
        if any_auto_origin:
            print("  Computing global bounding box after rotation...")
            global_min = compute_global_min_after_rotation(stone_mesh_paths, stone_transforms)
            auto_translate_offset = (-global_min[0], -global_min[1], -global_min[2])
            print(f"  Global min: {global_min}, auto-translate offset: {auto_translate_offset}")

    for i, stone_path in enumerate(stone_mesh_paths):
        if not os.path.exists(stone_path):
            print(f"  Warning: Stone file not found: {stone_path}")
            continue

        print(f"  Processing stone {i+1}/{len(stone_mesh_paths)}: {os.path.basename(stone_path)}")

        # Apply preprocessing transforms if specified
        stone_filename = os.path.basename(stone_path)
        if stone_transforms and stone_filename in stone_transforms:
            transform = stone_transforms[stone_filename]
            # Only apply auto_translate_offset if this stone has auto_origin enabled
            stone_auto_offset = auto_translate_offset if transform.get('auto_origin', False) else None
            temp_ms = preprocess_stone_mesh(
                mesh_path=stone_path,
                rotation_deg=transform.get('rotation', (0, 0, 0)),
                auto_translate_offset=stone_auto_offset,
                translation=transform.get('translation', (0, 0, 0))
            )
            print(f"    - Applied transforms: rot={transform.get('rotation')}, auto_offset={stone_auto_offset}, trans={transform.get('translation')}")
        else:
            # Load stone in a temporary MeshSet for processing
            temp_ms = pymeshlab.MeshSet()
            temp_ms.load_new_mesh(stone_path)

        # Optional: Simplify mesh
        if simplify_stones:
            temp_ms.apply_filter(
                'meshing_decimation_quadric_edge_collapse',
                targetperc=simplification_ratio
            )
            print(f"    - Simplified to {simplification_ratio*100}%")
        
        # Optional: Process normals
        if process_stone_normals:
            # Compute face normals
            temp_ms.compute_normal_per_face()
            
            # Re-orient faces by geometry
            temp_ms.meshing_re_orient_faces_by_geometry()
            
            # Invert face orientation
            temp_ms.meshing_invert_face_orientation(forceflip=True)
            
            # Compute vertex normals
            temp_ms.compute_normal_per_vertex()
            
            print(f"    - Processed normals")
        
        # Get the current mesh and add it to final MeshSet
        # Save to temporary file and load into final_ms
        stone_id = stone_filename.split("stone_")[1].split(".")[0]
        stone_id = int(stone_id)
        temp_file = temp_dir+f"/_temp_stone_{stone_id}.ply"
        temp_ms.save_current_mesh(temp_file, save_vertex_normal=True)
        final_ms.load_new_mesh(temp_file)
        
        # # Clean up temporary file
        # if os.path.exists(temp_file):
        #     os.remove(temp_file)
    
    # Merge all meshes into a single mesh
    print("\nMerging all meshes into one...")
    final_ms.apply_filter('generate_by_merging_visible_meshes')
    print(f"  - Successfully merged {len(stone_mesh_paths) + 3} meshes")
    
    # Optional: Save the merged mesh
    if output_path:
        final_ms.save_current_mesh(output_path)
        print(f"\nSaved merged mesh to: {output_path}")
    
    return final_ms


import os
import glob
from typing import List, Optional, Dict
import pymeshlab
from stl import mesh
import json


def process_wall_assembly(
    json_path: str,
    stone_mesh_paths: List[str],
    output_path: Optional[str] = None,
    temp_dir: str = "temp",
    process_stone_normals: bool = True,
    simplify_stones: bool = False,
    simplification_ratio: float = 0.2,
    stone_transforms: Optional[Dict[str, dict]] = None
) -> pymeshlab.MeshSet:
    """
    Complete pipeline: Generate beam/ground/wall cubes from JSON and merge with stones.
    Temporary cube files are saved in temp_dir and NOT cleaned up.
    
    Args:
        json_path: Path to the geometry JSON file
        stone_mesh_paths: List of paths to stone mesh files
        output_path: Optional path to save the final merged mesh
        temp_dir: Directory to store temporary cube files (default: "temp")
        process_stone_normals: Whether to reorient and invert stone face normals
        simplify_stones: Whether to simplify stone meshes
        simplification_ratio: Target percentage for mesh simplification (0.0 to 1.0)
        stone_transforms: Optional dict mapping stone filename to transform settings
            Each entry: {'rotation': (rx, ry, rz), 'auto_origin': bool, 'translation': (tx, ty, tz)}

    Returns:
        pymeshlab.MeshSet: MeshSet containing the final merged mesh

    Note:
        Temporary cube files (beam_origin.stl, ground_origin.stl, wall_origin.stl)
        are saved in temp_dir and are NOT automatically deleted.
    """

    print("="*60)
    print("WALL ASSEMBLY PIPELINE")
    print("="*60)
    
    # Step 1: Create temp directory if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        print(f"\nCreated temporary directory: {temp_dir}")
    else:
        print(f"\nUsing temporary directory: {temp_dir}")
    
    # Step 2: Generate cube meshes from JSON
    print("\n" + "-"*60)
    print("STEP 1: Generating structural cubes from JSON")
    print("-"*60)
    
    cube_meshes = generate_cubes_from_json(
        json_path=json_path,
        output_dir=temp_dir,
        save_files=True
    )
    
    # Define paths to the generated cube files
    beam_path = os.path.join(temp_dir, 'beam_origin.stl')
    ground_path = os.path.join(temp_dir, 'ground_origin.stl')
    wall_path = os.path.join(temp_dir, 'wall_origin.stl')
    
    # Verify files were created
    for name, path in [('beam', beam_path), ('ground', ground_path), ('wall', wall_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Failed to generate {name} cube at {path}")
    
    print(f"\n✓ Successfully generated 3 structural cubes in {temp_dir}/")
    
    # Step 3: Merge all components
    print("\n" + "-"*60)
    print("STEP 2: Merging structural cubes with stones")
    print("-"*60)
    
    merged_mesh = merge_wall_components(
        beam_mesh_path=beam_path,
        ground_mesh_path=ground_path,
        wall_mesh_path=wall_path,
        stone_mesh_paths=stone_mesh_paths,
        output_path=output_path,
        process_stone_normals=process_stone_normals,
        simplify_stones=simplify_stones,
        simplification_ratio=simplification_ratio,
        stone_transforms=stone_transforms,temp_dir=temp_dir
    )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Temporary cube files preserved in: {temp_dir}/")
    print(f"  - {beam_path}")
    print(f"  - {ground_path}")
    print(f"  - {wall_path}")
    if output_path:
        print(f"Final merged mesh saved to: {output_path}")
    
    return merged_mesh


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <folder_id>")
        sys.exit(1)
    
    folder_id = sys.argv[1]
    
    # Define paths
    json_path = f'../data/data_{folder_id}/geometry.json'
    stone_paths = sorted(glob.glob(f'../data/data_{folder_id}/3d/tri/stone_*.ply'))
    output_path = f'../data/data_{folder_id}/mortar.ply'
    temp_dir = f'../data/data_{folder_id}/temp'
    
    print(f"JSON file: {json_path}")
    print(f"Found {len(stone_paths)} stone files")
    print(f"Output: {output_path}")
    print(f"Temp directory: {temp_dir}")
    
    # Run the complete pipeline
    merged_mesh = process_wall_assembly(
        json_path=json_path,
        stone_mesh_paths=stone_paths,
        output_path=output_path,
        temp_dir=temp_dir,
        process_stone_normals=True,
        simplify_stones=False
    )
    
    print("\nDone! Temporary files are kept in temp directory for inspection.")

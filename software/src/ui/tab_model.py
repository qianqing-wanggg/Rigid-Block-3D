import streamlit as st
import json
import os
import glob
from utils.mesh_preview import preview_meshes, preview_single_mesh

def render(refresh_token: int = 0):
    """Render the Model Generation tab"""
    #st.header("Model Generation")
    
    # Create two columns: left for parameters, right for preview
    col_params, col_preview = st.columns([1, 1])
    
    with col_params:
        st.subheader("📋 Geometry")
        
        # === FILE UPLOADS ===
        st.write("**Stones**")
        st.info("""
        You may run the analysis **without stone blocks**.
        In this case, only mortar, beam, and ground are considered.
        """)
        
        # Stone mesh files upload
        stone_files = st.file_uploader(
            "Stone Geometry Files (OBJ/PLY)",
            type=['obj', 'ply'],
            accept_multiple_files=True,
            key='stone_files',
            help="Upload stone_*.ply or similar files"
        )

        # Stone transformation controls
        stone_transforms = {}

        if stone_files:
            with st.expander("Stone Transformations", expanded=False):
                st.caption("Transforms applied in order: Rotation -> Auto-translate to origin -> Custom translation")

                # Batch controls
                apply_to_all = st.checkbox("Apply same transform to all stones", value=True, key="batch_transform")

                if apply_to_all:
                    # Single set of controls for all stones
                    st.write("**All Stones**")

                    st.write("Rotation (degrees)")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        rx_all = st.number_input("Rx", value=0.0, step=1.0, key="rx_all")
                    with col2:
                        ry_all = st.number_input("Ry", value=0.0, step=1.0, key="ry_all")
                    with col3:
                        rz_all = st.number_input("Rz", value=0.0, step=1.0, key="rz_all")

                    auto_origin_all = st.checkbox("Auto-translate to origin (min=0,0,0)", value=True, key="auto_origin_all")

                    st.write("Custom Translation (m)")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        tx_all = st.number_input("X", value=0.0, step=0.001, format="%.3f", key="tx_all")
                    with col2:
                        ty_all = st.number_input("Y", value=0.0, step=0.001, format="%.3f", key="ty_all")
                    with col3:
                        tz_all = st.number_input("Z", value=0.0, step=0.001, format="%.3f", key="tz_all")

                    # Apply to all stones
                    for stone_file in stone_files:
                        stone_transforms[stone_file.name] = {
                            'rotation': (rx_all, ry_all, rz_all),
                            'auto_origin': auto_origin_all,
                            'translation': (tx_all, ty_all, tz_all)
                        }
                else:
                    # Individual controls per stone
                    for i, stone_file in enumerate(stone_files):
                        with st.container():
                            st.write(f"**{stone_file.name}**")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                rx = st.number_input("Rx (deg)", value=0.0, step=1.0, key=f"rx_{i}")
                            with col2:
                                ry = st.number_input("Ry (deg)", value=0.0, step=1.0, key=f"ry_{i}")
                            with col3:
                                rz = st.number_input("Rz (deg)", value=0.0, step=1.0, key=f"rz_{i}")

                            auto_origin = st.checkbox("Auto-translate to origin", value=True, key=f"auto_{i}")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                tx = st.number_input("Tx (m)", value=0.0, step=0.001, format="%.3f", key=f"tx_{i}")
                            with col2:
                                ty = st.number_input("Ty (m)", value=0.0, step=0.001, format="%.3f", key=f"ty_{i}")
                            with col3:
                                tz = st.number_input("Tz (m)", value=0.0, step=0.001, format="%.3f", key=f"tz_{i}")

                            stone_transforms[stone_file.name] = {
                                'rotation': (rx, ry, rz),
                                'auto_origin': auto_origin,
                                'translation': (tx, ty, tz)
                            }
                            st.divider()

        #st.divider()
        
        # === LOAD/SAVE CONFIG SECTION ===
        #st.write("**Configuration**")
        
        #col_load, col_save = st.columns(2)
        
        # with col_load:
        #     uploaded_config = st.file_uploader(
        #         "📂 Load Config",
        #         type=['json'],
        #         key='config_upload',
        #         help="Upload a previously saved configuration"
        #     )
        
        # Handle loaded config
        default_values = {}
        # if uploaded_config:
        #     try:
        #         loaded_config = json.load(uploaded_config)
        #         # Handle both old format (flat) and new format (nested)
        #         if 'geometry' in loaded_config:
        #             # New format: merge geometry and material
        #             default_values = {**loaded_config['geometry']}
        #         else:
        #             # Old format: use as-is
        #             default_values = loaded_config
        #         st.success("✅ Configuration loaded and applied!")
        #     except Exception as e:
        #         st.error(f"❌ Error loading config: {str(e)}")
        
        # Helper function to get value with fallback
        def get_value(key, fallback):
            return default_values.get(key, fallback)
        
        #st.divider()
        
        # === GEOMETRY PARAMETERS ===
        #with st.expander("📐 Wall Geometry", expanded=True):
            
        # Wall dimensions and center
        st.write("**Wall**")
        col1, col2, col3 = st.columns(3)
        with col1:
            wall_dim_x = st.number_input("Wall size X (m)", value=get_value('wall_dim_x', 0.015), step=0.1, format="%.3f", key="wall_x")
        with col2:
            wall_dim_y = st.number_input("Wall size Y (m)", value=get_value('wall_dim_y', 0.05), step=0.1, format="%.3f", key="wall_y")
        with col3:
            wall_dim_z = st.number_input("Wall size Z (m)", value=get_value('wall_dim_z', 0.1), step=0.1, format="%.3f", key="wall_z")
        st.session_state.wall_dim_x = wall_dim_x
        st.session_state.wall_dim_y = wall_dim_y
        st.session_state.wall_dim_z = wall_dim_z
            
        #process to get beam and ground dimensions
        # =========================
        # Auto-derive geometry
        # =========================
        # Coordinate convention: wall box spans
        #   x in [0, wall_dim_x], y in [0, wall_dim_y], z in [0, wall_dim_z]
        wall_center_x = wall_dim_x / 2.0
        wall_center_y = wall_dim_y / 2.0
        wall_center_z = wall_dim_z / 2.0

        # Beam and ground: same cross-section as wall, thin in X
        # (matches your previous pattern: beam_dim_z and ground_dim_z = 0.1)
        # You can tweak these ratios if needed.
        support_thickness_x = 0.01  # e.g., 10% of wall length, min 0.05m
        support_dim_y = wall_dim_y
        support_dim_z = wall_dim_z

        beam_dim_x = support_thickness_x
        beam_dim_y = support_dim_y
        beam_dim_z = support_dim_z

        ground_dim_x = support_thickness_x
        ground_dim_y = support_dim_y
        ground_dim_z = support_dim_z

        beam_center_x = wall_dim_x + support_thickness_x / 2.0
        beam_center_y = wall_center_y
        beam_center_z = wall_center_z

        ground_center_x = 0-support_thickness_x / 2.0
        ground_center_y = wall_center_y
        ground_center_z = wall_center_z

        # Planes/bounds used by util_meshing.py (it uses wall_plane_xs/ys/zs as bounds)
        wall_plane_xs = [0.0, wall_dim_x]
        wall_plane_ys = [0.0, wall_dim_y]
        wall_plane_zs = [0.0, wall_dim_z]

        # Also needed by util_meshing.py:
        Wall_height = wall_dim_x       # util_meshing treats "height" along X
        Wall_diagonal = (wall_dim_x**2 + wall_dim_y**2 + wall_dim_z**2) ** 0.5




        #st.divider()
            
        # === PROCESSING OPTIONS ===
        with st.expander("⚙️ Options", expanded=False):
            process_stone_normals = st.checkbox(
                "Process Stone Normals",
                value=True,
                help="Reorient and invert stone face normals"
            )
            
            simplify_stones = st.checkbox(
                "Simplify Stone Meshes",
                value=False,
                help="Simplify a mesh using a quadric based edge-collapse strategy"
            )
            
            if simplify_stones:
                simplification_ratio = st.slider(
                    "Simplification Ratio",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.2,
                    step=0.05,
                    help="Target percentage of original mesh complexity"
                )
            else:
                simplification_ratio = 0.2
        
        #st.divider()
        
        # === OUTPUT SETTINGS ===
        #with st.expander("💾 Output Settings", expanded=False):
            output_filename = st.text_input(
                "Output Filename",
                value="mortar.ply",
                help="Name for the merged output file"
            )
        
        st.divider()
        
        # === SAVE CONFIG BUTTON ===
        # with col_save:
        #     if st.button("💾 Save Config", use_container_width=True):
        #         st.session_state.show_download = True
        
        if st.session_state.get('show_download', False):
            # Create combined config with both geometry and material
            combined_config = {
                "geometry": create_geometry_json(
                    beam_dim_x, beam_dim_y, beam_dim_z,
                    beam_center_x, beam_center_y, beam_center_z,
                    ground_dim_x, ground_dim_y, ground_dim_z,
                    ground_center_x, ground_center_y, ground_center_z,
                    wall_dim_x, wall_dim_y, wall_dim_z,
                    wall_center_x, wall_center_y, wall_center_z
                )
            }
            
            config_json = json.dumps(combined_config, indent=2)
            st.download_button(
                label="📥 Download Complete Config",
                data=config_json,
                file_name="complete_config.json",
                mime="application/json",
                use_container_width=True,
                on_click=lambda: st.session_state.update({'show_download': False})
            )
    
    # === PREVIEW COLUMN ===
    with col_preview:
        # Show either merged mesh or input stones
        if st.session_state.get('model_generated', False) and st.session_state.get('output_mesh_path'):
            st.subheader("🎨 Merged Geometry Preview")
            
            # Add tabs for different views
            preview_tabs = st.tabs(["Merged Geometry", "Input Stones"])
            
            with preview_tabs[0]:
                try:
                    output_path = st.session_state.output_mesh_path
                    if os.path.exists(output_path):
                        preview_single_mesh(output_path, color='lightcoral', name="Merged Geometry")
                        
                        # Show statistics
                        st.write("**Model Statistics:**")
                        import pymeshlab
                        ms = pymeshlab.MeshSet()
                        ms.load_new_mesh(output_path)
                        m = ms.current_mesh()
                        st.write(f"  • Vertices: {m.vertex_number()}")
                        st.write(f"  • Faces: {m.face_number()}")
                        
                        # Download button
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download Merged Mesh",
                                data=f,
                                file_name=os.path.basename(output_path),
                                mime="application/octet-stream",
                                use_container_width=True
                            )
                    else:
                        st.warning("Merged mesh file not found")
                except Exception as e:
                    st.error(f"Error previewing merged mesh: {str(e)}")
            
            with preview_tabs[1]:
                if stone_files:
                    st.write(f"**Uploaded {len(stone_files)} stone file(s)**")
                    preview_meshes(stone_files)
                else:
                    st.info("No input stones to display")
        
        else:
            # Show input stones preview
            st.subheader("🔍 Stone Geometry Preview")
            
            if stone_files:
                st.write(f"**Uploaded {len(stone_files)} stone file(s)**")
                preview_meshes(stone_files)
                
                # Show file list
                with st.expander("📄 Uploaded Files"):
                    for i, stone_file in enumerate(stone_files):
                        st.write(f"{i+1}. {stone_file.name}")
            else:
                st.info("👈 Upload stone mesh files to preview")
        
        # # Show geometry summary (always visible)
        # st.divider()
        # st.write("**Component Dimensions:**")
        # st.write(f"🔹 Beam: {beam_dim_x}×{beam_dim_y}×{beam_dim_z}m")
        # st.write(f"🔹 Ground: {ground_dim_x}×{ground_dim_y}×{ground_dim_z}m")
        # st.write(f"🔹 Wall: {wall_dim_x}×{wall_dim_y}×{wall_dim_z}m")
    
    # === GENERATE BUTTON ===
    st.divider()
    
    if st.button("🔨 Generate Model", type="primary", use_container_width=True):
        if not stone_files:
            st.warning("ℹ️ No stone meshes uploaded. Running with mortar + beam + ground only.")
        
        with st.spinner("Generating model... This may take a few minutes"):
            try:
                import tempfile
                import sys
                
                from analysis.model_generation import process_wall_assembly
                
                # # Create working directory
                # work_dir = tempfile.mkdtemp()
                # st.session_state.work_dir = work_dir
                work_dir = st.session_state.get("work_dir")
                if not work_dir:
                    work_dir = os.path.join(".", "workspaces", st.session_state.get("session_id", "default"))
                    st.session_state.work_dir = work_dir
                os.makedirs(work_dir, exist_ok=True)

                
                # Create subdirectories
                temp_dir = os.path.join(work_dir, "temp")
                stones_dir = os.path.join(work_dir, "stones")
                os.makedirs(temp_dir, exist_ok=True)
                os.makedirs(stones_dir, exist_ok=True)
                st.session_state.stones_dir = stones_dir
                
                # Save geometry JSON
                geometry_config = create_geometry_json(
                    beam_dim_x, beam_dim_y, beam_dim_z,
                    beam_center_x, beam_center_y, beam_center_z,
                    ground_dim_x, ground_dim_y, ground_dim_z,
                    ground_center_x, ground_center_y, ground_center_z,
                    wall_dim_x, wall_dim_y, wall_dim_z,
                    wall_center_x, wall_center_y, wall_center_z
                )
                
                geometry_json_path = os.path.join(work_dir, "geometry.json")
                with open(geometry_json_path, 'w') as f:
                    json.dump(geometry_config, f, indent=2)
                
                st.write(f"✓ Created geometry.json")
                
                # Save stone files
                stone_paths = []
                for i, stone_file in enumerate(stone_files):
                    stone_path = os.path.join(stones_dir, stone_file.name)
                    with open(stone_path, 'wb') as f:
                        f.write(stone_file.getvalue())
                    stone_paths.append(stone_path)
                
                st.write(f"✓ Saved {len(stone_paths)} stone files")
                
                # Output path for merged mesh
                output_path = os.path.join(work_dir, output_filename)
                
                # Run the pipeline
                st.write("🔄 Running wall assembly pipeline...")
                
                merged_mesh = process_wall_assembly(
                    json_path=geometry_json_path,
                    stone_mesh_paths=stone_paths,
                    output_path=output_path,
                    temp_dir=temp_dir,
                    process_stone_normals=process_stone_normals,
                    simplify_stones=simplify_stones,
                    simplification_ratio=simplification_ratio,
                    stone_transforms=stone_transforms
                )
                
                st.session_state.model_generated = True
                st.session_state.output_mesh_path = output_path
                
                st.success("✅ Model generated successfully!")
                st.info("👉 Proceed to 'Meshing' tab")
                
                # Show generated files
                with st.expander("📄 Generated Files"):
                    st.write("**Configuration files:**")
                    st.write(f"  • geometry.json")
                    #st.write(f"  • material.json")
                    st.write("**Stone files:**")
                    st.write(f"  • {len(stone_paths)} files in stones/")
                    st.write("**Merged output:**")
                    st.write(f"  • {output_filename}")
                
                # Rerun to show preview
                st.rerun()
                
            except ImportError as e:
                st.error(f"❌ Error importing pipeline module: {str(e)}")
                st.info("Make sure your pipeline.py is in the analysis/ directory")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())


def create_material_json(
    nb_points_per_interface,
    fc_from_test, fc_stone, fc_beam,
    flt_from_test, flt_h,
    alpha, beta, ratio_strength_beam,
    mu, mu_interface, mu_beam,
    density_stone, density_mortar,
    emodulus_stone, emodulus_mortar, emodulus_beam,
    beam_ground_element_center_to_interface,
    wall_height, wall_diagonal,
    sample_points_radius_to_D,
    force_ground_beam_by_x,
    wall_plane_xs, wall_plane_ys, wall_plane_zs
):
    """Create material.json configuration for contact generation"""
    
    return {
        "nb_points_per_interface": nb_points_per_interface,
        "fc_from_test": fc_from_test,
        "fc_stone": fc_stone,
        "fc_beam": fc_beam,
        "flt_from_test": flt_from_test,
        "flt_h": flt_h,
        "alpha": alpha,
        "beta": beta,
        "ratio_strength_beam": ratio_strength_beam,
        "mu": mu,
        "mu_interface": mu_interface,
        "mu_beam": mu_beam,
        "Density_stone": density_stone,
        "Density_mortar": density_mortar,
        "Emodulus_stone": emodulus_stone,
        "Emodulus_mortar": emodulus_mortar,
        "Emodulus_beam": emodulus_beam,
        "beam_ground_element_center_to_interface": beam_ground_element_center_to_interface,
        "Wall_height": wall_height,
        "Wall_diagonal": wall_diagonal,
        "Sample_points_radius_to_D": sample_points_radius_to_D,
        "Force_ground_beam_by_x": force_ground_beam_by_x,
        "wall_plane_xs": wall_plane_xs,
        "wall_plane_ys": wall_plane_ys,
        "wall_plane_zs": wall_plane_zs
    }


def create_geometry_json(
    beam_dim_x, beam_dim_y, beam_dim_z,
    beam_center_x, beam_center_y, beam_center_z,
    ground_dim_x, ground_dim_y, ground_dim_z,
    ground_center_x, ground_center_y, ground_center_z,
    wall_dim_x, wall_dim_y, wall_dim_z,
    wall_center_x, wall_center_y, wall_center_z
):
    """Create geometry.json configuration matching your pipeline's expected format"""
    
    return {
        "beam_dim_x": beam_dim_x,
        "beam_dim_y": beam_dim_y,
        "beam_dim_z": beam_dim_z,
        "beam_center": [beam_center_x, beam_center_y, beam_center_z],
        
        "ground_dim_x": ground_dim_x,
        "ground_dim_y": ground_dim_y,
        "ground_dim_z": ground_dim_z,
        "ground_center": [ground_center_x, ground_center_y, ground_center_z],
        
        "wall_dim_x": wall_dim_x,
        "wall_dim_y": wall_dim_y,
        "wall_dim_z": wall_dim_z,
        "wall_center": [wall_center_x, wall_center_y, wall_center_z]
    }
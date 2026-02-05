import streamlit as st
import os
import json

def render(refresh_token: int = 0):
    """Render the Contact Generation tab"""
    #st.header("Contact Point Generation")
    
    # ========================================
    # Check if mesh is ready
    # ========================================
    
    if not st.session_state.get('meshing_done', False):
        st.warning("⚠️ Please generate the mesh first (Tab 2: Meshing)")
        return
    
    if not st.session_state.get('mesh_output_path'):
        st.error("❌ Mesh file not found. Please regenerate the mesh.")
        return
    
    st.success("✅ Mesh is ready for contact generation")
    
    st.divider()
    
    # ========================================
    # Material Properties Input
    # ========================================
    
    st.subheader("📋 Material Properties")
    
    # Check if material.json already exists
    material_json_path = os.path.join(st.session_state.work_dir, 'material.json')
    
    # Material input tabs
    material_tabs = st.tabs(["✏️ Enter Manually","📂 Upload material.json"])
    
    
    
    with material_tabs[0]:
        st.write("**Enter material properties manually:**")
        
        # # Load existing if available
        default_values = {}
        # if os.path.exists(material_json_path):
        #     try:
        #         with open(material_json_path, 'r') as f:
        #             default_values = json.load(f)
        #         #st.info("✓ Loaded existing material.json")
        #     except:
        #         pass
        
        def get_value(key, fallback):
            return default_values.get(key, fallback)
        
        # Use tabs for different materials
        # =========================================================
        # Material / Interface properties (minimal UI + defaults)
        # =========================================================

        mat_property_tabs = st.tabs(["Stone", "Mortar", "Beam&Ground"])

        # ---------- Defaults (no UI) ----------
        # These are required by util_meshing.py -> material.json
        DEFAULTS = {
            # friction + interface model
            #"mu": 1.0,                     # stone-stone / mortar-mortar friction
            "nb_points_per_interface": 6,   # used for cohesion scaling
            #"alpha": 1.0,                  # ratio_c_ft
            # beam/ground contact properties
            #"fc_beam": 6560.0,
            #"Emodulus_beam": 16900.0,
            #"mu_beam": 1.0,
            #"ratio_strength_beam": 100.0,
            # geometry-analysis
            "Sample_points_radius_to_D": 0.01,
            "Force_ground_beam_by_x": True,
        }
        # So make sure wall_dim_x/y/z exist above this block.
        wall_dim_x = float(st.session_state.get("wall_dim_x", 0.9))
        wall_dim_y = float(st.session_state.get("wall_dim_y", 0.9))
        wall_dim_z = float(st.session_state.get("wall_dim_z", 0.2))
        # Derive what util_meshing.py expects:
        wall_height = float(wall_dim_x)  # your util_meshing uses X as "height"
        wall_diagonal = float((wall_dim_x**2 + wall_dim_y**2 + wall_dim_z**2) ** 0.5)

        # Beam/ground interface distance: set a default based on wall size (no UI)
        beam_ground_element_center_to_interface = 0.01

        # Wall plane bounds (derived; util_meshing uses these for skipping boundary faces)
        wall_plane_xs = [0.0, float(wall_dim_x)]
        wall_plane_ys = [0.0, float(wall_dim_y)]
        wall_plane_zs = [0.0, float(wall_dim_z)]

        # ---------- Tab: Stone ----------
        with mat_property_tabs[0]:
            st.write("**Stone Properties**")
            fc_stone = float(st.number_input(
                "Compressive Strength fc (MPa)",
                value=float(get_value("fc_stone", 100.0)),
                min_value=0.0,
                key="fc_stone",
            ))
            emodulus_stone = float(st.number_input(
                "Elastic Modulus E (MPa)",
                value=float(get_value("Emodulus_stone", 16700.0)),
                min_value=0.0,
                key="emodulus_stone",
            ))
            lambda_stone = float(st.number_input(
                "Poisson Ratio",
                value=float(get_value("lambda_stone", 0.15)),
                min_value=0.0,
                key="lambda_stone",
            ))
            density_stone = float(st.number_input(
                "Density (kg/m³)",
                value=float(get_value("Density_stone", 0.0)),
                min_value=0.0,
                key="density_stone",
            ))
            G_f1_stone = float(st.number_input(
                "Mode 1 Fracture Energy (N/mm)",
                value=float(get_value("G_f1_stone", 0.012)),
                min_value=0.0,
                key="G_f1_stone",
            ))
            G_f2_stone = float(st.number_input(
                "Mode 2 Fracture Energy (N/mm)",
                value=float(get_value("G_f2_stone", 0.071)),
                min_value=0.0,
                key="G_f2_stone",
            ))
            G_c_stone = float(st.number_input(
                "Compression Fracture Energy (N/mm)",
                value=float(get_value("G_c_stone", 100)),
                min_value=0.0,
                key="G_c_stone",
            ))
            beta = float(st.number_input(
                "Ratio of cohesion and tensile strength of stone-mortar interface compared to mortar-mortar interface",
                value=float(get_value("beta", 0.33333)),
                min_value=0.0,
                step=0.01,
                format="%.5f",
                key="beta",
            ))
            mu_interface_stone = float(st.number_input(
                "Stone-mortar Interface Friction Coefficient μ",
                value=float(get_value("mu_interface_stone", 1.01)),
                min_value=0.0,
                step=0.1,
                key="mu_interface_stone",
            ))

        # ---------- Tab: Mortar ----------
        with mat_property_tabs[1]:
            st.write("**Mortar Properties**")
            fc_from_test = float(st.number_input(
                "Compressive Strength of mortar fc (MPa)",
                value=float(get_value("fc_from_test", 100.0)),
                min_value=0.0,
                key="fc_from_test",
            ))
            emodulus_mortar = float(st.number_input(
                "Elastic Modulus E (MPa)",
                value=float(get_value("Emodulus_mortar", 2970.0)),
                min_value=0.0,
                key="emodulus_mortar",
            ))
            lambda_mortar = float(st.number_input(
                "Poisson Ratio",
                value=float(get_value("lambda_mortar", 0.15)),
                min_value=0.0,
                key="lambda_mortar",
            ))
            m_m_tensile = float(st.number_input(
                "Tensile Strength of mortar ft (MPa)",
                value=float(get_value("m_m_tensile", 0.3)),
                min_value=0.0,
                key="m_m_tensile",
            ))
            m_m_cohesion = float(st.number_input(
                "Cohesion of mortar c (MPa)",
                value=float(get_value("m_m_cohesion", 0.87)),
                min_value=0.0,
                key="m_m_cohesion",
            ))
            mu_interface_mortar = float(st.number_input(
                "Mortar-mortar Interface Friction Coefficient μ",
                value=float(get_value("mu_interface_mortar", 1.01)),
                min_value=0.0,
                step=0.1,
                key="mu_interface_mortar",
            ))
            density_mortar = float(st.number_input(
                "Density (kg/m³)",
                value=float(get_value("Density_mortar", 0.0)),
                min_value=0.0,
                key="density_mortar",
            ))

            G_f1_mortar = float(st.number_input(
                "Mode 1 Fracture Energy (N/mm)",
                value=float(get_value("G_f1_mortar", 0.012)),
                min_value=0.0,
                key="G_f1_mortar",
            ))
            G_f2_mortar = float(st.number_input(
                "Mode 2 Fracture Energy (N/mm)",
                value=float(get_value("G_f2_mortar", 0.071)),
                min_value=0.0,
                key="G_f2_mortar",
            ))
            G_c_mortar = float(st.number_input(
                "Compression Fracture Energy (N/mm)",
                value=float(get_value("G_c_mortar", 100.0)),
                min_value=0.0,
                key="G_c_mortar",
            ))


        # ---------- Tab: Beam ----------
        with mat_property_tabs[2]:
            st.write("**Beam & Ground Properties**")
            fc_beam = float(st.number_input(
                "Compressive Strength of beam fc (MPa)",
                value=float(get_value("fc_beam", 100.0)),
                min_value=0.0,
                key="fc_beam",
            ))
            emodulus_beam = float(st.number_input(
                "Elastic Modulus of beam E (MPa)",
                value=float(get_value("Emodulus_beam", 16700.0)),
                min_value=0.0,
                key="emodulus_beam",
            ))
            lambda_beam = float(st.number_input(
                "Poisson Ratio",
                value=float(get_value("lambda_beam", 0.15)),
                min_value=0.0,
                key="lambda_beam",
            ))
            m_b_tensile = float(st.number_input(
                "Tensile Strength of beam-mortar interface ft (MPa)",
                value=float(get_value("m_b_tensile", 0.3)),
                min_value=0.0,
                key="m_b_tensile",
            ))
            m_b_cohesion = float(st.number_input(
                "Cohesion of beam-mortar interface c (MPa)",
                value=float(get_value("m_b_cohesion", 0.87)),
                min_value=0.0,
                key="m_b_cohesion",
            ))
            mu_beam = float(st.number_input(
                "Beam-mortar Interface Friction Coefficient μ",
                value=float(get_value("mu_interface_beam", 1.01)),
                min_value=0.0,
                step=0.1,
                key="mu_interface_beam",
            ))
            # density_beam = float(st.number_input(
            #     "Density (kg/m³)",
            #     value=float(get_value("Density_beam", 0.0)),
            #     min_value=0.0,
            #     key="density_beam",
            # ))
            G_f1_beam = float(st.number_input(
                "Mode 1 Fracture Energy (N/mm)",
                value=float(get_value("G_f1_beam", 0.012)),
                min_value=0.0,
                key="G_f1_beam",
            ))
            G_f2_beam = float(st.number_input(
                "Mode 2 Fracture Energy (N/mm)",
                value=float(get_value("G_f2_beam", 0.071)),
                min_value=0.0,
                key="G_f2_beam",
            ))
            G_c_beam = float(st.number_input(
                "Compression Fracture Energy (N/mm)",
                value=float(get_value("G_c_beam", 100)),
                min_value=0.0,
                key="G_c_beam",
            ))
            
            

        # ---------- Final values used to write material.json ----------
        # Keep all required keys for util_meshing.py
        #mu = float(DEFAULTS["mu"])
        nb_points_per_interface = int(DEFAULTS["nb_points_per_interface"])
        #alpha = float(DEFAULTS["alpha"])

        #fc_beam = fc_stone
        #emodulus_beam = emodulus_stone
        #mu_beam = float(DEFAULTS["mu_beam"])
        #ratio_strength_beam = float(DEFAULTS["ratio_strength_beam"])

        sample_points_radius_to_D = float(DEFAULTS["Sample_points_radius_to_D"])
        force_ground_beam_by_x = bool(DEFAULTS["Force_ground_beam_by_x"])

        # Save material.json button
        if st.button("OK", use_container_width=True):
            #try:
            # Create material.json
            material_config = {
                # --- global / defaults you set above ---
                "nb_points_per_interface": nb_points_per_interface,
                #"alpha": alpha,
                "Sample_points_radius_to_D": sample_points_radius_to_D,
                "Force_ground_beam_by_x": force_ground_beam_by_x,

                # --- Stone properties (Tab 0) ---
                "fc_stone": fc_stone,
                "Emodulus_stone": emodulus_stone,
                "lambda_stone":lambda_stone,
                "Density_stone": density_stone,
                "G_f1_stone": G_f1_stone,
                "G_f2_stone": G_f2_stone,
                "G_c_stone": G_c_stone,
                "beta": beta,
                "mu_interface_stone": mu_interface_stone,

                # --- Mortar properties (Tab 1) ---
                "fc_from_test": fc_from_test,
                "Emodulus_mortar": emodulus_mortar,
                "lambda_mortar":lambda_mortar,
                "Density_mortar": density_mortar,
                "m_m_tensile": m_m_tensile,
                "m_m_cohesion": m_m_cohesion,
                "mu_interface_mortar": mu_interface_mortar,
                "G_f1_mortar": G_f1_mortar,
                "G_f2_mortar": G_f2_mortar,
                "G_c_mortar": G_c_mortar,

                # --- Beam & ground properties (Tab 2) ---
                "fc_beam": fc_beam,
                "Emodulus_beam": emodulus_beam,
                "lambda_beam":lambda_beam,
                #"Density_beam": density_beam,
                "m_b_tensile": m_b_tensile,
                "m_b_cohesion": m_b_cohesion,
                "mu_interface_beam": mu_beam,  # your variable name is mu_beam but the meaning is interface friction
                "G_f1_beam": G_f1_beam,
                "G_f2_beam": G_f2_beam,
                "G_c_beam": G_c_beam,

                # --- geometry / other inputs that exist elsewhere in your app ---
                "beam_ground_element_center_to_interface": beam_ground_element_center_to_interface,
                "Wall_height": wall_height,
                "Wall_diagonal": wall_diagonal,
                "wall_plane_xs": wall_plane_xs,
                "wall_plane_ys": wall_plane_ys,
                "wall_plane_zs": wall_plane_zs,
            }

                
            # Save to work directory
            with open(material_json_path, 'w+') as f:
                json.dump(material_config, f, indent=2)
                
            st.success("✅ Material properties saved to material.json")
            st.rerun()
                
            # except Exception as e:
            #     st.error(f"❌ Error saving material.json: {str(e)}")
    
    with material_tabs[1]:
        st.write("**Upload existing material.json file:**")
        
        uploaded_material = st.file_uploader(
            "Upload material.json",
            type=['json'],
            key='material_upload',
            help="Upload a material.json file from a previous project"
        )
        
        if uploaded_material:
            try:
                material_data = json.load(uploaded_material)
                
                # Save to work directory
                with open(material_json_path, 'w') as f:
                    json.dump(material_data, f, indent=2)
                
                st.success("✅ material.json uploaded and saved")
                
                # Show preview
                with st.expander("📄 View Uploaded Properties"):
                    st.json(material_data)
                
            except Exception as e:
                st.error(f"❌ Error loading material.json: {str(e)}")
    #st.divider()
    
    # ========================================
    # Check prerequisites
    # ========================================
    
    st.write("**Checking required files...**")
    
    mortar_ply_path = st.session_state.get("output_mesh_path")
    stones_dir = st.session_state.get("temp_dir")# use processed stones stored in temp
    mesh_path = st.session_state.mesh_output_path
    
    missing_files = []
    if not os.path.exists(material_json_path):
        missing_files.append('material.json')
    if not os.path.exists(mortar_ply_path):
        missing_files.append('mortar.ply')
    if not os.path.exists(stones_dir) or not os.path.isdir(stones_dir):
        missing_files.append('stones/ directory')
    
    if missing_files:
        st.error(f"❌ Missing required files: {', '.join(missing_files)}")
        st.info("""
        **Required files:**
        - `material.json` - Upload or enter material properties above ☝️
        - `mortar.ply` - Surface mesh (from Model Generation)
        - `stones/` - Directory with stone meshes (from Model Generation)
        - `mortar_01.msh` - Tetrahedral mesh (from Meshing) ✓
        """)
        return
    
    st.success("✅ All required files present")
    
    st.divider()
    
    # ========================================
    # Parameters
    # ========================================
    
    st.subheader("🔧 Contact Generation Parameters")
    
    # Boundary condition selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        boundary_condition = st.selectbox(
            "Boundary Condition",
            ["double_bending", "cantilever"],
            index=0,
            help="Type of boundary condition for the loading beam"
        )
        
        st.caption("""
        - **double bending**: beam rotation fixed
        - **cantilevel**: beam rotation free
        """)
    
    with col2:
        st.write("**Options**")
        stone_stone_contact = st.checkbox("Detect stone-stone contact", value=False)
        show_progress = st.checkbox("Show detailed progress", value=True)
    
    st.divider()

    
    
    # ========================================
    # Generate Contact Points Button
    # ========================================
    
    if st.button("⚡ Generate Contact Points", type="primary", use_container_width=True):
        
        # Create containers for progress
        progress_container = st.container()
        log_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        if show_progress:
            with log_container:
                with st.expander("📝 Generation Log", expanded=True):
                    log_output = st.empty()
        
        try:
            from analysis.contact_generation import generate_contact_points
            import io
            import sys
            
            status_text.write("🔄 Generating contact points...")
            progress_bar.progress(10)
            
            # Capture stdout if showing progress
            if show_progress:
                old_stdout = sys.stdout
                sys.stdout = log_buffer = io.StringIO()
            
            progress_bar.progress(20)
            
            # Run contact point generation
            stone_stone_contact_verb = "false" if not stone_stone_contact else "true"
            output_files = generate_contact_points(
                material_json_path=material_json_path,
                mortar_ply_path=mortar_ply_path,
                stones_dir=stones_dir,
                mortar_msh_path=mesh_path,
                output_dir=st.session_state.work_dir,
                boundary_string=boundary_condition,
                stone_stone_contact=stone_stone_contact_verb
            )
            
            progress_bar.progress(90)
            
            # Restore stdout and show log
            if show_progress:
                sys.stdout = old_stdout
                log_output.code(log_buffer.getvalue())
            
            # Update session state
            st.session_state.contact_generation_done = True
            st.session_state.contact_output_files = output_files
            
            progress_bar.progress(100)
            status_text.empty()
            
            st.success("✅ Contact points generated successfully!")
            st.balloons()
            st.info("👉 Proceed to 'Run Analysis' tab")
            
            # Show generated files with download
            with st.expander("📄 Generated Files", expanded=True):
                st.write("**Model Files:**")
                
                # Create ZIP for download
                import zipfile
                import io as io_module
                
                zip_buffer = io_module.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add mesh file
                    if os.path.exists(mesh_path):
                        zip_file.write(mesh_path, os.path.basename(mesh_path))
                    
                    # Add important CSV files
                    for name in ['point_mortar', 'element', 'properties']:
                        if name in output_files and os.path.exists(output_files[name]):
                            zip_file.write(output_files[name], os.path.basename(output_files[name]))
                            file_size = os.path.getsize(output_files[name]) / 1024
                            st.write(f"  • `{os.path.basename(output_files[name])}` ({file_size:.1f} KB)")
                
                zip_buffer.seek(0)
                
                st.divider()
                
                st.download_button(
                    label="📦 Download Model",
                    data=zip_buffer,
                    file_name="masonry_model.zip",
                    mime="application/zip",
                    use_container_width=True,
                    help="Downloads: mortar_01.msh, point_mortar.csv, element.csv, properties.json"
                )
            
            # Show statistics
            with st.expander("📊 Statistics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Contact Points:**")
                    if os.path.exists(output_files['point']):
                        import pandas as pd
                        df = pd.read_csv(output_files['point'])
                        st.metric("Total Contact Points", f"{len(df):,}")
                        st.metric("Unique Faces", f"{df['face_id'].nunique():,}")
                
                with col2:
                    st.write("**Elements:**")
                    if os.path.exists(output_files['element']):
                        import pandas as pd
                        df = pd.read_csv(output_files['element'])
                        st.metric("Total Elements", f"{len(df):,}")
                        total_mass = df['mass'].sum()
                        st.metric("Total Mass", f"{total_mass:.2f} kg")
            
            # Wait then rerun to update UI
            import time
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            if show_progress and 'old_stdout' in locals():
                sys.stdout = old_stdout
            
            progress_bar.progress(0)
            status_text.error(f"❌ Contact generation failed")
            
            st.error(f"Error: {str(e)}")
            
            import traceback
            with st.expander("🔍 Show Error Details"):
                st.code(traceback.format_exc())
            
            st.info("""
            **Troubleshooting:**
            - Verify all required files are present
            - Check material.json parameters
            - Ensure mesh file is valid
            - Check that stones directory contains .ply files
            """)
    
    # ========================================
    # Show Status if Already Complete
    # ========================================
    
    if st.session_state.get('contact_generation_done', False):
        st.divider()
        st.success("✅ Contact points have been generated")
        
        if st.session_state.get('contact_output_files'):
            # Create ZIP download
            mesh_output_path = st.session_state.get('mesh_output_path')
            output_files = st.session_state.contact_output_files
            
            import zipfile
            import io as io_module
            
            zip_buffer = io_module.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add mesh file
                if mesh_output_path and os.path.exists(mesh_output_path):
                    zip_file.write(mesh_output_path, os.path.basename(mesh_output_path))
                
                # Add CSV files
                for name in ['point', 'element', 'properties']:
                    if name in output_files and os.path.exists(output_files[name]):
                        zip_file.write(output_files[name], os.path.basename(output_files[name]))
            
            zip_buffer.seek(0)
            
            st.download_button(
                label="📦 Download Model",
                data=zip_buffer,
                file_name="masonry_model.zip",
                mime="application/zip",
                use_container_width=True,
                help="Downloads: mortar_01.msh, point_mortar.csv, element.csv, properties.json"
            )
            
            with st.expander("📄 View Individual Files"):
                allowed_keys = {
                    "point": "Contact Points (point.csv)",
                    "element": "Elements (element.csv)",
                }
                for key, label in allowed_keys.items():
                    path = st.session_state.contact_output_files.get(key)
                    if path and os.path.exists(path):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            file_size = os.path.getsize(path) / 1024
                            st.write(f"**{label}** — `{os.path.basename(path)}` ({file_size:.1f} KB)")

                        with col2:
                            with open(path, "rb") as f:
                                st.download_button(
                                    label="📥",
                                    data=f,
                                    file_name=os.path.basename(path),
                                    key=f"download_{key}",
                                    use_container_width=True,
                                )
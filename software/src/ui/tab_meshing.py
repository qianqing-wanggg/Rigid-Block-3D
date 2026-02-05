import streamlit as st
import os
from utils.mesh_preview import preview_single_mesh

def render(refresh_token: int = 0):
    """Render the Meshing tab"""
    #st.header("Mesh Generation")
    
    # ========================================
    # Check fTetWild Configuration
    # ========================================
    
    from analysis.meshing import check_ftetwild_available, find_floattetwild_binary
    
    ftetwild_path = st.session_state.get('ftetwild_path', '')
    
    if ftetwild_path:
        os.environ['FTETWILD_BIN'] = ftetwild_path
    
    if not check_ftetwild_available():
        st.error("❌ fTetWild not found")
        
        # Configuration dialog
        with st.container(border=True):
            st.subheader("🔧 Configure fTetWild")
            
            st.write("Please specify where you installed fTetWild:")
            
#             st.info("""
#             **Need the path?** In your terminal, navigate to your fTetWild directory and run:
# ```bash
#             realpath build/FloatTetwild_bin
# ```
#             Then paste the output below.
#             """)
            
            new_path = st.text_input(
                "Path to FloatTetwild_bin",
                placeholder="/home/user/fTetWild/build/FloatTetwild_bin",
                key='ftetwild_setup_path'
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save Path", type="primary", use_container_width=True):
                    if new_path and os.path.exists(new_path):
                        st.session_state.ftetwild_path = new_path
                        os.environ['FTETWILD_BIN'] = new_path
                        st.success("✅ Configuration saved!")
                        st.rerun()
                    elif new_path:
                        st.error("❌ File not found at this path")
                    else:
                        st.warning("Please enter a path")
            
            with col2:
                if st.button("🔍 Try Auto-Detect", use_container_width=True):
                    detected = find_floattetwild_binary()
                    if os.path.exists(detected):
                        st.session_state.ftetwild_path = detected
                        os.environ['FTETWILD_BIN'] = detected
                        st.success(f"✅ Found: {detected}")
                        st.rerun()
                    else:
                        st.error("❌ Could not auto-detect fTetWild")
        
        return  # Don't show meshing UI until configured
    
    # Show configured path
    st.success(f"✅ fTetWild configured: `{ftetwild_path}`")
    
    st.divider()
    
    # ========================================
    # Check if model is ready
    # ========================================
    
    if not st.session_state.get('model_generated', False):
        st.warning("⚠️ Please generate the model first (Tab 1: Model Generation)")
        return
    
    if not st.session_state.get('output_mesh_path'):
        st.error("❌ Model file not found. Please regenerate the model.")
        return
    
    # ========================================
    # Main UI Layout
    # ========================================
    
    col_params, col_preview = st.columns([1, 1])
    
    with col_params:
        st.subheader("🔧 Mesh Parameters")
        
        # === MESH SIZE ===
        #st.write("**Mesh Resolution**")
        
        mesh_preset = st.selectbox(
            "Quality Preset",
            ["Coarse", "Medium", "Fine","Custom"],
            index=0,
            help="Choose mesh resolution. Finer meshes are more accurate but take longer to generate and analyze."
        )
        
        # Preset mappings
        preset_values = {
            "Coarse": (0.1, "Smaller model"),
            "Medium": (0.05, "Medium-sized model"),
            "Fine": (0.01, "Larger model")
        }
        
        if mesh_preset == "Custom":
            edge_length = st.number_input(
                "Relative Edge Length",
                value=0.02,
                min_value=0.01,
                max_value=0.5,
                step=0.01,
                format="%.4f",
                help="Target relative edge length for mesh elements. Smaller values create finer meshes."
            )
            st.caption("💡 Please refer to fTetWild documentary for detailed explaination")
        else:
            edge_length, description = preset_values[mesh_preset]
            st.caption(f"📏 Edge length: **{edge_length} * diagonal length of input mesh**")
        
        #st.divider()
        
        # === STOP ENERGY ===
        #st.write("**Mesh Quality Optimization**")
        
        stop_energy = st.slider(
            "Stop Energy",
            min_value=1,
            max_value=20,
            value=8,
            help="Controls mesh optimization termination. LOWER values produce BETTER quality but take longer. Higher values are faster but lower quality."
        )
        
        # # Quality indicator (corrected: lower is better)
        # col1, col2 = st.columns([1, 2])
        # with col1:
        #     if stop_energy < 5:
        #         st.caption("🎯 **Best Quality**")
        #     elif stop_energy <= 10:
        #         st.caption("⚖️ **Good Quality**")
        #     else:
        #         st.caption("⚡ **Lower Quality**")
        
        # with col2:
        #     if stop_energy < 5:
        #         st.caption("Highest quality (slowest)")
        #     elif stop_energy <= 10:
        #         st.caption("Balanced (recommended)")
        #     else:
        #         st.caption("Faster but lower quality")
        
        st.caption("💡 **Note:** Lower stop energy = better quality but longer processing time")
        
        #st.divider()
        
        # === ADVANCED OPTIONS ===
        #with st.expander("⚙️ Advanced Options", expanded=False):
        enable_coarsen = st.checkbox(
            "Enable Mesh Coarsening",
            value=False,
            help="Allows the mesher to remove unnecessary elements, reducing mesh size and improving performance."
        )
        
        #st.caption("💡 Recommended: Keep this enabled unless you have specific requirements.")
    
    # ========================================
    # Preview Column
    # ========================================
    
    with col_preview:
        st.subheader("📊 Model & Mesh Preview")
        st.caption("💡 Make sure that there is no holes in the generated tetrahedron mesh.")
        
        # Show current model
        if st.session_state.get('output_mesh_path'):
            output_path = st.session_state.output_mesh_path
            
            if os.path.exists(output_path):
                # Check if mesh has been generated
                if st.session_state.get('meshing_done', False) and st.session_state.get('mesh_output_path'):
                    # Show tabs for input model and generated mesh
                    preview_tabs = st.tabs(["Generated Mesh", "Input Model"])
                    
                    with preview_tabs[0]:
                        mesh_path = st.session_state.mesh_output_path
                        if os.path.exists(mesh_path):
                            try:
                                st.write("**Tetrahedral Mesh**")
                                
                                # Add option to show/hide surface extraction
                                show_surface = st.checkbox(
                                    "Show 3D Surface Visualization", 
                                    value=True,
                                    help="Extract and visualize the surface of the tetrahedral mesh (may be slow for large meshes)"
                                )
                                
                                preview_single_mesh(mesh_path, color='lightgreen', name="Mesh", show_surface=show_surface)
                                
                                st.divider()
                                
                                # Download button for MSH file
                                with open(mesh_path, 'rb') as f:
                                    st.download_button(
                                        label="📥 Download Mesh File (.msh)",
                                        data=f,
                                        file_name=os.path.basename(mesh_path),
                                        mime="application/octet-stream",
                                        use_container_width=True
                                    )
                                
                                # Show file info
                                file_size = os.path.getsize(mesh_path) / (1024 * 1024)
                                st.caption(f"📁 File: {os.path.basename(mesh_path)} ({file_size:.2f} MB)")
                                
                            except Exception as e:
                                st.error(f"Error loading mesh: {str(e)}")
                        else:
                            st.warning("Mesh file not found")
                    
                    with preview_tabs[1]:
                        st.write("**Input Surface Model**")
                        preview_single_mesh(output_path, color='lightblue', name="Model", show_surface=False)
                
                else:
                    # Show only input model
                    st.write("**Input Surface Model**")
                    preview_single_mesh(output_path, color='lightblue', name="Model", show_surface=False)
                    
                    try:
                        import pymeshlab
                        ms = pymeshlab.MeshSet()
                        ms.load_new_mesh(output_path)
                        m = ms.current_mesh()
                        
                        st.write("**Model Statistics:**")
                        col1, col2 = st.columns(2)
                        col1.metric("Vertices", f"{m.vertex_number():,}")
                        col2.metric("Faces", f"{m.face_number():,}")
                    except:
                        pass
            else:
                st.warning("Model file not found")
        else:
            st.info("No model available for preview")
    
    # ========================================
    # Generate Mesh Button
    # ========================================
    
    st.divider()
    
    # Warning if mesh already exists
    if st.session_state.get('meshing_done', False):
        st.warning("⚠️ A mesh has already been generated. Generating a new mesh will overwrite the previous one.")
    
    if st.button("🔨 Generate Mesh", type="primary", use_container_width=True):
        if not st.session_state.get('output_mesh_path'):
            st.error("⚠️ Model not found. Please generate model first.")
            return
        
        # Create containers for progress and logs
        progress_container = st.container()
        log_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        with log_container:
            with st.expander("📝 fTetWild Output Log", expanded=True):
                log_output = st.empty()
        
        try:
            import subprocess
            from analysis.meshing import find_floattetwild_binary
            
            status_text.write("🔄 Preparing meshing...")
            progress_bar.progress(10)
            
            # Get fTetWild binary
            floattetwild_bin = find_floattetwild_binary()
            
            if not os.path.exists(floattetwild_bin):
                st.error(f"❌ fTetWild binary not found at: {floattetwild_bin}")
                return
            
            # Prepare paths
            input_mesh = st.session_state.output_mesh_path
            output_dir = st.session_state.work_dir
            output_basename = 'mortar_01.msh'
            output_path = os.path.join(output_dir, output_basename)
            
            status_text.write("🔄 Running fTetWild...")
            status_text.caption(f"Edge length: {edge_length}m, Stop energy: {stop_energy}")
            progress_bar.progress(20)
            
            # Build command
            command = [
                floattetwild_bin,
                "-l", str(edge_length),
                "--coarsen",
                "--stop-energy", str(stop_energy),
                "--input", input_mesh,
                "-o", output_path
            ]
            
            log_lines = []
            log_lines.append("=" * 60)
            log_lines.append("RUNNING fTetWild")
            log_lines.append("=" * 60)
            log_lines.append(f"Command: {' '.join(command)}")
            log_lines.append("")
            log_output.code('\n'.join(log_lines))
            
            # Run fTetWild with real-time output capture
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            progress_bar.progress(30)
            
            # Read output in real-time
            for line in process.stdout:
                log_lines.append(line.rstrip())
                if len(log_lines) > 60:
                    log_lines = log_lines[:10] + ["...(earlier output truncated)..."] + log_lines[-40:]
                log_output.code('\n'.join(log_lines))
                
                # Update progress based on output keywords
                if "Preprocessing" in line or "Loading" in line:
                    progress_bar.progress(40)
                elif "Tetrahedralizing" in line or "tetrahedron" in line.lower():
                    progress_bar.progress(60)
                elif "Optimizing" in line or "optimization" in line.lower():
                    progress_bar.progress(80)
            
            # Wait for process to complete
            return_code = process.wait()
            
            log_lines.append("")
            log_lines.append("=" * 60)
            
            if return_code != 0:
                log_lines.append(f"ERROR: fTetWild exited with code {return_code}")
                log_lines.append("=" * 60)
                log_output.code('\n'.join(log_lines))
                progress_bar.progress(100)
                status_text.error(f"❌ fTetWild failed with exit code {return_code}")
                st.error("Check the log above for details")
                return
            
            log_lines.append("fTetWild completed successfully")
            log_lines.append("=" * 60)
            log_output.code('\n'.join(log_lines))
            
            progress_bar.progress(90)
            status_text.write("🔍 Verifying output...")
            
            # Find the output file
            expected_output = output_path
            
            if not os.path.exists(expected_output):
                files_in_dir = [f for f in os.listdir(output_dir) if f.endswith('.msh')]
                st.error(f"❌ Output file not found: {expected_output}")
                st.write(f"Files in output directory: {files_in_dir}")
                return
            
            mesh_output_path = expected_output
            
            # Update session state
            st.session_state.meshing_done = True
            st.session_state.mesh_output_path = mesh_output_path
            
            progress_bar.progress(100)
            status_text.empty()
            
            st.success("✅ Mesh generated successfully!")
            st.balloons()
            st.info("👉 Proceed to 'Contact Generation' tab")
            
            # Show output info
            with st.expander("📄 Output Details", expanded=True):
                st.write(f"**Mesh file:** `{mesh_output_path}`")
                if os.path.exists(mesh_output_path):
                    file_size = os.path.getsize(mesh_output_path) / (1024 * 1024)
                    st.write(f"**File size:** {file_size:.2f} MB")
                    
                    try:
                        import meshio
                        mesh_data = meshio.read(mesh_output_path)
                        st.write(f"**Vertices:** {len(mesh_data.points):,}")
                        if 'tetra' in mesh_data.cells_dict:
                            st.write(f"**Tetrahedra:** {len(mesh_data.cells_dict['tetra']):,}")
                    except Exception as e:
                        st.caption(f"Could not read mesh statistics: {e}")
            
            # Wait a moment then rerun to show preview
            import time
            time.sleep(1)
            st.rerun()
            
        except FileNotFoundError as e:
            st.error(f"❌ {str(e)}")
            st.info("💡 Make sure fTetWild is properly configured in Settings (sidebar)")
            
        except Exception as e:
            st.error(f"❌ Meshing failed: {str(e)}")
            import traceback
            with st.expander("🔍 Show Error Details"):
                st.code(traceback.format_exc())
            
            st.info("""
            **Troubleshooting Tips:**
            - Check that the input model is valid
            - Try using a coarser mesh (larger edge length)
            - Increase the stop energy value
            - Check fTetWild logs above for specific errors
            """)
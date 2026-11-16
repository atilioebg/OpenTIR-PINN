import os

def patch_file(fpath):
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    is_crlf = "\r\n" in content
    norm = content.replace("\r\n", "\n")

    # 1. Old Tab 5 header and inputs
    old_tab5_top = """with tab5:
    st.subheader("🌤️ Atmospheric Radiative Transfer & Dynamic RTE Studio")
    st.markdown(r\"\"\"
    **The Coupled Boundary-Value Radiative Transfer Equation:**
    $$L_{\\mathrm{sensor}}(\\lambda) = \\tau(\\lambda) \\left[ \\epsilon(\\lambda) B(\\lambda, T_s) + (1-\\epsilon(\\lambda)) \\frac{L^\\downarrow(\\lambda)}{\\pi} \\right] + L^\\uparrow(\\lambda)$$
    Select any ground material sample, specify surface temperature $T_s$, pair with any of the 30 MODTRAN 5.2 atmospheric profiles,
    and dynamically toggle **individual or grouped radiative transfer components** to analyze photon interactions.
    \"\"\")

    if atm_catalog_df is None or len(atm_catalog_df) == 0:
        st.warning("Atmospheric look-up table catalog not found in `data/atmospheric_luts/`.")
    else:
        # Step 1: Atmosphere & Sample Selectors
        st.markdown("#### 1. Atmospheric Profile & Surface Sample Pairing")
        row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)

        with row1_c1:
            atm_models = atm_catalog_df["atm_model_name"].unique().tolist()
            selected_model = st.selectbox("Climatological Model:", atm_models, index=0)

        with row1_c2:
            model_df = atm_catalog_df[atm_catalog_df["atm_model_name"] == selected_model]
            h2o_scales = model_df["h2o_scale"].tolist()
            scale_labels = {
                r["h2o_scale"]: f"{r['h2o_scale']:.2f}x (PWV = {r['pwv_g_cm2']:.2f} g/cm²)"
                for _, r in model_df.iterrows()
            }
            selected_scale = st.selectbox(
                "Water Vapor Scaling (PWV):",
                h2o_scales,
                index=2 if len(h2o_scales) > 2 else 0,
                format_func=lambda x: scale_labels.get(x, f"{x:.2f}x")
            )

        with row1_c3:
            sensor_path = st.selectbox(
                "Observation Geometry:",
                ["ASTER Spaceborne TOA (100 km)", "HyTES Airborne Path (20 km)"],
                index=0
            )

        with row1_c4:
            catalog_options = catalog_df["spectrum_id"].tolist()
            spec_format = {
                r["spectrum_id"]: f"[{r['material_class']}] {r['name']} ({r['sample_id']})"
                for _, r in catalog_df.iterrows()
            }
            default_target_idx = [i for i, s in enumerate(catalog_options) if "quartz" in s.lower()]
            target_sel = st.selectbox(
                "Ground Surface Material:",
                options=catalog_options,
                index=default_target_idx[0] if default_target_idx else 0,
                format_func=lambda x: spec_format.get(x, x),
                key="rte_target_select_dynamic"
            )

        # Step 2: Surface Temperature Slider
        st.markdown("#### 2. Surface Thermodynamic State")
        temp_col1, temp_col2 = st.columns([3, 1])
        with temp_col1:
            sim_temp = st.slider(
                "Surface Kinetic Temperature Ts (K):",
                min_value=270.0, max_value=340.0, value=300.0, step=0.5,
                help="Kinetic temperature setpoint of the radiating ground."
            )
        with temp_col2:
            st.metric("Temperature (°C)", f"{sim_temp - 273.15:.2f} °C", delta=f"{sim_temp:.1f} K")"""

    new_tab5_top = """with tab5:
    st.subheader("🌤️ Atmospheric Radiative Transfer & Dynamic RTE Studio")
    
    st.markdown(r\"\"\"
    <div style="background-color: rgba(30, 41, 59, 0.45); border-left: 4px solid #3B82F6; padding: 14px 20px; border-radius: 6px; margin-bottom: 18px;">
        <p style="margin: 0 0 6px 0; font-weight: 600; font-size: 1.05em; color: #93C5FD;">Coupled Boundary-Value Radiative Transfer Equation (RTE):</p>
        <div style="font-size: 1.18em; text-align: center; margin: 10px 0; color: #F8FAFC;">
            $$L_{\\mathrm{sensor}}(\\lambda) = \\tau(\\lambda) \\left[ \\epsilon(\\lambda) B(\\lambda, T_s) + [1 - \\epsilon(\\lambda)] L^\\downarrow(\\lambda) \\right] + L^\\uparrow(\\lambda)$$
        </div>
        <p style="margin: 8px 0 0 0; font-size: 0.92em; color: #CBD5E1; line-height: 1.5;">
            <b>Library & Physical Origin:</b> This interactive studio couples any of the <b>241 pristine laboratory FTIR spectra</b> from the curated <b>ECOSTRESS Library v1.0</b> (141 geological/pedological/urban solids + 100 diversity-maximized botanical endmembers via $L_2$ Euclidean FPS) with <b>30 MODTRAN 5.2 look-up tables</b> across arbitrary kinetic temperatures $T_s$.
        </p>
    </div>
    \"\"\", unsafe_allow_html=True)

    if atm_catalog_df is None or len(atm_catalog_df) == 0:
        st.warning("Atmospheric look-up table catalog not found in `data/atmospheric_luts/`.")
    else:
        # Step 1: Atmosphere & Sample Selectors
        st.markdown("#### 1. Atmospheric Profile & Surface Sample Pairing")
        row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)

        with row1_c1:
            atm_models = atm_catalog_df["atm_model_name"].unique().tolist()
            selected_model = st.selectbox("1. Climatological Model:", atm_models, index=0)

        with row1_c2:
            model_df = atm_catalog_df[atm_catalog_df["atm_model_name"] == selected_model]
            h2o_scales = model_df["h2o_scale"].tolist()
            scale_labels = {
                r["h2o_scale"]: f"{r['h2o_scale']:.2f}x (PWV = {r['pwv_g_cm2']:.2f} g/cm²)"
                for _, r in model_df.iterrows()
            }
            selected_scale = st.selectbox(
                "2. Water Vapor Scaling (PWV):",
                h2o_scales,
                index=2 if len(h2o_scales) > 2 else 0,
                format_func=lambda x: scale_labels.get(x, f"{x:.2f}x"),
                help="Dynamically cascades to the 5 valid scaled moisture profiles for the selected climatological model."
            )

        with row1_c3:
            sensor_path = st.selectbox(
                "3. Observation Geometry:",
                ["ASTER Spaceborne TOA (100 km)", "HyTES Airborne Path (20 km)"],
                index=0
            )

        with row1_c4:
            catalog_options = catalog_df["spectrum_id"].tolist()
            spec_format = {
                r["spectrum_id"]: f"[{r['material_class'].upper()}] {r['name']} ({r['sample_id']})"
                for _, r in catalog_df.iterrows()
            }
            default_target_idx = [i for i, s in enumerate(catalog_options) if "quartz_1" in s.lower()]
            target_sel = st.selectbox(
                "4. Ground Surface (241 Targets):",
                options=catalog_options,
                index=default_target_idx[0] if default_target_idx else 0,
                format_func=lambda x: spec_format.get(x, x),
                key="rte_target_select_dynamic",
                help="Choose from the complete universe of 241 curated ECOSTRESS laboratory spectra."
            )

        # Climatological Context Guidance Banner
        clima_context = {
            "Tropical": "15°N maritime atmosphere | High moisture column (PWV 2.06 – 8.23 g/cm²) | Typical surface Ts: 295 – 320 K",
            "Mid-Latitude Summer": "45°N summer continental | Moderate moisture column (PWV 1.46 – 5.84 g/cm²) | Typical surface Ts: 290 – 315 K",
            "Mid-Latitude Winter": "45°N winter continental | Low moisture column (PWV 0.42 – 1.70 g/cm²) | Typical surface Ts: 265 – 290 K",
            "Sub-Arctic Summer": "60°N summer sub-polar | Moderate-low moisture (PWV 1.04 – 4.18 g/cm²) | Typical surface Ts: 280 – 300 K",
            "Sub-Arctic Winter": "60°N winter polar vortex | Ultra-dry cold column (PWV 0.21 – 0.84 g/cm²) | Typical surface Ts: 250 – 275 K",
            "1976 US Standard": "Global mean reference atmosphere | Mid moisture (PWV 0.71 – 2.84 g/cm²) | Typical surface Ts: 285 – 305 K"
        }
        st.caption(f"ℹ️ **Climatological Context:** {clima_context.get(selected_model, 'Standard MODTRAN 5.2 Atmosphere')}")

        # Step 2: Surface Temperature Slider
        st.markdown("#### 2. Surface Thermodynamic State")
        temp_col1, temp_col2 = st.columns([3, 1])
        with temp_col1:
            sim_temp = st.slider(
                "Surface Kinetic Temperature Ts (K):",
                min_value=250.0, max_value=340.0, value=300.0, step=0.5,
                help="Kinetic temperature setpoint of the radiating ground."
            )
        with temp_col2:
            st.metric("Surface Temperature", f"{sim_temp:.1f} K", help="Kinetic temperature in Kelvin")
            st.caption(f"Equivalent: **{sim_temp - 273.15:.2f} °C**")"""

    if old_tab5_top in norm:
        norm = norm.replace(old_tab5_top, new_tab5_top)
        print(f"Patched Tab 5 top in {fpath}")
    else:
        print(f"Tab 5 top target not found in {fpath}")

    # 2. Fix the double-division by pi in Tab 5 forward calculation
    old_refl_calc = 'sim_l_refl = (1.0 - sim_emis) * prof_df["L_down_surface"].values / np.pi'
    new_refl_calc = 'sim_l_refl = (1.0 - sim_emis) * prof_df["L_down_surface"].values'
    if old_refl_calc in norm:
        norm = norm.replace(old_refl_calc, new_refl_calc)
        print(f"Patched sim_l_refl in {fpath}")

    # 3. Fix L_refl label in all_comp_defs
    old_refl_label = '"L_refl": {"label": "Reflected Downwelling Sky Radiance: L_refl = (1-ε)·L↓/π"'
    new_refl_label = '"L_refl": {"label": "Reflected Downwelling Sky Radiance: L_refl = (1-ε)·L↓"'
    if old_refl_label in norm:
        norm = norm.replace(old_refl_label, new_refl_label)
        print(f"Patched L_refl label in {fpath}")

    # 4. Fix Step 4 telemetry metrics and remove misleading green up-arrows
    old_eb_metrics = """            st.markdown("#### 4. Photon Energy Budget Telemetry (8.0 - 12.0 μm Atmospheric Window)")
            eb1, eb2, eb3, eb4 = st.columns(4)
            with eb1:
                st.metric("Direct Ground Emission (τ·ε·B)", f"{emit_pct:.1f}%", help="Percentage of detected photons originating directly from surface kinetic temperature.")
            with eb2:
                st.metric("Reflected Sky Irradiance (τ·L_refl)", f"{refl_pct:.1f}%", help="Downwelling sky radiance reflected by the ground into the sensor path.")
            with eb3:
                st.metric("Atmospheric Path Radiance (L↑)", f"{path_pct:.1f}%", help="Upwelling thermal photons emitted by atmospheric gases directly toward the sensor.")
            with eb4:
                st.metric("Mean Window Transmittance τ̄", f"{mean_tau_val:.3f}", delta=f"PWV: {selected_prof_row['pwv_g_cm2']:.2f} g/cm²")"""

    new_eb_metrics = """            st.markdown("#### 4. Photon Energy Budget Telemetry (8.0 - 12.0 μm Atmospheric Window)")
            eb1, eb2, eb3, eb4 = st.columns(4)
            with eb1:
                st.metric("Direct Ground Emission (τ·ε·B)", f"{emit_pct:.1f}%", help="Percentage of detected photons originating directly from surface kinetic temperature.")
            with eb2:
                st.metric("Reflected Sky Radiance (τ·L_refl)", f"{refl_pct:.1f}%", help="Downwelling sky radiance reflected by the ground into the sensor path.")
            with eb3:
                st.metric("Atmospheric Path Radiance (L↑)", f"{path_pct:.1f}%", help="Upwelling thermal photons emitted by atmospheric gases directly toward the sensor.")
            with eb4:
                st.metric("Mean Window Transmittance τ̄", f"{mean_tau_val:.3f}", help="Average atmospheric transmittance across the 8.0 - 12.0 μm window.")
                st.caption(f"Atmospheric PWV: **{selected_prof_row['pwv_g_cm2']:.2f} g/cm²**")"""

    if old_eb_metrics in norm:
        norm = norm.replace(old_eb_metrics, new_eb_metrics)
        print(f"Patched eb_metrics in {fpath}")

    # 5. Fix Tab 6 delta arrows and l_refl / np.pi
    old_tab6_mean_ts = """            mean_t = filtered_samples["ts_k"].mean() if len(filtered_samples) > 0 else 0.0
            st.metric("Mean Surface Ts", f"{mean_t:.1f} K", delta=f"{mean_t - 273.15:.1f} °C")"""
    new_tab6_mean_ts = """            mean_t = filtered_samples["ts_k"].mean() if len(filtered_samples) > 0 else 0.0
            st.metric("Mean Surface Ts", f"{mean_t:.1f} K")
            st.caption(f"Equivalent: **{mean_t - 273.15:.1f} °C**")"""
    if old_tab6_mean_ts in norm:
        norm = norm.replace(old_tab6_mean_ts, new_tab6_mean_ts)
        print(f"Patched tab6_mean_ts in {fpath}")

    old_tab6_refl = "l_refl = (1.0 - emis) * (l_down / np.pi)"
    new_tab6_refl = "l_refl = (1.0 - emis) * l_down"
    if old_tab6_refl in norm:
        norm = norm.replace(old_tab6_refl, new_tab6_refl)
        print(f"Patched tab6_refl in {fpath}")

    old_tab6_p_metrics = """            with p1:
                st.metric("Kinetic Temp Ts", f"{ts_val:.2f} K", delta=f"{ts_val - 273.15:.2f} °C")
            with p2:
                mmd_val = float(emis.max() - emis.min())
                st.metric("Spectral Contrast (MMD)", f"{mmd_val:.4f}", help="Min-Max Difference: measure of spectral diagnostic strength.")
            with p3:
                emit_fraction = (np.mean(l_emit) / np.mean(l_surf)) * 100.0 if np.mean(l_surf) > 0 else 0.0
                st.metric("Emitted Photon Fraction", f"{emit_fraction:.1f}%", help="Percentage of surface signal originating from thermal blackbody emission.")
            with p4:
                st.metric("Atmospheric Profile & PWV", atm_id, delta=f"PWV: {pwv_val:.2f} g/cm²")"""

    new_tab6_p_metrics = """            with p1:
                st.metric("Kinetic Temp Ts", f"{ts_val:.2f} K")
                st.caption(f"Equivalent: **{ts_val - 273.15:.2f} °C**")
            with p2:
                mmd_val = float(emis.max() - emis.min())
                st.metric("Spectral Contrast (MMD)", f"{mmd_val:.4f}", help="Min-Max Difference: measure of spectral diagnostic strength.")
            with p3:
                emit_fraction = (np.mean(l_emit) / np.mean(l_surf)) * 100.0 if np.mean(l_surf) > 0 else 0.0
                st.metric("Emitted Photon Fraction", f"{emit_fraction:.1f}%", help="Percentage of surface signal originating from thermal blackbody emission.")
            with p4:
                st.metric("Atmospheric Profile", atm_id)
                st.caption(f"Water Vapor Column: **{pwv_val:.2f} g/cm²**")"""

    if old_tab6_p_metrics in norm:
        norm = norm.replace(old_tab6_p_metrics, new_tab6_p_metrics)
        print(f"Patched tab6_p_metrics in {fpath}")

    updated = norm.replace("\n", "\r\n") if is_crlf else norm
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"Successfully saved {fpath}")

if __name__ == "__main__":
    patch_file("zenodo_release/dataset_explorer_dashboard.py")
    patch_file("scripts/dataset_explorer_dashboard.py")


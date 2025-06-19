"""
Improved 3-Tab Gradio interface for the Image Remapping Suite

This module provides a clean, organized web interface with three main tabs:
1. Distortion Simulation - Core distortion modeling and preview
2. Correction & Analysis - Method comparison and quality assessment  
3. Export & Data - Data export and utilities
"""

import gradio as gr
import random
from application.processor import processor
from config.settings import (
    DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT,
    DEFAULT_GRID_ROWS, DEFAULT_GRID_COLS,
    DEFAULT_K1, DEFAULT_K2, DEFAULT_K3, DEFAULT_P1, DEFAULT_P2,
    DEFAULT_PATTERN_TYPE, PATTERN_TYPES,
    MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH, MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT,
    MIN_GRID_SIZE, MAX_GRID_SIZE, VALIDATION_RANGES,
    DEFAULT_GDC_WIDTH, DEFAULT_GDC_HEIGHT
)

def create_gradio_interface():
    """
    Create the improved 3-tab Gradio interface for image remapping
    """
    
    with gr.Blocks(title="Image Remapping Suite", theme=gr.themes.Soft()) as interface:
        
        # Header
        gr.Markdown("""
        # 🔍 Image Remapping Suite - Advanced Lens Distortion Toolkit
        
        Professional-grade tool for lens distortion simulation, correction algorithms, and quality assessment.
        Use the tabs below to navigate through different aspects of the workflow.
        """)
        
        # Shared state components (hidden)
        with gr.Row(visible=False):
            shared_image_width = gr.State(DEFAULT_IMAGE_WIDTH)
            shared_image_height = gr.State(DEFAULT_IMAGE_HEIGHT)
            shared_grid_rows = gr.State(DEFAULT_GRID_ROWS)
            shared_grid_cols = gr.State(DEFAULT_GRID_COLS)
            shared_k1 = gr.State(DEFAULT_K1)
            shared_k2 = gr.State(DEFAULT_K2)
            shared_k3 = gr.State(DEFAULT_K3)
            shared_p1 = gr.State(DEFAULT_P1)
            shared_p2 = gr.State(DEFAULT_P2)
            shared_pattern = gr.State(DEFAULT_PATTERN_TYPE)
        
        # Tab Interface
        with gr.Tabs():
            
            # =================== TAB 1: DISTORTION SIMULATION ===================
            with gr.Tab("🌀 Distortion Simulation", id="simulation"):
                gr.Markdown("""
                ### Configure distortion parameters and see real-time preview
                Adjust parameters below to simulate different lens distortion effects using the Brown-Conrady model.
                """)
                
                with gr.Row():
                    # Left Column - Controls
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📐 Image Configuration")
                        
                        sim_image_width = gr.Slider(
                            minimum=MIN_IMAGE_WIDTH, maximum=MAX_IMAGE_WIDTH, step=64, 
                            value=DEFAULT_IMAGE_WIDTH,
                            label="Image Width", info="Width in pixels"
                        )
                        
                        sim_image_height = gr.Slider(
                            minimum=MIN_IMAGE_HEIGHT, maximum=MAX_IMAGE_HEIGHT, step=48, 
                            value=DEFAULT_IMAGE_HEIGHT,
                            label="Image Height", info="Height in pixels"
                        )
                        
                        with gr.Row():
                            sim_grid_rows = gr.Slider(
                                minimum=MIN_GRID_SIZE, maximum=MAX_GRID_SIZE, step=2, 
                                value=DEFAULT_GRID_ROWS,
                                label="Grid Rows", info="Number of grid rows"
                            )
                            
                            sim_grid_cols = gr.Slider(
                                minimum=MIN_GRID_SIZE, maximum=MAX_GRID_SIZE, step=2, 
                                value=DEFAULT_GRID_COLS,
                                label="Grid Columns", info="Number of grid columns"
                            )
                        
                        gr.Markdown("#### 🌀 Distortion Parameters")
                        
                        with gr.Row():
                            sim_k1 = gr.Slider(
                                minimum=VALIDATION_RANGES['k1'][0], maximum=VALIDATION_RANGES['k1'][1], 
                                step=0.01, value=DEFAULT_K1,
                                label="K1 (Primary Radial)", info="Main barrel/pincushion"
                            )
                            
                            sim_k2 = gr.Slider(
                                minimum=VALIDATION_RANGES['k2'][0], maximum=VALIDATION_RANGES['k2'][1], 
                                step=0.01, value=DEFAULT_K2,
                                label="K2 (Secondary Radial)", info="Higher-order correction"
                            )
                        
                        with gr.Row():
                            sim_k3 = gr.Slider(
                                minimum=VALIDATION_RANGES['k3'][0], maximum=VALIDATION_RANGES['k3'][1], 
                                step=0.005, value=DEFAULT_K3,
                                label="K3 (Tertiary Radial)", info="Extreme distortion"
                            )
                            
                            sim_p1 = gr.Slider(
                                minimum=VALIDATION_RANGES['p1'][0], maximum=VALIDATION_RANGES['p1'][1], 
                                step=0.005, value=DEFAULT_P1,
                                label="P1 (Tangential)", info="Decentering correction"
                            )
                        
                        sim_p2 = gr.Slider(
                            minimum=VALIDATION_RANGES['p2'][0], maximum=VALIDATION_RANGES['p2'][1], 
                            step=0.005, value=DEFAULT_P2,
                            label="P2 (Tangential)", info="Secondary decentering"
                        )
                        
                        gr.Markdown("#### 🎯 Pattern & Presets")
                        
                        sim_pattern_type = gr.Dropdown(
                            choices=PATTERN_TYPES,
                            value=DEFAULT_PATTERN_TYPE,
                            label="Test Pattern", info="Pattern for demonstration"
                        )
                        
                        with gr.Row():
                            preset_barrel = gr.Button("📷 Barrel", variant="secondary", size="sm")
                            preset_pincushion = gr.Button("📐 Pincushion", variant="secondary", size="sm")
                            preset_fisheye = gr.Button("🐟 Fisheye", variant="secondary", size="sm")
                            preset_reset = gr.Button("🔄 Reset", variant="secondary", size="sm")
                        
                        simulate_btn = gr.Button("🔄 Update Simulation", variant="primary", size="lg")
                        
                        # Parameter validation display
                        param_validation = gr.Textbox(
                            label="Parameter Validation",
                            lines=4,
                            placeholder="Parameter validation results will appear here...",
                            interactive=False
                        )
                    
                    # Right Column - Preview
                    with gr.Column(scale=2):
                        gr.Markdown("#### 🖼️ Real-time Preview")
                        
                        with gr.Row():
                            sim_original_out = gr.Image(label="Original Pattern", height=250)
                            sim_distorted_out = gr.Image(label="Distorted", height=250)
                            sim_corrected_out = gr.Image(label="Corrected", height=250)
                        
                        with gr.Tabs():
                            with gr.Tab("Grid Visualization"):
                                sim_grid_vis = gr.Image(label="Distortion Grid Mapping", height=400)
                            
                            with gr.Tab("Distortion Heatmap"):
                                sim_heatmap = gr.Image(label="Distortion Magnitude", height=400)
                        
                        gr.Markdown("""
                        **Understanding the Preview:**
                        - **Grid Visualization**: Shows how regular grid points are displaced by distortion
                        - **Heatmap**: Color-coded distortion magnitude (red = high, blue = low)
                        - **K1 < 0**: Barrel distortion (wide-angle effect)
                        - **K1 > 0**: Pincushion distortion (telephoto effect)
                        """)
            
            # =================== TAB 2: CORRECTION & ANALYSIS ===================
            with gr.Tab("⚖️ Correction & Analysis", id="analysis"):
                gr.Markdown("""
                ### Method comparison, quality assessment, and validation
                Compare different correction algorithms and analyze their performance.
                """)
                
                with gr.Row():
                    # Left Column - Controls
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🔧 Correction Methods")
                        
                        correction_method = gr.Dropdown(
                            choices=["iterative", "analytical", "polynomial", "original"],
                            value="iterative",
                            label="Correction Algorithm", 
                            info="Select correction method"
                        )
                        
                        gr.Markdown("""
                        **Method Guide:**
                        - **Iterative**: Most accurate (Newton-Raphson)
                        - **Analytical**: Fastest for K1-only distortion  
                        - **Polynomial**: Good speed/accuracy balance
                        - **Original**: Basic forward mapping
                        """)
                        
                        gr.Markdown("#### 🎯 Analysis Tools")
                        
                        with gr.Row():
                            validate_btn = gr.Button("✅ Validate Quality", variant="primary")
                            compare_btn = gr.Button("⚖️ Compare Methods", variant="secondary")
                        
                        with gr.Row():
                            accuracy_btn = gr.Button("🎯 Test Accuracy", variant="secondary")
                            summary_btn = gr.Button("📊 Show Summary", variant="secondary")
                        
                        # Results Display
                        validation_output = gr.Textbox(
                            label="Quality Validation Results",
                            lines=12,
                            placeholder="Click 'Validate Quality' to analyze correction performance...",
                            show_copy_button=True
                        )
                        
                        accuracy_output = gr.Textbox(
                            label="Accuracy Test Results",
                            lines=8,
                            placeholder="Click 'Test Accuracy' to compare correction methods...",
                            show_copy_button=True
                        )
                        
                        summary_output = gr.Textbox(
                            label="Processing Summary",
                            lines=6,
                            placeholder="Click 'Show Summary' to view current configuration...",
                            show_copy_button=True
                        )
                    
                    # Right Column - Analysis Results
                    with gr.Column(scale=2):
                        gr.Markdown("#### 📊 Analysis Results")
                        
                        with gr.Tabs():
                            with gr.Tab("Method Comparison"):
                                method_comparison_plot = gr.Image(
                                    label="Method Performance Comparison", height=400
                                )
                            
                            with gr.Tab("Quality Metrics"):
                                quality_metrics_plot = gr.Image(
                                    label="Quality Assessment Visualization", height=400
                                )
                            
                            with gr.Tab("Custom Image Processing"):
                                gr.Markdown("#### Upload Your Own Image")
                                
                                custom_image_input = gr.Image(
                                    label="Upload Image",
                                    type="numpy",
                                    height=200
                                )
                                
                                with gr.Row():
                                    custom_method = gr.Dropdown(
                                        choices=["iterative", "analytical", "polynomial", "original"],
                                        value="iterative",
                                        label="Correction Method"
                                    )
                                    process_custom_btn = gr.Button("Process Custom Image", variant="primary")
                                
                                with gr.Row():
                                    custom_distorted_out = gr.Image(label="Distorted Result", height=200)
                                    custom_corrected_out = gr.Image(label="Corrected Result", height=200)
                        
                        gr.Markdown("""
                        **Quality Thresholds:**
                        - **PSNR**: >30dB (good), >40dB (excellent)
                        - **Correlation**: >0.95 (good), >0.99 (excellent)  
                        - **Geometric Error**: <0.5px (excellent), <1.0px (good)
                        """)
            
            # =================== TAB 3: EXPORT & DATA ===================
            with gr.Tab("📁 Export & Data", id="export"):
                gr.Markdown("""
                ### Data export, hardware integration, and utilities
                Export calibration data in various formats for analysis and hardware implementation.
                """)
                
                with gr.Row():
                    # Left Column - Export Controls
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📊 Standard Export")
                        
                        with gr.Row():
                            export_csv_btn = gr.Button("📊 Export Grid CSV", variant="primary")
                            export_package_btn = gr.Button("📦 Export Package", variant="secondary")
                        
                        csv_output = gr.Textbox(
                            label="Grid Displacement CSV",
                            lines=8,
                            placeholder="Click 'Export Grid CSV' to generate displacement data...",
                            show_copy_button=True
                        )
                        
                        gr.Markdown("#### 🎯 GDC Hardware Format")
                        
                        with gr.Row():
                            gdc_width = gr.Number(
                                value=DEFAULT_GDC_WIDTH, minimum=1024, maximum=16384,
                                label="GDC Width", info="Target width for GDC format"
                            )
                            gdc_height = gr.Number(
                                value=DEFAULT_GDC_HEIGHT, minimum=768, maximum=12288,
                                label="GDC Height", info="Target height for GDC format"
                            )
                        
                        export_gdc_btn = gr.Button("🎯 Export GDC CSV", variant="primary")
                        
                        gdc_csv_output = gr.Textbox(
                            label="GDC Format CSV",
                            lines=8,
                            placeholder="Click 'Export GDC CSV' to generate hardware-ready data...",
                            show_copy_button=True
                        )
                        
                        gr.Markdown("#### 📈 Processing History")
                        
                        with gr.Row():
                            history_btn = gr.Button("📜 View History", variant="secondary")
                            clear_history_btn = gr.Button("🗑️ Clear History", variant="secondary")
                        
                        history_output = gr.Textbox(
                            label="Processing History",
                            lines=6,
                            placeholder="Processing history will appear here...",
                            show_copy_button=True
                        )
                    
                    # Right Column - Documentation & Utilities
                    with gr.Column(scale=2):
                        gr.Markdown("#### 📚 Documentation & Help")
                        
                        with gr.Tabs():
                            with gr.Tab("Export Formats"):
                                gr.Markdown("""
                                #### Available Export Formats
                                
                                **Grid Displacement CSV:**
                                - Contains displacement vectors for each grid point
                                - Format: `grid_x_row_col, value_decimal, value_hex`
                                - Suitable for analysis and visualization
                                
                                **GDC Format CSV:**
                                - Hardware-ready fixed-point values
                                - Compatible with FPGA/ISP implementations
                                - Includes bit-shifting and scaling for target resolution
                                
                                **Calibration Package:**
                                - Complete parameter set with metadata
                                - Multiple formats: JSON, XML, YAML
                                - Includes validation data and quality metrics
                                """)
                            
                            with gr.Tab("Usage Guide"):
                                gr.Markdown("""
                                #### Workflow Guide
                                
                                **1. Distortion Simulation Tab:**
                                - Configure image dimensions and grid size
                                - Adjust distortion parameters (K1, K2, K3, P1, P2)
                                - Use presets for common distortion types
                                - Monitor real-time preview
                                
                                **2. Correction & Analysis Tab:**
                                - Select appropriate correction method
                                - Validate correction quality
                                - Compare different algorithms
                                - Process custom images
                                
                                **3. Export & Data Tab:**
                                - Export grid data in various formats
                                - Generate hardware-compatible files
                                - Access processing history
                                - Download complete calibration packages
                                """)
                            
                            with gr.Tab("Technical Info"):
                                gr.Markdown("""
                                #### Technical Specifications
                                
                                **Distortion Model:**
                                - Brown-Conrady model with radial and tangential components
                                - Supports up to 3rd-order radial distortion (K1, K2, K3)
                                - Tangential distortion correction (P1, P2)
                                
                                **Correction Algorithms:**
                                - **Iterative**: Newton-Raphson inverse mapping
                                - **Analytical**: Closed-form solution for simple cases
                                - **Polynomial**: Approximation for moderate distortions
                                - **Original**: Basic forward mapping correction
                                
                                **Quality Metrics:**
                                - PSNR (Peak Signal-to-Noise Ratio)
                                - Image correlation coefficient
                                - Geometric error analysis
                                - MSE/MAE image metrics
                                """)
                            
                            with gr.Tab("System Status"):
                                system_info = gr.Textbox(
                                    label="System Information",
                                    lines=10,
                                    value=get_system_info(),
                                    interactive=False
                                )
                                
                                refresh_info_btn = gr.Button("🔄 Refresh Info", variant="secondary")
        
        # Event Handlers
        setup_event_handlers(
            interface,
            # Simulation tab components
            sim_image_width, sim_image_height, sim_grid_rows, sim_grid_cols,
            sim_k1, sim_k2, sim_k3, sim_p1, sim_p2, sim_pattern_type,
            preset_barrel, preset_pincushion, preset_fisheye, preset_reset,
            simulate_btn, param_validation,
            sim_original_out, sim_distorted_out, sim_corrected_out,
            sim_grid_vis, sim_heatmap,
            # Analysis tab components
            correction_method, validate_btn, compare_btn, accuracy_btn, summary_btn,
            validation_output, accuracy_output, summary_output,
            method_comparison_plot, quality_metrics_plot,
            custom_image_input, custom_method, process_custom_btn,
            custom_distorted_out, custom_corrected_out,
            # Export tab components
            export_csv_btn, export_package_btn, csv_output,
            gdc_width, gdc_height, export_gdc_btn, gdc_csv_output,
            history_btn, clear_history_btn, history_output,
            system_info, refresh_info_btn,
            # Shared state
            shared_image_width, shared_image_height, shared_grid_rows, shared_grid_cols,
            shared_k1, shared_k2, shared_k3, shared_p1, shared_p2, shared_pattern
        )
    
    return interface

def setup_event_handlers(interface, *components):
    """Setup all event handlers for the interface"""
    
    # Unpack components
    (sim_image_width, sim_image_height, sim_grid_rows, sim_grid_cols,
     sim_k1, sim_k2, sim_k3, sim_p1, sim_p2, sim_pattern_type,
     preset_barrel, preset_pincushion, preset_fisheye, preset_reset,
     simulate_btn, param_validation,
     sim_original_out, sim_distorted_out, sim_corrected_out,
     sim_grid_vis, sim_heatmap,
     correction_method, validate_btn, compare_btn, accuracy_btn, summary_btn,
     validation_output, accuracy_output, summary_output,
     method_comparison_plot, quality_metrics_plot,
     custom_image_input, custom_method, process_custom_btn,
     custom_distorted_out, custom_corrected_out,
     export_csv_btn, export_package_btn, csv_output,
     gdc_width, gdc_height, export_gdc_btn, gdc_csv_output,
     history_btn, clear_history_btn, history_output,
     system_info, refresh_info_btn,
     shared_image_width, shared_image_height, shared_grid_rows, shared_grid_cols,
     shared_k1, shared_k2, shared_k3, shared_p1, shared_p2, shared_pattern) = components
    
    # Helper functions for presets
    def update_sliders_from_preset(preset_name):
        presets = {
            'barrel': {'k1': -0.2, 'k2': 0.05, 'k3': 0.0, 'p1': 0.0, 'p2': 0.0},
            'pincushion': {'k1': 0.15, 'k2': -0.03, 'k3': 0.0, 'p1': 0.0, 'p2': 0.0},
            'fisheye': {'k1': -0.4, 'k2': 0.1, 'k3': -0.02, 'p1': 0.0, 'p2': 0.0},
            'reset': {'k1': DEFAULT_K1, 'k2': DEFAULT_K2, 'k3': DEFAULT_K3, 'p1': DEFAULT_P1, 'p2': DEFAULT_P2}
        }
        if preset_name in presets:
            preset = presets[preset_name]
            return (preset['k1'], preset['k2'], preset['k3'], preset['p1'], preset['p2'])
        return (DEFAULT_K1, DEFAULT_K2, DEFAULT_K3, DEFAULT_P1, DEFAULT_P2)
    
    def validate_parameters_func(k1, k2, k3, p1, p2):
        validation_result = processor.validate_parameters(k1, k2, k3, p1, p2)
        
        result_text = []
        if validation_result['errors']:
            result_text.append("❌ ERRORS:")
            result_text.extend([f"  • {error}" for error in validation_result['errors']])
        
        if validation_result['warnings']:
            result_text.append("⚠️ WARNINGS:")
            result_text.extend([f"  • {warning}" for warning in validation_result['warnings']])
        
        if validation_result['recommendations']:
            result_text.append("💡 RECOMMENDATIONS:")
            result_text.extend([f"  • {rec}" for rec in validation_result['recommendations']])
        
        if not result_text:
            result_text.append("✅ All parameters are within acceptable ranges")
        
        return "\n".join(result_text)
    
    def update_shared_state(width, height, rows, cols, k1, k2, k3, p1, p2, pattern):
        return width, height, rows, cols, k1, k2, k3, p1, p2, pattern
    
    # === SIMULATION TAB EVENTS ===
    
    # Preset buttons
    preset_barrel.click(
        fn=lambda: update_sliders_from_preset('barrel'),
        outputs=[sim_k1, sim_k2, sim_k3, sim_p1, sim_p2]
    )
    
    preset_pincushion.click(
        fn=lambda: update_sliders_from_preset('pincushion'),
        outputs=[sim_k1, sim_k2, sim_k3, sim_p1, sim_p2]
    )
    
    preset_fisheye.click(
        fn=lambda: update_sliders_from_preset('fisheye'),
        outputs=[sim_k1, sim_k2, sim_k3, sim_p1, sim_p2]
    )
    
    preset_reset.click(
        fn=lambda: update_sliders_from_preset('reset'),
        outputs=[sim_k1, sim_k2, sim_k3, sim_p1, sim_p2]
    )
    
    # Main simulation update
    simulate_btn.click(
        fn=processor.process_distortion,
        inputs=[sim_image_width, sim_image_height, sim_grid_rows, sim_grid_cols,
               sim_k1, sim_k2, sim_k3, sim_p1, sim_p2, sim_pattern_type, gr.State("iterative")],
        outputs=[sim_original_out, sim_distorted_out, sim_corrected_out, sim_grid_vis, sim_heatmap]
    ).then(
        fn=validate_parameters_func,
        inputs=[sim_k1, sim_k2, sim_k3, sim_p1, sim_p2],
        outputs=param_validation
    ).then(
        fn=update_shared_state,
        inputs=[sim_image_width, sim_image_height, sim_grid_rows, sim_grid_cols,
               sim_k1, sim_k2, sim_k3, sim_p1, sim_p2, sim_pattern_type],
        outputs=[shared_image_width, shared_image_height, shared_grid_rows, shared_grid_cols,
                shared_k1, shared_k2, shared_k3, shared_p1, shared_p2, shared_pattern]
    )
    
    # === ANALYSIS TAB EVENTS ===
    
    validate_btn.click(
        fn=processor.validate_correction_quality,
        inputs=[shared_pattern, correction_method],
        outputs=validation_output
    )
    
    accuracy_btn.click(
        fn=processor.test_correction_accuracy,
        outputs=accuracy_output
    )
    
    summary_btn.click(
        fn=processor.get_processing_summary,
        outputs=summary_output
    )
    
    # Custom image processing
    process_custom_btn.click(
        fn=processor.process_custom_image,
        inputs=[custom_image_input, custom_method],
        outputs=[custom_distorted_out, custom_corrected_out]
    )
    
    # === EXPORT TAB EVENTS ===
    
    export_csv_btn.click(
        fn=processor.export_grid_csv,
        outputs=csv_output
    )
    
    export_gdc_btn.click(
        fn=processor.export_gdc_grid_csv,
        inputs=[gdc_width, gdc_height],
        outputs=gdc_csv_output
    )
    
    # System info refresh
    refresh_info_btn.click(
        fn=get_system_info,
        outputs=system_info
    )
    
    # Auto-load with default values
    interface.load(
        fn=processor.process_distortion,
        inputs=[sim_image_width, sim_image_height, sim_grid_rows, sim_grid_cols,
               sim_k1, sim_k2, sim_k3, sim_p1, sim_p2, sim_pattern_type, gr.State("iterative")],
        outputs=[sim_original_out, sim_distorted_out, sim_corrected_out, sim_grid_vis, sim_heatmap]
    ).then(
        fn=validate_parameters_func,
        inputs=[sim_k1, sim_k2, sim_k3, sim_p1, sim_p2],
        outputs=param_validation
    )

def get_system_info():
    """Get system information for display"""
    import sys
    import numpy as np
    import cv2
    from datetime import datetime
    
    info_lines = [
        f"System Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 50,
        f"Python Version: {sys.version.split()[0]}",
        f"NumPy Version: {np.__version__}",
        f"OpenCV Version: {cv2.__version__}",
        "",
        "Application Status:",
        "✅ Image Remapping Suite - Ready",
        "✅ All dependencies loaded",
        "✅ Processor initialized",
        "",
        "Available Features:",
        "• Brown-Conrady distortion model",
        "• Multiple correction algorithms",
        "• Real-time parameter validation", 
        "• Quality assessment metrics",
        "• Hardware format export (GDC)",
        "• Custom image processing",
        "",
        f"Default Configuration:",
        f"• Image: {DEFAULT_IMAGE_WIDTH}x{DEFAULT_IMAGE_HEIGHT}",
        f"• Grid: {DEFAULT_GRID_ROWS}x{DEFAULT_GRID_COLS}",
        f"• Pattern: {DEFAULT_PATTERN_TYPE}",
    ]
    
    return "\n".join(info_lines)
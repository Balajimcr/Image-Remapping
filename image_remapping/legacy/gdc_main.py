#!/usr/bin/env python3
"""
GDC Image Remapping GUI Interface - Main Module

Advanced Gradio interface for GDC-based image geometric transformation with
comprehensive workflow management, real-time visualization, and quality assessment.

Author: Balaji R
License: MIT
"""

import gradio as gr
import numpy as np
import os
from typing import Tuple, Optional
from datetime import datetime

# Import the modular components
from gdc_image_processor import GDCImageRemappingProcessor
from gdc_visualizer import GDCVisualizer
from gdc_event_handlers import GDCEventHandlers
from gdc_interface_components import GDCInterfaceComponents

# Configuration Constants
DEFAULT_GRID_SIZES = [7, 9, 11, 15, 21, 33]
DEFAULT_IMAGE_SIZES = [(640, 480), (800, 600), (1024, 768), (1280, 720), (1920, 1080)]


class GDCRemappingInterface:
    """Main interface controller for GDC Image Remapping."""
    
    def __init__(self):
        # Initialize core components
        self.processor = GDCImageRemappingProcessor()
        self.visualizer = GDCVisualizer()
        self.event_handlers = GDCEventHandlers(self.processor, self.visualizer)
        self.interface_components = GDCInterfaceComponents()
        
        # Processing state
        self.processing_state = {
            'gdc_loaded': False,
            'grid_setup': False,
            'remapping_ready': False,
            'image_processed': False
        }
        
        # Component references (will be set during interface creation)
        self.components = {}
        
    def create_interface(self):
        """Create the main Gradio interface."""
        
        with gr.Blocks(
            title="GDC Image Remapping Suite",
            theme=gr.themes.Soft(),
            css=self.interface_components.get_custom_css()
        ) as interface:
            
            # Header
            gr.Markdown("""
            # 🎯 GDC Image Remapping Suite
            
            **Professional geometric image transformation using GDC (Geometric Distortion Correction) grids**
            
            Complete workflow: Import GDC → Setup Grid → Process Image → Export Results
            """)
            
            # Status indicators
            with gr.Row():
                status_indicators = gr.HTML(
                    value=self.interface_components.generate_status_html(self.processing_state),
                    elem_id="status_indicators"
                )
            
            # Main tabbed interface
            with gr.Tabs():
                
                # TAB 1: GDC DATA IMPORT
                with gr.Tab("📁 GDC Data Import"):
                    import_components = self.interface_components.create_import_tab_components()
                    self.components.update(import_components)
                
                # TAB 2: GRID PROCESSING
                with gr.Tab("🌐 Grid Processing"):
                    grid_components = self.interface_components.create_grid_tab_components()
                    self.components.update(grid_components)
                
                # TAB 3: IMAGE REMAPPING
                with gr.Tab("🖼️ Image Remapping"):
                    remap_components = self.interface_components.create_remapping_tab_components()
                    self.components.update(remap_components)
                
                # TAB 4: ANALYSIS & EXPORT
                with gr.Tab("📊 Analysis & Export"):
                    analysis_components = self.interface_components.create_analysis_tab_components()
                    self.components.update(analysis_components)
                
                # TAB 5: HELP
                with gr.Tab("❓ Help"):
                    self._create_help_tab()
            
            # Create sample buttons for import tab
            sample_buttons = self.interface_components.create_sample_buttons(
                self.components['gdc_text_input']
            )
            
            # Setup event handlers
            self._setup_event_handlers()
            
        return interface
    
    def _create_help_tab(self):
        """Create help and documentation tab."""
        gr.Markdown("""
        ### 📚 GDC Image Remapping Help & Documentation
        
        #### 🚀 Getting Started
        
        **1. Import GDC Data** 📁
        - Paste GDC grid data or upload a file
        - Use format: `yuv_gdc_grid_dx_0_N value` and `yuv_gdc_grid_dy_0_N value`
        - Specify grid dimensions to match your data
        
        **2. Setup Grid** 🌐
        - Configure original grid dimensions (from your data)
        - Set target dimensions (higher for smoother interpolation)
        - Choose interpolation method (bicubic recommended)
        
        **3. Process Image** 🖼️
        - Upload image or generate sample pattern
        - Adjust displacement scale factor (start with 0.1)
        - Configure image dimensions and interpolation method
        
        **4. Analyze & Export** 📊
        - Generate analysis visualizations
        - Export results in various formats
        - Review processing summary
        
        #### 📋 GDC Data Format
        
        ```
        yuv_gdc_grid_dx_0_0 -50    # X displacement for element 0
        yuv_gdc_grid_dx_0_1 0      # X displacement for element 1
        yuv_gdc_grid_dx_0_2 50     # X displacement for element 2
        yuv_gdc_grid_dy_0_0 25     # Y displacement for element 0
        yuv_gdc_grid_dy_0_1 30     # Y displacement for element 1
        yuv_gdc_grid_dy_0_2 25     # Y displacement for element 2
        ```
        
        #### ⚙️ Parameters Guide
        
        **Scale Factor** 🎯
        - Controls displacement magnitude
        - Start with 0.1 and adjust based on results
        - Higher values = more distortion
        
        **Interpolation Methods** 🔧
        - **Bicubic**: Smoothest results, slower processing
        - **Linear**: Good balance of quality and speed
        
        **Grid Sizes** 📐
        - Original: Determined by your GDC data
        - Target: Higher values = smoother interpolation
        - Typical scaling: 3×3 → 9×9 or 5×5 → 15×15
        
        #### 🚨 Troubleshooting
        
        **Common Issues:**
        
        1. **"No GDC data loaded"**
           - Ensure data is properly formatted
           - Check element naming convention
           - Verify complete DX/DY pairs
        
        2. **"Grid setup failed"**
           - Verify grid dimensions match data
           - Check for missing elementscles
           - Ensure positive grid sizes
        
        3. **"Excessive displacement"**
           - Reduce scale factor
           - Check GDC data validity
           - Verify grid interpolation
        """)
    
    def _setup_event_handlers(self):
        """Setup all event handlers for the interface."""
        
        # Import handlers
        self.components['import_btn'].click(
            fn=self.event_handlers.handle_import_gdc_simple,
            inputs=[
                self.components['gdc_text_input'], 
                self.components['gdc_file_input'],
                self.components['input_grid_rows'],
                self.components['input_grid_cols']
            ],
            outputs=[
                self.components['import_status'],
                self.components['import_stats'],
                self.components['data_preview']
            ]
        )
        
        # Grid handlers
        self.components['setup_grid_btn'].click(
            fn=self.event_handlers.handle_setup_grid,
            inputs=[
                self.components['orig_rows'],
                self.components['orig_cols'],
                self.components['target_rows'],
                self.components['target_cols'],
                self.components['interp_method']
            ],
            outputs=[
                self.components['grid_stats'],
                self.components['original_grid_viz'],
                self.components['interpolated_grid_viz'],
                self.components['grid_comparison_viz']
            ]
        )
        
        # Auto-update interpolation factor
        for component in [self.components['orig_rows'], self.components['orig_cols'], 
                         self.components['target_rows'], self.components['target_cols']]:
            component.change(
                fn=self.interface_components.update_interpolation_factor,
                inputs=[self.components['orig_rows'], self.components['orig_cols'], 
                       self.components['target_rows'], self.components['target_cols']],
                outputs=self.components['interp_factor']
            )
        
        # Auto-update expected elements
        for component in [self.components['input_grid_rows'], self.components['input_grid_cols']]:
            component.change(
                fn=self.interface_components.update_expected_elements,
                inputs=[self.components['input_grid_rows'], self.components['input_grid_cols']],
                outputs=self.components['expected_elements']
            )
        
        # Image processing handlers
        self.components['generate_sample_btn'].click(
            fn=self.event_handlers.handle_generate_sample_image,
            inputs=[self.components['sample_pattern'], self.components['img_width'], self.components['img_height']],
            outputs=self.components['input_image']
        )
        
        self.components['process_image_btn'].click(
            fn=self.event_handlers.handle_process_image,
            inputs=[
                self.components['input_image'],
                self.components['img_width'],
                self.components['img_height'],
                self.components['scale_factor'],
                self.components['cv_interpolation']
            ],
            outputs=[
                self.components['output_image'],
                self.components['image_comparison'],
                self.components['displacement_viz'],
                self.components['processing_status'],
                self.components['quality_metrics']
            ]
        )
        
        # Analysis handlers
        self.components['create_analysis_btn'].click(
            fn=self.event_handlers.handle_create_analysis,
            inputs=self.components['analysis_type'],
            outputs=[self.components['analysis_viz'], self.components['processing_summary']]
        )
        
        # Export handlers
        self.components['export_all_btn'].click(
            fn=self.event_handlers.handle_export_grid_data,
            inputs=[],
            outputs=[self.components['download_files'], self.components['export_status']]
        )


def create_gdc_interface():
    """Create and return the GDC remapping interface."""
    try:
        print("🚀 Initializing GDC Image Remapping Suite...")
        interface_controller = GDCRemappingInterface()
        interface = interface_controller.create_interface()
        print("✅ Interface created successfully")
        return interface
        
    except Exception as e:
        print(f"❌ Error creating interface: {e}")
        raise


def main():
    """Main entry point for the application."""
    try:
        print("=" * 60)
        print("🎯 GDC Image Remapping Suite")
        print("=" * 60)
        print("🚀 Starting application...")
        
        # Create the interface
        interface = create_gdc_interface()
        
        print("✅ Interface ready")
        print("🌐 Launching web interface...")
        print(f"🔗 Interface will be available at: http://localhost:7860")
        print("📋 Features available:")
        print("   ✓ GDC data import and validation")
        print("   ✓ Grid interpolation and visualization")
        print("   ✓ Image remapping and processing")
        print("   ✓ Analysis and export tools")
        print("   ✓ Comprehensive help documentation")
        print("\n🎉 Ready to process GDC data!")
        print("=" * 60)
        
        # Launch the interface
        interface.launch(
            server_name="localhost",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        
    except Exception as e:
        print(f"❌ Application error: {e}")
        raise


if __name__ == "__main__":
    main()
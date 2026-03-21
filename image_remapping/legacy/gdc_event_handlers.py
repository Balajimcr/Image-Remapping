#!/usr/bin/env python3
"""
GDC Event Handlers Module

Contains all event handling logic for the GDC remapping interface.
Separates UI logic from business logic for better maintainability.

Author: Balaji R
License: MIT
"""

import numpy as np
import os
import tempfile
import zipfile
import json
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from gdc_image_processor import GDCImageRemappingProcessor
from gdc_visualizer import GDCVisualizer


class GDCEventHandlers:
    """
    Event handling logic for GDC remapping interface.
    
    This class provides:
    - Import handling for GDC data
    - Grid processing event handlers
    - Image processing event handlers
    - Analysis and export handlers
    """
    
    def __init__(self, processor: GDCImageRemappingProcessor, visualizer: GDCVisualizer):
        self.processor = processor
        self.visualizer = visualizer
        self.processing_state = {
            'gdc_loaded': False,
            'grid_setup': False,
            'remapping_ready': False,
            'image_processed': False
        }
    
    def handle_import_gdc_simple(self, text_input: str, file_input, 
                            input_rows: int, input_cols: int) -> Tuple[str, dict, Optional[list]]:
        try:
            # Validate inputs
            if not isinstance(input_rows, (int, float)) or not isinstance(input_cols, (int, float)):
                return (
                    "⚠️ Invalid grid dimensions. Please enter valid numbers.",
                    {"error": "Invalid grid dimensions"},
                    None
                )
            input_rows = int(input_rows)
            input_cols = int(input_cols)
            if input_rows < 1 or input_cols < 1:
                return (
                    "⚠️ Grid dimensions must be positive integers.",
                    {"error": "Grid dimensions must be positive"},
                    None
                )
            
            # Determine input source
            content = ""
            source = ""
            if file_input is not None and hasattr(file_input, 'name'):
                with open(file_input.name, 'r', encoding='utf-8') as f:
                    content = f.read()
                source = f"file: {os.path.basename(file_input.name)}"
            elif text_input and text_input.strip():
                content = text_input
                source = "text input"
            else:
                return (
                    "⚠️ No input provided. Please enter GDC data or upload a file.",
                    {"error": "No input provided"},
                    None
                )
            
            # Basic format validation
            lines = content.strip().split('\n')
            valid_line_count = 0
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) == 2 and re.match(r"yuv_gdc_grid_(dx|dy)_0_\d+$", parts[0]):
                        valid_line_count += 1
            if valid_line_count == 0:
                return (
                    "⚠️ No valid GDC data found in input. Ensure format is 'yuv_gdc_grid_dx_0_N value' or 'yuv_gdc_grid_dy_0_N value'.",
                    {"error": "No valid GDC data found"},
                    None
                )
            
            # Clear previous data
            self.processor.clear_data()
            
            # Import the data
            result = self.processor.import_gdc_from_text(content)
            
            if result['success']:
                self.processing_state['gdc_loaded'] = True
                
                # Validate against expected dimensions
                validation = self.processor.grid_processor.validate_grid_data(input_rows, input_cols)
                
                # Create status message
                status_msg = f"✅ **Import Successful from {source}**\n\n"
                
                stats = result['statistics']
                status_msg += f"📊 **Data Summary:**\n"
                status_msg += f"   • Total Elements: {stats['total_elements']}\n"
                status_msg += f"   • DX Elements: {stats['dx_elements']}\n"
                status_msg += f"   • DY Elements: {stats['dy_elements']}\n"
                status_msg += f"   • Value Range: {stats['data_range']['min']} to {stats['data_range']['max']}\n\n"
                
                # Validation results
                if validation['valid']:
                    status_msg += f"✅ **Validation:** Passed\n"
                    status_msg += f"   • Expected grid: {input_rows}×{input_cols} = {input_rows * input_cols} elements\n"
                    status_msg += f"   • Data coverage: Complete\n\n"
                else:
                    status_msg += f"⚠️ **Validation:** Issues found\n"
                    status_msg += f"   • Expected grid: {input_rows}×{input_cols} = {input_rows * input_cols} elements\n"
                    for error in validation['errors']:
                        status_msg += f"   • ❌ {error}\n"
                    for warning in validation['warnings']:
                        status_msg += f"   • ⚠️ {warning}\n"
                    status_msg += "\n"
                
                # Add parsing warnings
                if result['warnings']:
                    status_msg += f"⚠️ **Parsing Warnings:**\n"
                    for warning in result['warnings'][:5]:
                        status_msg += f"   • {warning}\n"
                    if len(result['warnings']) > 5:
                        status_msg += f"   • ... and {len(result['warnings']) - 5} more warnings\n"
                    status_msg += "\n"
                
                status_msg += f"🎯 **Ready for grid setup!**"
                
                # Create preview data
                preview_data = self._create_preview_data()
                
                # Format statistics for display
                stats_output = {
                    'Import Statistics': result['statistics'],
                    'Validation Results': validation
                }
                
                return status_msg, stats_output, preview_data
                
            else:
                self.processing_state['gdc_loaded'] = False
                status_msg = f"❌ Import failed: {result['message']}\n\n"
                if result['warnings']:
                    status_msg += f"⚠️ **Parsing Warnings:**\n"
                    for warning in result['warnings'][:5]:
                        status_msg += f"   • {warning}\n"
                    if len(result['warnings']) > 5:
                        status_msg += f"   • ... and {len(result['warnings']) - 5} more warnings\n"
                return status_msg, {"error": result['message'], "warnings": result['warnings']}, None
                
        except Exception as e:
            self.processing_state['gdc_loaded'] = False
            self.processor.clear_data()
            return (
                f"❌ Import error: {str(e)}",
                {"error": f"Unexpected error: {str(e)}"},
                None
            )
    
    def handle_setup_grid(self, orig_rows: int, orig_cols: int, 
                         target_rows: int, target_cols: int, 
                         interp_method: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        """
        Handle grid setup and interpolation.
        
        Args:
            orig_rows: Original grid rows
            orig_cols: Original grid columns
            target_rows: Target grid rows
            target_cols: Target grid columns
            interp_method: Interpolation method
            
        Returns:
            Tuple of (status_message, original_viz, interpolated_viz, comparison_viz)
        """
        try:
            if not self.processing_state['gdc_loaded']:
                return "⚠️ Please import GDC data first", None, None, None
            
            # Setup grid with processor
            result = self.processor.setup_and_interpolate_grid(
                orig_rows, orig_cols, target_rows, target_cols, interp_method
            )
            
            if result['success']:
                self.processing_state['grid_setup'] = True
                
                # Generate visualizations
                original_viz = None
                interpolated_viz = None
                comparison_viz = None
                
                if self.processor.dx_grid is not None and self.processor.dy_grid is not None:
                    original_viz = self.visualizer.create_grid_heatmap(
                        self.processor.dx_grid, self.processor.dy_grid, "Original Grid"
                    )
                
                if (self.processor.interpolated_dx is not None and 
                    self.processor.interpolated_dy is not None):
                    interpolated_viz = self.visualizer.create_grid_heatmap(
                        self.processor.interpolated_dx, self.processor.interpolated_dy, 
                        "Interpolated Grid"
                    )
                    
                    # Create comparison if both grids exist
                    if original_viz is not None:
                        comparison_viz = self.visualizer.create_grid_comparison(
                            self.processor.dx_grid, self.processor.dy_grid,
                            self.processor.interpolated_dx, self.processor.interpolated_dy
                        )
                
                # Create enhanced status message
                grid_info = result.get('grid_info', {})
                statistics = result.get('statistics', {})
                
                stats_msg = f"✅ **Grid Setup Successful!**\n\n"
                stats_msg += "📐 **Grid Configuration:**\n"
                stats_msg += f"   • Original: {orig_rows}×{orig_cols} = {orig_rows * orig_cols} elements\n"
                stats_msg += f"   • Target: {target_rows}×{target_cols} = {target_rows * target_cols} elements\n"
                stats_msg += f"   • Interpolation: {interp_method.title()}\n"
                
                if 'scale_factor_x' in grid_info and 'scale_factor_y' in grid_info:
                    stats_msg += f"   • Scale factor: {grid_info['scale_factor_x']:.1f}× (cols) | {grid_info['scale_factor_y']:.1f}× (rows)\n\n"
                
                # Add grid statistics
                if statistics:
                    stats_msg += "📊 **Grid Statistics:**\n"
                    
                    for grid_type, stats in statistics.items():
                        if isinstance(stats, dict) and 'min' in stats:
                            grid_name = grid_type.replace('_', ' ').title()
                            stats_msg += f"   **{grid_name}:**\n"
                            stats_msg += f"     ├─ Range: {stats['min']:.1f} to {stats['max']:.1f}\n"
                            stats_msg += f"     ├─ Mean: {stats['mean']:.2f}\n"
                            stats_msg += f"     └─ Std Dev: {stats['std']:.2f}\n"
                
                stats_msg += "\n🎯 **Ready for image remapping!**"
                
                return stats_msg, original_viz, interpolated_viz, comparison_viz
            else:
                return f"❌ Grid setup failed: {result['message']}", None, None, None
                
        except Exception as e:
            return f"❌ Error during grid setup: {str(e)}", None, None, None
    
    def handle_process_image(self, input_image: np.ndarray, img_width: int, img_height: int,
                           scale_factor: float, cv_interpolation: str) -> Tuple[Optional[np.ndarray], 
                                                                               Optional[str], 
                                                                               Optional[str], 
                                                                               str, Dict]:
        """
        Handle image processing with current grid configuration.
        
        Args:
            input_image: Input image array
            img_width: Image width
            img_height: Image height
            scale_factor: Displacement scale factor
            cv_interpolation: OpenCV interpolation method
            
        Returns:
            Tuple of (output_image, comparison_viz, displacement_viz, status_message, quality_metrics)
        """
        try:
            if not self.processing_state['grid_setup']:
                return None, None, None, "⚠️ Please setup grid first", {}
            
            if input_image is None:
                return None, None, None, "⚠️ Please provide input image", {}
            
            # Setup image remapping
            setup_result = self.processor.setup_image_remapping(
                img_width, img_height, scale_factor, cv_interpolation
            )
            
            if not setup_result['success']:
                return None, None, None, f"❌ Remapping setup failed: {setup_result['message']}", {}
            
            self.processing_state['remapping_ready'] = True
            
            # Process the image
            process_result = self.processor.process_image(input_image)
            
            if process_result['success']:
                self.processing_state['image_processed'] = True
                
                # Create visualizations
                comparison_viz = self.visualizer.create_image_comparison(
                    input_image, process_result.get('output_image')
                )
                
                displacement_viz = None
                if (self.processor.dx_grid is not None and 
                    self.processor.dy_grid is not None):
                    displacement_viz = self.visualizer.create_displacement_analysis(
                        self.processor.dx_grid, self.processor.dy_grid, scale_factor
                    )
                
                # Create status message
                input_info = process_result.get('input_info', {})
                output_info = process_result.get('output_info', {})
                processing_stats = process_result.get('processing_stats', {})
                
                status_msg = f"✅ **Image Processing Successful!**\n\n"
                status_msg += f"📊 **Input Image:**\n"
                status_msg += f"   • Shape: {input_info.get('shape', 'Unknown')}\n"
                status_msg += f"   • Type: {input_info.get('dtype', 'Unknown')}\n"
                status_msg += f"   • Size: {input_info.get('size_mb', 0):.1f} MB\n\n"
                
                status_msg += f"📊 **Output Image:**\n"
                status_msg += f"   • Shape: {output_info.get('shape', 'Unknown')}\n"
                status_msg += f"   • Type: {output_info.get('dtype', 'Unknown')}\n"
                status_msg += f"   • Size: {output_info.get('size_mb', 0):.1f} MB\n\n"
                
                status_msg += f"⏱️ **Processing Info:**\n"
                status_msg += f"   • Method: {processing_stats.get('method', 'Unknown')}\n"
                status_msg += f"   • Scale Factor: {processing_stats.get('scale_factor_used', scale_factor)}\n"
                status_msg += f"   • Grid Shape: {processing_stats.get('grid_shape', 'Unknown')}\n"
                status_msg += f"   • Processing Time: {processing_stats.get('processing_time', 0):.3f}s"
                
                return (
                    process_result.get('output_image'),
                    comparison_viz,
                    displacement_viz,
                    status_msg,
                    processing_stats
                )
            else:
                return None, None, None, f"❌ Processing failed: {process_result['message']}", {}
                
        except Exception as e:
            return None, None, None, f"❌ Error during processing: {str(e)}", {}
    
    def handle_create_analysis(self, analysis_type: str) -> Tuple[Optional[str], str]:
        """
        Handle analysis generation.
        
        Args:
            analysis_type: Type of analysis to generate
            
        Returns:
            Tuple of (visualization_path, summary_text)
        """
        try:
            if analysis_type == "Grid Statistics":
                viz_path = self._create_grid_statistics_analysis()
                summary = self._generate_grid_analysis_summary()
            elif analysis_type == "Displacement Analysis":
                viz_path = self._create_displacement_analysis()
                summary = self._generate_displacement_analysis_summary()
            elif analysis_type == "Processing Flow":
                viz_path = self._create_processing_flow_analysis()
                summary = self._generate_processing_flow_summary()
            elif analysis_type == "Quality Assessment":
                viz_path = self._create_quality_assessment_analysis()
                summary = self._generate_quality_assessment_summary()
            else:
                return None, "❌ Unknown analysis type"
            
            return viz_path, summary
            
        except Exception as e:
            return None, f"❌ Error generating analysis: {str(e)}"
    
    def handle_export_grid_data(self) -> Tuple[Optional[str], str]:
        """
        Handle grid data export in various formats.
        
        Returns:
            Tuple of (export_file_path, status_message)
        """
        try:
            if not self.processing_state['grid_setup']:
                return None, "⚠️ No grid data to export. Please setup grid first."
            
            # Create export package
            export_files = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = tempfile.mkdtemp(prefix="gdc_export_")
            
            # Export original grids if available
            if (self.processor.dx_grid is not None and 
                self.processor.dy_grid is not None):
                
                # CSV format
                dx_orig_csv = os.path.join(temp_dir, f"original_dx_{timestamp}.csv")
                dy_orig_csv = os.path.join(temp_dir, f"original_dy_{timestamp}.csv")
                
                np.savetxt(dx_orig_csv, self.processor.dx_grid, delimiter=',', fmt='%.6f')
                np.savetxt(dy_orig_csv, self.processor.dy_grid, delimiter=',', fmt='%.6f')
                export_files.extend([dx_orig_csv, dy_orig_csv])
                
                # GDC format
                dx_orig_gdc = self.processor.grid_processor.grid_2d_to_gdc_format(
                    self.processor.dx_grid, 'dx'
                )
                dy_orig_gdc = self.processor.grid_processor.grid_2d_to_gdc_format(
                    self.processor.dy_grid, 'dy'
                )
                
                dx_orig_txt = os.path.join(temp_dir, f"original_dx_{timestamp}.txt")
                dy_orig_txt = os.path.join(temp_dir, f"original_dy_{timestamp}.txt")
                
                with open(dx_orig_txt, 'w') as f:
                    f.write(dx_orig_gdc)
                with open(dy_orig_txt, 'w') as f:
                    f.write(dy_orig_gdc)
                    
                export_files.extend([dx_orig_txt, dy_orig_txt])
            
            # Export interpolated grids if available
            if (self.processor.interpolated_dx is not None and 
                self.processor.interpolated_dy is not None):
                
                # CSV format
                dx_interp_csv = os.path.join(temp_dir, f"interpolated_dx_{timestamp}.csv")
                dy_interp_csv = os.path.join(temp_dir, f"interpolated_dy_{timestamp}.csv")
                
                np.savetxt(dx_interp_csv, self.processor.interpolated_dx, delimiter=',', fmt='%.6f')
                np.savetxt(dy_interp_csv, self.processor.interpolated_dy, delimiter=',', fmt='%.6f')
                export_files.extend([dx_interp_csv, dy_interp_csv])
                
                # GDC format
                dx_interp_gdc = self.processor.grid_processor.grid_2d_to_gdc_format(
                    self.processor.interpolated_dx, 'dx'
                )
                dy_interp_gdc = self.processor.grid_processor.grid_2d_to_gdc_format(
                    self.processor.interpolated_dy, 'dy'
                )
                
                dx_interp_txt = os.path.join(temp_dir, f"interpolated_dx_{timestamp}.txt")
                dy_interp_txt = os.path.join(temp_dir, f"interpolated_dy_{timestamp}.txt")
                combined_txt = os.path.join(temp_dir, f"combined_grids_{timestamp}.txt")
                
                with open(dx_interp_txt, 'w') as f:
                    f.write(dx_interp_gdc)
                with open(dy_interp_txt, 'w') as f:
                    f.write(dy_interp_gdc)
                with open(combined_txt, 'w') as f:
                    f.write(f"# Combined GDC Export - {timestamp}\n")
                    f.write(f"# DX Grid ({self.processor.interpolated_dx.shape[0]}x{self.processor.interpolated_dx.shape[1]})\n")
                    f.write(dx_interp_gdc + '\n\n')
                    f.write(f"# DY Grid ({self.processor.interpolated_dy.shape[0]}x{self.processor.interpolated_dy.shape[1]})\n")
                    f.write(dy_interp_gdc)
                    
                export_files.extend([dx_interp_txt, dy_interp_txt, combined_txt])
            
            # Create metadata file
            metadata = self._create_export_metadata()
            metadata_file = os.path.join(temp_dir, f"export_metadata_{timestamp}.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            export_files.append(metadata_file)
            
            # Create zip file
            zip_path = os.path.join(temp_dir, f"gdc_export_{timestamp}.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in export_files:
                    if os.path.exists(file_path):
                        zipf.write(file_path, os.path.basename(file_path))
            
            status_msg = f"✅ **Export Successful!**\n\n"
            status_msg += f"📦 **Package Contents:**\n"
            status_msg += f"   • {len(export_files)} files exported\n"
            status_msg += f"   • Original grids (CSV + GDC format)\n"
            status_msg += f"   • Interpolated grids (CSV + GDC format)\n"
            status_msg += f"   • Combined GDC file\n"
            status_msg += f"   • Export metadata\n\n"
            status_msg += f"📁 **Archive:** gdc_export_{timestamp}.zip"
            
            return zip_path, status_msg
            
        except Exception as e:
            return None, f"❌ Export failed: {str(e)}"
    
    def handle_generate_sample_image(self, pattern: str, width: int, height: int) -> np.ndarray:
        """
        Handle sample image generation.
        
        Args:
            pattern: Pattern type to generate
            width: Image width
            height: Image height
            
        Returns:
            Generated sample image
        """
        try:
            return self.processor.create_sample_image(width, height, pattern)
        except Exception as e:
            print(f"Error generating sample image: {e}")
            # Return a simple default image
            return np.zeros((height, width, 3), dtype=np.uint8)
    
    def update_expected_elements(self, rows: int, cols: int) -> str:
        """
        Update expected elements display.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            
        Returns:
            Formatted string showing expected elements
        """
        try:
            if rows and cols and rows > 0 and cols > 0:
                total_grid_elements = int(rows * cols)
                total_elements = total_grid_elements * 2  # DX + DY
                return f"{total_elements} total ({total_grid_elements} DX + {total_grid_elements} DY)"
            return "Invalid dimensions"
        except (TypeError, ValueError):
            return "Invalid dimensions"
    
    def update_interpolation_factor(self, orig_rows: int, orig_cols: int, 
                                  target_rows: int, target_cols: int) -> str:
        """
        Calculate and format interpolation factor.
        
        Args:
            orig_rows: Original grid rows
            orig_cols: Original grid columns
            target_rows: Target grid rows
            target_cols: Target grid columns
            
        Returns:
            Formatted interpolation factor string
        """
        try:
            if orig_rows and orig_cols and target_rows and target_cols:
                factor_x = target_cols / orig_cols
                factor_y = target_rows / orig_rows
                avg_factor = (factor_x + factor_y) / 2
                return f"{avg_factor:.1f}×"
        except (TypeError, ZeroDivisionError, ValueError):
            pass
        return "N/A"
    
    def generate_traditional_gdc_sample(self, rows: int, cols: int) -> str:
        """
        Generate sample GDC data in traditional format.
        
        Args:
            rows: Number of grid rows
            cols: Number of grid columns
            
        Returns:
            Sample GDC data string
        """
        try:
            lines = []
            lines.append(f"# Sample GDC data for {rows}×{cols} grid")
            lines.append(f"# Generated for testing purposes")
            lines.append(f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")
            
            # Generate DX values
            for i in range(rows * cols):
                # Simple pattern: create some displacement
                dx_value = int(np.sin(i * 0.5) * 50 + np.random.randint(-10, 11))
                lines.append(f"yuv_gdc_grid_dx_0_{i}    {dx_value}")
            
            lines.append("")
            
            # Generate DY values
            for i in range(rows * cols):
                # Simple pattern: create some displacement
                dy_value = int(np.cos(i * 0.3) * 30 + np.random.randint(-5, 6))
                lines.append(f"yuv_gdc_grid_dy_0_{i}    {dy_value}")
            
            lines.append("")
            lines.append(f"# End of sample data - {rows * cols * 2} total elements")
            
            return "\n".join(lines)
            
        except Exception as e:
            print(f"Error generating sample data: {e}")
            return f"# Error generating sample data: {str(e)}"
    
    # Private helper methods
    
    def _create_preview_data(self) -> Optional[list]:
        """Create preview data for the imported GDC data."""
        try:
            if not self.processor.gdc_data:
                return None
            
            # Extract DX and DY data for preview
            dx_data = {}
            dy_data = {}
            
            for name, value in self.processor.gdc_data.items():
                if 'dx' in name:
                    # Extract index from name
                    import re
                    match = re.search(r'_(\d+)$', name)
                    if match:
                        index = int(match.group(1))
                        dx_data[index] = value
                elif 'dy' in name:
                    # Extract index from name
                    import re
                    match = re.search(r'_(\d+)$', name)
                    if match:
                        index = int(match.group(1))
                        dy_data[index] = value
            
            # Create preview list
            preview_data = []
            max_index = max(max(dx_data.keys(), default=-1), max(dy_data.keys(), default=-1))
            
            # Show first 10 elements for preview
            for i in range(min(10, max_index + 1)):
                dx_val = dx_data.get(i, 'N/A')
                dy_val = dy_data.get(i, 'N/A')
                preview_data.append([i, dx_val, dy_val])
            
            return preview_data
            
        except Exception as e:
            print(f"Error creating preview data: {e}")
            return None
    
    def _create_grid_statistics_analysis(self) -> Optional[str]:
        """Create grid statistics visualization."""
        try:
            if not self.processing_state['grid_setup']:
                return None
            
            # Get comprehensive statistics
            statistics = self.processor._compute_comprehensive_grid_statistics()
            
            # Use visualizer to create statistics plot
            return self.visualizer.create_grid_statistics_plot(statistics)
            
        except Exception as e:
            print(f"Error creating grid statistics analysis: {e}")
            return None
    
    def _create_displacement_analysis(self) -> Optional[str]:
        """Create displacement analysis visualization."""
        try:
            if (self.processor.dx_grid is None or 
                self.processor.dy_grid is None):
                return None
            
            return self.visualizer.create_displacement_analysis(
                self.processor.dx_grid, 
                self.processor.dy_grid, 
                self.processor.scale_factor
            )
            
        except Exception as e:
            print(f"Error creating displacement analysis: {e}")
            return None
    
    def _create_processing_flow_analysis(self) -> Optional[str]:
        """Create processing flow visualization."""
        try:
            return self.visualizer.create_processing_flow_diagram(
                self.processor.processing_history
            )
            
        except Exception as e:
            print(f"Error creating processing flow analysis: {e}")
            return None
    
    def _create_quality_assessment_analysis(self) -> Optional[str]:
        """Create quality assessment visualization."""
        try:
            # This would typically analyze image quality metrics
            # For now, return processing flow as placeholder
            return self._create_processing_flow_analysis()
            
        except Exception as e:
            print(f"Error creating quality assessment analysis: {e}")
            return None
    
    def _generate_grid_analysis_summary(self) -> str:
        """Generate grid analysis summary."""
        summary = "📊 **Grid Analysis Summary**\n"
        summary += "=" * 40 + "\n\n"
        
        if self.processor.gdc_data:
            summary += f"📁 **Data Overview:**\n"
            summary += f"   • Total GDC Elements: {len(self.processor.gdc_data)}\n"
            
            dx_count = sum(1 for name in self.processor.gdc_data.keys() if 'dx' in name)
            dy_count = sum(1 for name in self.processor.gdc_data.keys() if 'dy' in name)
            summary += f"   • DX Elements: {dx_count}\n"
            summary += f"   • DY Elements: {dy_count}\n\n"
            
            # Grid statistics
            if self.processor.dx_grid is not None and self.processor.dy_grid is not None:
                summary += f"📐 **Grid Properties:**\n"
                summary += f"   • Original Shape: {self.processor.dx_grid.shape}\n"
                
                if self.processor.interpolated_dx is not None:
                    summary += f"   • Interpolated Shape: {self.processor.interpolated_dx.shape}\n"
                    
                # Statistical analysis
                stats = self.processor._compute_comprehensive_grid_statistics()
                for grid_type, grid_stats in stats.items():
                    if isinstance(grid_stats, dict):
                        grid_name = grid_type.replace('_', ' ').title()
                        summary += f"\n   **{grid_name}:**\n"
                        summary += f"     ├─ Range: {grid_stats.get('min', 0):.1f} to {grid_stats.get('max', 0):.1f}\n"
                        summary += f"     ├─ Mean: {grid_stats.get('mean', 0):.2f}\n"
                        summary += f"     ├─ Std Dev: {grid_stats.get('std', 0):.2f}\n"
                        summary += f"     └─ Total Elements: {grid_stats.get('total_elements', 0)}\n"
        else:
            summary += "⚠️ No grid data available for analysis\n"
        
        return summary
    
    def _generate_displacement_analysis_summary(self) -> str:
        """Generate displacement analysis summary."""
        summary = "🎯 **Displacement Analysis Summary**\n"
        summary += "=" * 40 + "\n\n"
        
        if self.processing_state['grid_setup']:
            summary += f"📐 **Displacement Field Analysis:**\n"
            
            if self.processor.dx_grid is not None and self.processor.dy_grid is not None:
                # Calculate displacement statistics
                dx_magnitude = np.abs(self.processor.dx_grid)
                dy_magnitude = np.abs(self.processor.dy_grid)
                total_magnitude = np.sqrt(self.processor.dx_grid**2 + self.processor.dy_grid**2)
                
                summary += f"   • Grid Shape: {self.processor.dx_grid.shape}\n"
                summary += f"   • Max DX Displacement: {np.max(dx_magnitude):.1f}\n"
                summary += f"   • Max DY Displacement: {np.max(dy_magnitude):.1f}\n"
                summary += f"   • Max Total Displacement: {np.max(total_magnitude):.1f}\n"
                summary += f"   • Mean Total Displacement: {np.mean(total_magnitude):.2f}\n"
                summary += f"   • Displacement Std Dev: {np.std(total_magnitude):.2f}\n\n"
                
                # Scale factor effects
                if hasattr(self.processor, 'scale_factor'):
                    summary += f"⚙️ **Scale Factor Analysis:**\n"
                    summary += f"   • Current Scale Factor: {self.processor.scale_factor}\n"
                    summary += f"   • Effective Max Displacement: {np.max(total_magnitude) * self.processor.scale_factor:.1f}\n"
                    summary += f"   • Effective Mean Displacement: {np.mean(total_magnitude) * self.processor.scale_factor:.2f}\n\n"
                
                # Quality indicators
                summary += f"🏆 **Quality Indicators:**\n"
                
                # Count significant displacements
                significant_threshold = np.mean(total_magnitude) + np.std(total_magnitude)
                significant_count = np.sum(total_magnitude > significant_threshold)
                summary += f"   • Significant Displacements: {significant_count}/{total_magnitude.size} ({100*significant_count/total_magnitude.size:.1f}%)\n"
                
                # Check for potential issues
                if np.max(total_magnitude) > 100:
                    summary += f"   • ⚠️ High displacement values detected\n"
                if np.std(total_magnitude) > np.mean(total_magnitude):
                    summary += f"   • ⚠️ High displacement variability\n"
                
                # Out-of-bounds estimation (simplified)
                if self.processor.image_shape:
                    img_h, img_w = self.processor.image_shape
                    max_displacement = np.max(total_magnitude) * self.processor.scale_factor
                    if max_displacement > min(img_h, img_w) * 0.1:
                        summary += f"   • ⚠️ Displacement may cause boundary issues\n"
                
            else:
                summary += "   • No displacement grids available\n"
        else:
            summary += "⚠️ Grid not setup - no displacement analysis available\n"
        
        return summary
    
    def _generate_processing_flow_summary(self) -> str:
        """Generate processing flow summary."""
        summary = "📋 **Processing Flow Summary**\n"
        summary += "=" * 40 + "\n\n"
        
        # Use processor's built-in summary
        processor_summary = self.processor.get_processing_summary()
        
        # Add processing state information
        summary += "🔄 **Pipeline Status:**\n"
        for stage, status in self.processing_state.items():
            icon = "✅" if status else "❌"
            stage_name = stage.replace('_', ' ').title()
            summary += f"   {icon} {stage_name}: {'Complete' if status else 'Pending'}\n"
        
        summary += "\n" + processor_summary
        
        # Add processing history if available
        if self.processor.processing_history:
            summary += f"\n\n📝 **Processing History:**\n"
            for i, entry in enumerate(self.processor.processing_history[-5:], 1):  # Last 5 entries
                status_icon = "✅" if entry['status'] == 'success' else "❌"
                summary += f"   {i}. {status_icon} {entry['operation']}: {entry['message']}\n"
                summary += f"      └─ {entry['timestamp']}\n"
        
        return summary
    
    def _generate_quality_assessment_summary(self) -> str:
        """Generate quality assessment summary."""
        summary = "🏆 **Quality Assessment Summary**\n"
        summary += "=" * 40 + "\n\n"
        
        # Processing pipeline assessment
        summary += "📋 **Processing Pipeline Quality:**\n"
        
        total_stages = len(self.processing_state)
        completed_stages = sum(self.processing_state.values())
        completion_rate = (completed_stages / total_stages) * 100
        
        summary += f"   • Pipeline Completion: {completed_stages}/{total_stages} ({completion_rate:.0f}%)\n"
        
        for stage, status in self.processing_state.items():
            icon = "✅" if status else "❌"
            stage_name = stage.replace('_', ' ').title()
            summary += f"   {icon} {stage_name}: {'✓' if status else '✗'}\n"
        
        # Data quality assessment
        if self.processing_state['gdc_loaded']:
            summary += f"\n📊 **Data Quality Assessment:**\n"
            
            if self.processor.gdc_data:
                dx_count = sum(1 for name in self.processor.gdc_data.keys() if 'dx' in name)
                dy_count = sum(1 for name in self.processor.gdc_data.keys() if 'dy' in name)
                
                summary += f"   • Data Completeness: {'✅' if dx_count == dy_count else '⚠️'} DX/DY Balance\n"
                summary += f"   • Element Count: {len(self.processor.gdc_data)} total\n"
                
                # Value range assessment
                values = list(self.processor.gdc_data.values())
                value_range = max(values) - min(values)
                summary += f"   • Value Range: {min(values)} to {max(values)} (span: {value_range})\n"
                
                # Data consistency check
                if value_range > 1000:
                    summary += f"   • ⚠️ Large value range detected\n"
                else:
                    summary += f"   • ✅ Reasonable value range\n"
        
        # Grid quality assessment
        if self.processing_state['grid_setup']:
            summary += f"\n🌐 **Grid Quality Assessment:**\n"
            
            if (self.processor.dx_grid is not None and 
                self.processor.dy_grid is not None):
                
                # Grid dimension assessment
                orig_shape = self.processor.dx_grid.shape
                summary += f"   • Original Grid: {orig_shape[0]}×{orig_shape[1]} ✅\n"
                
                if (self.processor.interpolated_dx is not None and 
                    self.processor.interpolated_dy is not None):
                    interp_shape = self.processor.interpolated_dx.shape
                    scale_factor = (interp_shape[0] * interp_shape[1]) / (orig_shape[0] * orig_shape[1])
                    summary += f"   • Interpolated Grid: {interp_shape[0]}×{interp_shape[1]} (scale: {scale_factor:.1f}×) ✅\n"
                    
                    # Interpolation quality indicators
                    if scale_factor >= 4:
                        summary += f"   • ✅ High resolution interpolation\n"
                    elif scale_factor >= 2:
                        summary += f"   • ✅ Good interpolation resolution\n"
                    else:
                        summary += f"   • ⚠️ Low interpolation factor\n"
        
        # Image processing quality
        if self.processing_state['image_processed']:
            summary += f"\n🖼️ **Image Processing Quality:**\n"
            summary += f"   • ✅ Image successfully processed\n"
            
            if hasattr(self.processor, 'scale_factor'):
                summary += f"   • Scale Factor: {self.processor.scale_factor} "
                if 0.01 <= self.processor.scale_factor <= 0.5:
                    summary += "✅ (Appropriate)\n"
                else:
                    summary += "⚠️ (May cause artifacts)\n"
            
            if self.processor.current_image is not None:
                img_shape = self.processor.current_image.shape
                summary += f"   • Image Dimensions: {img_shape[1]}×{img_shape[0]} ✅\n"
        
        # Overall quality score
        summary += f"\n🎯 **Overall Quality Score:**\n"
        
        quality_score = 0
        max_score = 0
        
        # Pipeline completion (40% weight)
        quality_score += completion_rate * 0.4
        max_score += 40
        
        # Data quality (30% weight)
        if self.processing_state['gdc_loaded']:
            if self.processor.gdc_data:
                dx_count = sum(1 for name in self.processor.gdc_data.keys() if 'dx' in name)
                dy_count = sum(1 for name in self.processor.gdc_data.keys() if 'dy' in name)
                if dx_count == dy_count and dx_count > 0:
                    quality_score += 30
            max_score += 30
        
        # Grid quality (20% weight)
        if self.processing_state['grid_setup']:
            quality_score += 20
            max_score += 20
        
        # Processing success (10% weight)
        if self.processing_state['image_processed']:
            quality_score += 10
            max_score += 10
        
        final_score = (quality_score / max_score * 100) if max_score > 0 else 0
        
        if final_score >= 90:
            grade = "A+ (Excellent)"
            icon = "🏆"
        elif final_score >= 80:
            grade = "A (Very Good)"
            icon = "🥇"
        elif final_score >= 70:
            grade = "B (Good)"
            icon = "🥈"
        elif final_score >= 60:
            grade = "C (Fair)"
            icon = "🥉"
        else:
            grade = "D (Needs Improvement)"
            icon = "⚠️"
        
        summary += f"   {icon} **Score: {final_score:.0f}/100 - {grade}**\n"
        
        # Recommendations
        summary += f"\n💡 **Recommendations:**\n"
        
        if not self.processing_state['gdc_loaded']:
            summary += f"   • Import GDC data to begin processing\n"
        elif not self.processing_state['grid_setup']:
            summary += f"   • Setup and interpolate grid for better results\n"
        elif not self.processing_state['image_processed']:
            summary += f"   • Process an image to complete the workflow\n"
        else:
            summary += f"   • ✅ Processing pipeline complete!\n"
            summary += f"   • Consider experimenting with different scale factors\n"
            summary += f"   • Try different interpolation methods for comparison\n"
        
        return summary
    
    def _create_export_metadata(self) -> Dict[str, Any]:
        """Create metadata for export package."""
        metadata = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
                "tool": "GDC Image Remapping Suite"
            },
            "processing_state": self.processing_state.copy(),
            "grid_info": {},
            "statistics": {}
        }
        
        # Add grid information
        if self.processor.original_shape:
            metadata["grid_info"]["original_shape"] = self.processor.original_shape
        if self.processor.target_shape:
            metadata["grid_info"]["target_shape"] = self.processor.target_shape
        if hasattr(self.processor, 'scale_factor'):
            metadata["grid_info"]["scale_factor"] = self.processor.scale_factor
        
        # Add statistics
        if self.processing_state['grid_setup']:
            try:
                metadata["statistics"] = self.processor._compute_comprehensive_grid_statistics()
            except:
                pass
        
        # Add processing history
        if self.processor.processing_history:
            metadata["processing_history"] = self.processor.processing_history[-10:]  # Last 10 entries
        
        return metadata
#!/usr/bin/env python3
"""
Barrel Distortion Simulator & Corrector
Main application launcher

This application provides a comprehensive tool for:
- Simulating barrel distortion effects
- Testing advanced correction algorithms
- Visualizing distortion grids and heatmaps
- Exporting data in various formats (CSV, GDC)
- Quality assessment and validation

Author: Balaji R
License: MIT
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'numpy',
        'opencv-python',
        'matplotlib',
        'gradio',
        'scipy',
        'pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'pillow':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Main application entry point"""
    print("🔍 Barrel Distortion Simulator & Corrector")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Missing dependencies. Please install required packages.")
        sys.exit(1)
    
    print("✅ All dependencies found")
    print("🚀 Starting application...")
    
    try:
        from barrel_distortion_interface import create_gradio_interface
        
        # Create and launch the interface
        interface = create_gradio_interface()
        
        print("🌐 Launching web interface...")
        print("📝 Use Ctrl+C to stop the server")
        
        # Launch with auto port selection
        interface.launch(
            share=False,
            server_name="0.0.0.0",
            show_error=True,
            quiet=False
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all module files are in the same directory:")
        print("  - barrel_distortion_simulator.py")
        print("  - barrel_distortion_corrector.py") 
        print("  - barrel_distortion_visualization.py")
        print("  - barrel_distortion_processing.py")
        print("  - barrel_distortion_interface.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
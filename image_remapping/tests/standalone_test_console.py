#!/usr/bin/env python3
"""
Image Remapping Suite - Standalone Testing Console (Clean Version)

Independent console application for comprehensive testing and validation
of inverse mapping algorithms. Completely separate from the UI application.

Usage:
    python test_console.py              # Interactive menu
    python test_console.py --quick      # Quick test
    python test_console.py --stress     # Stress test
    python test_console.py --compare    # Method comparison

Author: Balaji R
License: MIT
"""

import sys
import os
import argparse
import time
import json
import math
from datetime import datetime
from collections import namedtuple

# --- Constants and Configuration ---

# Adjust sys.path to allow imports from the project root
try:
    # Get the directory where the current script (test_console.py) is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # The project root is two levels up from the current script (test/ -> ImageRemappingSuite/)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir)) 
    
    # Add the project root to sys.path if it's not already there
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT) # Use insert(0) to give it higher priority
    
    # Now import your existing modules, they should be found relative to PROJECT_ROOT
    from application.processor import processor
    from config.settings import (
        DEFAULT_K1, DEFAULT_K2, DEFAULT_K3, DEFAULT_P1, DEFAULT_P2,
        VALIDATION_RANGES
    )
    IMPORTS_AVAILABLE = True
    print("✅ Successfully imported Image Remapping Suite modules")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print(f"Attempted to add '{PROJECT_ROOT}' to sys.path.")
    print("Please ensure 'application' and 'config' directories are direct children of the project root.")
    print("Run this script from the 'test/' directory.")
    IMPORTS_AVAILABLE = False
    processor = None # Ensure processor is None if import fails

# Define constants for clarity and easy modification
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
GRID_ROWS = 9
GRID_COLS = 7
DEFAULT_TEST_POINTS = [
    (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2),  # Center
    (200, 200),  # Top-left area
    (IMAGE_WIDTH - 200, 200),  # Top-right area
    (200, IMAGE_HEIGHT - 200),  # Bottom-left area
    (IMAGE_WIDTH - 200, IMAGE_HEIGHT - 200),  # Bottom-right area
]

# Using a named tuple for better readability of distortion parameters
DistortionParams = namedtuple('DistortionParams', ['k1', 'k2', 'k3', 'p1', 'p2'])

# Define test parameter sets
TEST_PRESETS = {
    "quick_test": {
        "name": "Quick Validation Test",
        "description": "Standard validation with moderate barrel distortion",
        "params": DistortionParams(k1=-0.2, k2=0.05, k3=0.0, p1=0.0, p2=0.0)
    },
    "stress_test": {
        "name": "Stress Test",
        "description": "Extreme parameters to test algorithm limits",
        "params": DistortionParams(k1=-0.45, k2=0.15, k3=-0.05, p1=0.02, p2=0.01)
    }
}

COMPARISON_TEST_CASES = [
    {"name": "Mild Barrel", "params": DistortionParams(k1=-0.1, k2=0.01, k3=0.0, p1=0.0, p2=0.0)},
    {"name": "Moderate Barrel", "params": DistortionParams(k1=-0.2, k2=0.05, k3=0.0, p1=0.0, p2=0.0)},
    {"name": "Severe Barrel", "params": DistortionParams(k1=-0.35, k2=0.1, k3=-0.02, p1=0.0, p2=0.0)},
    {"name": "Pincushion", "params": DistortionParams(k1=0.15, k2=-0.03, k3=0.0, p1=0.0, p2=0.0)},
]

# Error grading thresholds
GRADE_THRESHOLDS = [
    (0.1, 'A', 'Excellent'),
    (0.5, 'B', 'Good'),
    (1.0, 'C', 'Fair'),
    (float('inf'), 'D', 'Poor'),
]

HELP_TEXT = """
🧪 TEST TYPES:

1. Quick Test - Standard validation with moderate barrel distortion
2. Stress Test - Extreme parameters to test algorithm limits
3. Method Comparison - Compare algorithms across distortion types
4. Custom Test - Test with your own parameters

📊 QUALITY METRICS:

• Round-trip Error:
  - < 0.1 pixels = Excellent
  - < 0.5 pixels = Good
  - < 1.0 pixels = Acceptable
  - > 1.0 pixels = Poor

🔧 ALGORITHMS TESTED:

• Iterative: Most accurate, uses Newton-Raphson
• Analytical: Fastest for simple K1-only distortion
• Polynomial: Good balance for moderate distortions

📝 DISTORTION PARAMETERS:

• K1: Primary radial (negative=barrel, positive=pincushion)
• K2, K3: Higher-order radial corrections
• P1, P2: Tangential distortion corrections

💡 USAGE TIPS:

• Start with Quick Test to verify basic functionality
• Use Stress Test to find algorithm limits
• Use Custom Test for your specific parameters
• Export results for documentation

🔗 COMMAND LINE:

python test_console.py --quick      # Quick test
python test_console.py --stress     # Stress test
python test_console.py --compare    # Method comparison
"""

class SimpleTestRunner:
    """
    Simple, robust test runner for inverse mapping validation
    """

    def __init__(self):
        if not IMPORTS_AVAILABLE or processor is None:
            print("❌ Cannot initialize TestRunner - missing dependencies.")
            sys.exit(1)

        self.processor = processor
        self.test_results = {}
        self.session_start = datetime.now()

        self._print_header()

    def _print_header(self):
        """Prints the application header."""
        print("\n🔍 Image Remapping Suite - Testing Console")
        print("=" * 60)
        print(f"Session started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

    def show_main_menu(self):
        """Display the main interactive menu and handle user choices."""
        menu_options = {
            "1": (self.run_preset_test, TEST_PRESETS["quick_test"]),
            "2": (self.run_preset_test, TEST_PRESETS["stress_test"]),
            "3": (self.run_method_comparison, None),
            "4": (self.run_custom_test, None),
            "5": (self.show_results_summary, None),
            "6": (self.export_results, None),
            "7": (self.show_help, None),
            "0": (self.exit_application, None)
        }

        while True:
            self._print_menu()
            choice = input("Enter your choice (0-7): ").strip()

            try:
                if choice in menu_options:
                    action, args = menu_options[choice]
                    if args:
                        action(args)
                    else:
                        action()
                    if choice == "0": # Exit condition
                        break
                else:
                    print("❌ Invalid choice. Please enter 0-7.")

            except KeyboardInterrupt:
                print("\n\n👋 Test interrupted by user")
                break
            except Exception as e:
                print(f"❌ An unexpected error occurred: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for debugging
                input("Press Enter to continue...")

    def _print_menu(self):
        """Helper to print the menu options."""
        print("\n" + "="*50)
        print("🧪 TESTING MENU")
        print("="*50)
        print("1. 🚀 Quick Validation Test")
        print("2. 🔥 Stress Test")
        print("3. ⚖️ Method Comparison")
        print("4. 🎯 Custom Test")
        print("5. 📊 View Results")
        print("6. 💾 Export Results")
        print("7. ❓ Help")
        print("0. 🚪 Exit")
        print("="*50)

    def run_preset_test(self, preset_info: dict):
        """Runs a predefined test (quick or stress) with specific parameters."""
        test_id = preset_info['name'].lower().replace(' ', '_')
        params = preset_info['params']
        print(f"\n{preset_info['name']}...")
        print("-" * 40)

        self._set_processor_parameters(params)
        print(f"Parameters: K1={params.k1}, K2={params.k2}, K3={params.k3}, P1={params.p1}, P2={params.p2}")

        results = self._run_round_trip_tests(preset_info['name'])

        self.test_results[test_id] = {
            'timestamp': datetime.now().isoformat(),
            'parameters': params._asdict(), # Convert namedtuple to dict for storage
            'results': results
        }

        self._print_simple_summary(results, preset_info['name'])
        input("\nPress Enter to continue...")

    def run_method_comparison(self):
        """Runs a comparison of algorithms across different distortion types."""
        print("\n⚖️ Running Method Comparison...")
        print("-" * 35)

        comparison_results = {}

        for i, case in enumerate(COMPARISON_TEST_CASES, 1):
            print(f"\n[{i}/{len(COMPARISON_TEST_CASES)}] Testing: {case['name']}")
            params = case['params']

            self._set_processor_parameters(params)

            results = self._run_round_trip_tests(case['name'])

            comparison_results[case['name']] = {
                'parameters': params._asdict(),
                'results': results
            }

            print(f"  ✅ {case['name']} complete")

        self.test_results["method_comparison"] = {
            'timestamp': datetime.now().isoformat(),
            'results': comparison_results
        }

        self._print_comparison_summary(comparison_results)
        input("\nPress Enter to continue...")

    def run_custom_test(self):
        """Runs a test with user-defined parameters."""
        print("\n🎯 Custom Parameter Test")
        print("-" * 25)

        try:
            print("Enter distortion parameters (press Enter for defaults):")

            k1 = self._get_float_input("K1 (Primary Radial)", DEFAULT_K1)
            k2 = self._get_float_input("K2 (Secondary Radial)", DEFAULT_K2)
            k3 = self._get_float_input("K3 (Tertiary Radial)", DEFAULT_K3)
            p1 = self._get_float_input("P1 (Tangential)", DEFAULT_P1)
            p2 = self._get_float_input("P2 (Tangential)", DEFAULT_P2)
            custom_params = DistortionParams(k1, k2, k3, p1, p2)

            test_name = input("Test name (optional): ").strip()
            if not test_name:
                test_name = f"Custom_{datetime.now().strftime('%H%M%S')}"

            print(f"\nRunning test '{test_name}':")
            print(f"K1={k1:.4f}, K2={k2:.4f}, K3={k3:.4f}, P1={p1:.4f}, P2={p2:.4f}")

            self._set_processor_parameters(custom_params)

            results = self._run_round_trip_tests(test_name)

            self.test_results[f"custom_{test_name}"] = {
                'timestamp': datetime.now().isoformat(),
                'parameters': custom_params._asdict(),
                'results': results
            }

            self._print_simple_summary(results, test_name)

        except KeyboardInterrupt:
            print("\n❌ Custom test cancelled")
        except ValueError as ve:
            print(f"❌ Input error: {ve}")
        except Exception as e:
            print(f"❌ Error in custom test: {e}")

        input("\nPress Enter to continue...")

    def show_results_summary(self):
        """Shows a summary of all recorded test results."""
        print("\n📊 Test Results Summary")
        print("=" * 40)

        if not self.test_results:
            print("❌ No test results available yet.")
            input("\nPress Enter to continue...")
            return

        for test_key, test_data in self.test_results.items():
            display_name = test_key.replace('_', ' ').title()
            print(f"\n🔸 {display_name}")
            print(f"  Time: {test_data['timestamp']}")

            params = test_data.get('parameters')
            if params:
                print(f"  Params: K1={params.get('k1', 0):.3f}, K2={params.get('k2', 0):.3f}")

            results = test_data.get('results', {})
            if 'best_method' in results:
                print(f"  Best Method: {results['best_method'].title()}")
                if 'best_error' in results and results['best_error'] is not None:
                    print(f"  Best Error: {results['best_error']:.4f} px")
                print(f"  Assessment: {results.get('assessment', 'N/A')}")
            elif 'method_results' in results: # For method comparison summaries
                for method, method_data in results['method_results'].items():
                    if isinstance(method_data, dict) and 'mean_error' in method_data:
                        print(f"    {method.title()}: {method_data['mean_error']:.4f} px")

        input("\nPress Enter to continue...")

    def export_results(self):
        """Exports all test results to a JSON file."""
        print("\n💾 Export Test Results")
        print("-" * 25)

        if not self.test_results:
            print("❌ No test results to export.")
            input("\nPress Enter to continue...")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"

        try:
            export_data = {
                'session_info': {
                    'start_time': self.session_start.isoformat(),
                    'export_time': datetime.now().isoformat(),
                    'total_tests_recorded': len(self.test_results)
                },
                'test_results': self.test_results
            }

            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str) # default=str handles datetime objects

            print(f"✅ Results exported to: {filename}")
            print(f"📊 Tests exported: {len(self.test_results)}")

        except Exception as e:
            print(f"❌ Export failed: {e}")

        input("\nPress Enter to continue...")

    def show_help(self):
        """Displays help information about the console."""
        print("\n❓ Help & Information")
        print("=" * 30)
        print(HELP_TEXT)
        input("\nPress Enter to continue...")

    def exit_application(self):
        """Performs a clean exit with a session summary."""
        duration = datetime.now() - self.session_start

        print("\n👋 Exiting Test Console")
        print("=" * 25)
        print(f"Session time: {duration}")
        print(f"Tests run: {len(self.test_results)}")
        print("\nThank you for using the Testing Console!")
        sys.exit(0) # Explicitly exit after summary

    # --- Helper methods ---

    def _set_processor_parameters(self, params: DistortionParams):
        """Sets the distortion parameters on the processor's simulator."""
        self.processor.simulator.set_parameters(
            IMAGE_WIDTH, IMAGE_HEIGHT, GRID_ROWS, GRID_COLS,
            params.k1, params.k2, params.k3, params.p1, params.p2
        )

    def _run_round_trip_tests(self, test_case_name: str, test_points=None):
        """
        Runs round-trip accuracy tests for all available inverse mapping methods.
        Args:
            test_case_name (str): The name of the current test case.
            test_points (list, optional): A list of (x, y) tuples to test.
                                           Defaults to DEFAULT_TEST_POINTS.
        Returns:
            dict: A dictionary containing the results for each method and overall summary.
        """
        print(f"  Running validation for {test_case_name}...")

        test_points = test_points if test_points is not None else DEFAULT_TEST_POINTS
        methods = {
            'iterative': self.processor.corrector.iterative_inverse_mapping,
            'analytical': self.processor.corrector.analytical_inverse_mapping,
            'polynomial': self.processor.corrector.polynomial_inverse_mapping
        }
        results = {'method_results': {}}

        best_method = 'iterative' # Default best method
        best_error = float('inf')

        for method_name, method_func in methods.items():
            print(f"    Testing {method_name} method...")
            errors = []
            successful_tests = 0

            for x_orig, y_orig in test_points:
                try:
                    # Apply forward distortion
                    x_dist, y_dist = self.processor.simulator.apply_barrel_distortion(x_orig, y_orig)

                    # Apply inverse mapping
                    x_corr, y_corr = method_func(x_dist, y_dist)

                    # Calculate round-trip error
                    error = math.sqrt((x_orig - x_corr)**2 + (y_orig - y_corr)**2)

                    if math.isfinite(error):
                        errors.append(error)
                        successful_tests += 1

                except Exception as e:
                    # Log the specific point failure, but continue with other points
                    print(f"      ⚠️ {method_name} failed at ({x_orig}, {y_orig}): {e}")
                    # Optionally, append a large error or NaN to indicate failure for this point
                    # For simplicity, we just don't count it towards successful_tests for now.
                    continue

            # Calculate statistics for the current method
            if errors:
                mean_error = sum(errors) / len(errors)
                max_error = max(errors)
                min_error = min(errors)

                results['method_results'][method_name] = {
                    'mean_error': mean_error,
                    'max_error': max_error,
                    'min_error': min_error,
                    'successful_tests': successful_tests,
                    'total_tests': len(test_points)
                }

                # Update overall best method
                if mean_error < best_error:
                    best_error = mean_error
                    best_method = method_name

                print(f"      ✅ Mean error: {mean_error:.4f} px ({successful_tests}/{len(test_points)} success)")
            else:
                results['method_results'][method_name] = {
                    'error': 'All tests failed',
                    'successful_tests': 0,
                    'total_tests': len(test_points)
                }
                print(f"      ❌ All tests failed for {method_name}")

        # Overall results for this test run
        results['best_method'] = best_method
        results['best_error'] = best_error if math.isfinite(best_error) else None
        self._assign_grade(results)

        return results

    def _assign_grade(self, results: dict):
        """Assigns a performance grade based on the best error."""
        best_error = results.get('best_error')
        if best_error is None or not math.isfinite(best_error):
            results['grade'] = 'N/A'
            results['assessment'] = 'No successful tests'
            return

        for threshold, grade, assessment in GRADE_THRESHOLDS:
            if best_error < threshold:
                results['grade'] = grade
                results['assessment'] = assessment
                break

    def _get_float_input(self, param_name: str, default_value: float) -> float:
        """Gets validated float input from the user."""
        while True:
            try:
                user_input = input(f"{param_name} [{default_value}]: ").strip()
                if not user_input:
                    return default_value
                return float(user_input)
            except ValueError:
                print("❌ Invalid number. Please enter a numerical value.")

    def _print_simple_summary(self, results: dict, test_name: str):
        """Prints a simplified summary of a single test run."""
        print(f"\n📊 {test_name} Results")
        print("=" * 30)

        method_results = results.get('method_results', {})

        print("\n🎯 Round-Trip Accuracy:")
        for method, data in method_results.items():
            if isinstance(data, dict) and 'mean_error' in data:
                error = data['mean_error']
                success = data['successful_tests']
                total = data['total_tests']

                status_char = "✅" if error < 0.5 else ("⚠️" if error < 1.0 else "❌")
                print(f"  {status_char} {method.title()}: {error:.4f} px ({success}/{total})")
            else:
                print(f"  ❌ {method.title()}: Failed (No data)")

        print(f"\n💡 Summary:")
        print(f"  🏆 Best Method: {results.get('best_method', 'Unknown').title()}")
        print(f"  📊 Grade: {results.get('grade', 'Unknown')}")
        print(f"  🎯 Assessment: {results.get('assessment', 'Unknown')}")

        if results.get('best_error') is not None and math.isfinite(results['best_error']):
            print(f"  📐 Best Error: {results['best_error']:.4f} pixels")

    def _print_comparison_summary(self, comparison_results: dict):
        """Prints a summary for the method comparison test."""
        print(f"\n📊 Method Comparison Summary")
        print("=" * 40)

        for case_name, case_data in comparison_results.items():
            results = case_data.get('results', {})
            best_method = results.get('best_method', 'Unknown')
            best_error = results.get('best_error')
            grade = results.get('grade', 'Unknown')

            print(f"\n🔸 {case_name}:")
            print(f"  Best: {best_method.title()} - Grade: {grade}")

            if best_error is not None and math.isfinite(best_error):
                status_char = "✅" if best_error < 0.5 else ("⚠️" if best_error < 1.0 else "❌")
                print(f"  {status_char} Error: {best_error:.4f} px")
            else:
                print(f"  ❌ No successful tests for this case.")


def main():
    """Main entry point of the console application."""
    parser = argparse.ArgumentParser(description="Image Remapping Suite - Testing Console")
    parser.add_argument('--quick', action='store_true', help='Run quick test')
    parser.add_argument('--stress', action='store_true', help='Run stress test')
    parser.add_argument('--compare', action='store_true', help='Run method comparison')

    args = parser.parse_args()

    if not IMPORTS_AVAILABLE:
        sys.exit(1) # Already printed error, just exit

    test_runner = SimpleTestRunner()

    if args.quick:
        test_runner.run_preset_test(TEST_PRESETS["quick_test"])
    elif args.stress:
        test_runner.run_preset_test(TEST_PRESETS["stress_test"])
    elif args.compare:
        test_runner.run_method_comparison()
    else:
        # Interactive mode
        test_runner.show_main_menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Testing console interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ A fatal error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
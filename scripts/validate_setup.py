#!/usr/bin/env python3
"""Validation script to check FEP-RL-VAE setup."""

import sys
import importlib


def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        if package_name:
            mod = importlib.import_module(module_name, package=package_name)
        else:
            mod = importlib.import_module(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {module_name}: {e}")
        return False


def main():
    """Run validation checks."""
    print("Validating FEP-RL-VAE setup...\n")
    
    all_passed = True
    
    # Check core package imports
    print("Core package imports:")
    all_passed &= check_import("fep_rl_vae")
    all_passed &= check_import("fep_rl_vae.data")
    all_passed &= check_import("fep_rl_vae.utils")
    
    # Check data loader
    print("\nData loader:")
    all_passed &= check_import("fep_rl_vae.data.loader")
    
    # Check utilities
    print("\nUtilities:")
    all_passed &= check_import("fep_rl_vae.utils.logging")
    all_passed &= check_import("fep_rl_vae.utils.plotting")
    
    # Check encoders/decoders (may fail if general_FEP_RL not installed)
    print("\nEncoders/Decoders:")
    encoder_ok = check_import("fep_rl_vae.encoders")
    decoder_ok = check_import("fep_rl_vae.decoders")
    
    if not encoder_ok or not decoder_ok:
        print("\n⚠ Warning: Encoders/decoders require general_FEP_RL")
        print("  Install with: ./scripts/setup_general_fep_rl.sh")
        print("  Or: cd ../active-inference-sim-lab && uv pip install -e .")
    
    # Check dependencies
    print("\nDependencies:")
    deps = ["torch", "torchvision", "numpy", "matplotlib"]
    for dep in deps:
        all_passed &= check_import(dep)
    
    # Summary
    print("\n" + "="*50)
    if all_passed:
        print("✓ All core checks passed!")
        if not (encoder_ok and decoder_ok):
            print("⚠ Note: Install general_FEP_RL for full functionality")
        return 0
    else:
        print("✗ Some checks failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

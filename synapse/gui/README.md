# Synapse GUI Module

This module contains the graphical user interface components for the SynapseClusterEM system.

## Overview

The GUI module provides:
- A user-friendly interface to the SynapseClusterEM system
- Configuration options for analysis parameters
- Visualization tools for synapse data
- Report generation and viewing capabilities

## Key Components

- **synapse_gui.py**: Contains the main GUI implementation using Tkinter
- **assets/**: Contains icons and images used by the GUI
- **__init__.py**: Exports the SynapseGUI class

## Usage

There are two ways to use the GUI:

1. **Using the launcher script**:
   ```
   python run_gui.py
   ```

2. **From Python code**:
   ```python
   import tkinter as tk
   from synapse import SynapseGUI
   
   root = tk.Tk()
   app = SynapseGUI(root)
   root.mainloop()
   ```

## Requirements

The GUI requires:
- Python 3.6 or higher
- Tkinter (included with most Python installations)
- PIL/Pillow for image handling

## Asset Paths

The GUI looks for assets in several locations:
1. `assets/` in the root directory
2. `synapse/gui/assets/` for packaged installations
3. Relative to the script location

If assets are not found, the GUI will fall back to text-based elements. 
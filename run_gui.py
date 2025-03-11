#!/usr/bin/env python
"""
SynapseClusterEM GUI Launcher

This script launches the SynapseClusterEM GUI.
"""

import os
import sys
import tkinter as tk

# Add the current directory to the Python path if needed
if os.path.abspath('.') not in sys.path:
    sys.path.append(os.path.abspath('.'))

# Import the SynapseGUI class
from synapse.gui.synapse_gui import SynapseGUI

def main():
    """Create and run the GUI"""
    root = tk.Tk()
    app = SynapseGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
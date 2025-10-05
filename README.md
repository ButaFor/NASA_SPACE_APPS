# Terra Tools - 3D Earth Visualization

A GPU-accelerated 3D globe viewer with interactive Earth imagery using NASA Earthdata and Natural Earth vectors.

## Features

- 🌍 **Interactive 3D Globe**: Real-time Earth visualization with GPU acceleration
- 🛰️ **NASA Earthdata Integration**: Live satellite imagery from GIBS WMS and Worldview Snapshot API
- 🗺️ **Natural Earth Vectors**: High-quality geographic data for coastlines and countries
- 📊 **Multiple Instruments**: MODIS, MOPITT, and other NASA instruments
- 🎮 **Interactive Controls**: Mouse and keyboard navigation
- 💾 **Local Caching**: Automatic caching for offline viewing
- 🎨 **Modern UI**: Clean, intuitive control panel

## Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenGL-compatible graphics card
- Internet connection (for initial data download)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/terra-tools.git
   cd terra-tools
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python -m gui.sphere_viewer_labels_fix
   ```

## Usage

### Basic Controls

- **Mouse**: 
  - Left click + drag: Rotate globe
  - Right click + drag: Pan view
  - Scroll wheel: Zoom in/out

- **Keyboard Shortcuts:**
  - `A` - Toggle globe animation (slow rotation)
  - `L` - Toggle analysis legend
  - `H` - Show help
  - `ESC` - Exit application

### Data Sources

The application uses several data sources:

- **NASA GIBS WMS**: Real-time satellite imagery
- **Worldview Snapshot API**: Historical and current Earth imagery
- **Natural Earth**: Geographic vector data for coastlines and countries

### Instruments Available

- **MODIS True Color**: Natural-color composite imagery
- **MODIS False Color**: Enhanced vegetation and water detection
- **MODIS Thermal Anomalies**: Fire and thermal detection
- **MOPITT CO**: Carbon monoxide measurements

## File Structure

```
terra-tools/
├── gui/
│   ├── sphere_viewer_labels_fix.py    # Main application
│   └── geodata.py                     # Geographic data handling
├── data/
│   ├── 50m/physical/
│   │   └── ne_50m_coastline.json     # Coastline data
│   ├── ne_50m_admin_0_countries.geojson
│   └── crimea.json
└── requirements.txt
```

## Requirements

- **Python 3.8+**
- **OpenGL 3.3+** support
- **ModernGL** for GPU rendering
- **NumPy** for mathematical operations
- **GLFW** for window management

## Troubleshooting

### Common Issues

1. **"No OpenGL context" error:**
   - Ensure your graphics drivers are up to date
   - Check that your graphics card supports OpenGL 3.3+

2. **"Module not found" errors:**
   - Run `pip install -r requirements.txt` to install all dependencies

3. **Slow performance:**
   - Close other applications to free up GPU memory
   - Reduce image resolution in settings

### Performance Tips

- Use a dedicated graphics card for best performance
- Close unnecessary applications while running
- Ensure stable internet connection for data downloads

## Development

### Building from Source

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
3. Run tests:
   ```bash
   python -m pytest tests/
   ```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **NASA** for Earthdata and satellite imagery
- **Natural Earth** for geographic vector data
- **ModernGL** for GPU rendering capabilities

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section above
- Ensure all dependencies are properly installed

---

**Note**: This application requires a modern graphics card with OpenGL support. Performance may vary depending on your hardware configuration.

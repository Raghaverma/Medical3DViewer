# Medical 3D Viewer

A modern, feature-rich medical imaging application for viewing and analyzing 3D medical data.

## Features

- **3D Visualization**
  - Support for STL and OBJ file formats
  - DICOM image loading and viewing
  - Interactive 3D rendering with VTK
  - Multiple view modes (surface, volume, slice)

- **AI Analysis**
  - Tumor detection using deep learning
  - Anatomical segmentation
  - Landmark detection
  - Confidence scoring

- **Cloud Integration**
  - AWS S3 storage support
  - Firebase integration
  - Secure file upload/download
  - Cloud-based model storage

- **User Interface**
  - Modern, intuitive design
  - Customizable view settings
  - Keyboard shortcuts
  - Progress indicators
  - Error handling and logging

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Medical3DViewer.git
cd Medical3DViewer
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Unix/MacOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the AI model (optional):
```bash
python -m modules.ai_analysis.train_model
```

## Usage

1. Start the application:
```bash
python main.py
```

2. Load medical files:
   - Use File > Open to load STL/OBJ files
   - Use File > Open DICOM to load DICOM images
   - Drag and drop files into the application

3. View and interact:
   - Left mouse: Rotate view
   - Middle mouse: Pan view
   - Right mouse: Zoom
   - Space: Reset view
   - R: Toggle rendering mode
   - S: Save current view

4. AI Analysis:
   - Select a region of interest
   - Click "Analyze" to run tumor detection
   - View confidence scores and predictions

## Project Structure

```
Medical3DViewer/
├── assets/           # Static assets
│   ├── models/      # AI model files
│   ├── icons/       # Application icons
│   └── styles/      # QSS stylesheets
├── modules/         # Core functionality
│   ├── ai_analysis/ # AI-related functionality
│   ├── visualization.py
│   ├── interaction.py
│   ├── model_loader.py
│   ├── dicom_loader.py
│   └── cloud_integration.py
├── ui/             # User interface
│   ├── widgets/    # Custom widgets
│   └── main_window.py
├── utils/          # Utility functions
├── tests/          # Test cases
├── docs/           # Documentation
├── config.py       # Configuration
├── main.py         # Application entry point
└── README.md       # This file
```

## Development

- Python 3.8+
- PyQt5 for UI
- VTK for 3D visualization
- TensorFlow for AI analysis
- AWS SDK and Firebase Admin for cloud features

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- VTK for 3D visualization
- PyQt5 for the UI framework
- TensorFlow for AI capabilities
- AWS and Firebase for cloud services
